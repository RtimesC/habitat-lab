"""QLoRA training entrypoint for mixed NaVIDA VLN and IDS samples."""

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image

try:
    from .navida_data import (
        assistant_answer,
        dataset_summary,
        hierarchical_probabilistic_action_chunking,
        load_manifest,
        navigation_prompt,
        split_by_episode,
    )
    from .train_qwen_qlora import require_training_dependencies, resolve_model_class
except ImportError:
    from navida_data import (
        assistant_answer,
        dataset_summary,
        hierarchical_probabilistic_action_chunking,
        load_manifest,
        navigation_prompt,
        split_by_episode,
    )
    from train_qwen_qlora import require_training_dependencies, resolve_model_class


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_OUTPUT_DIR = "habitat_vln/outputs/navida_qlora"


class NaVIDADataset:
    """Small list-backed dataset compatible with Transformers Trainer."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class NaVIDACollator:
    """Build mixed multi-image batches and mask user prompts from the loss."""

    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"

    def messages(self, record, include_answer):
        content = [
            {"type": "image", "image": image_path}
            for image_path in record["_image_paths"]
        ]
        content.append({"type": "text", "text": navigation_prompt(record)})
        messages = [{"role": "user", "content": content}]
        if include_answer:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_answer(record)}
                    ],
                }
            )
        return messages

    def __call__(self, records):
        images = []
        for record in records:
            images.extend(
                Image.open(path).convert("RGB") for path in record["_image_paths"]
            )
        try:
            full_texts = [
                self.processor.apply_chat_template(
                    self.messages(record, include_answer=True),
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for record in records
            ]
            prompt_texts = [
                self.processor.apply_chat_template(
                    self.messages(record, include_answer=False),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for record in records
            ]
            batch = self.processor(
                text=full_texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            prompt_batch = self.processor(
                text=prompt_texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
        finally:
            for image in images:
                image.close()

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, : int(prompt_length)] = -100
        batch["labels"] = labels
        return batch


def freeze_visual_encoder_parameters(model):
    """Freeze the visual encoder while retaining merger/projector adapters."""
    frozen = 0
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        is_visual = "visual" in lowered or "vision" in lowered
        is_projector = "merger" in lowered or "projector" in lowered
        if is_visual and not is_projector:
            parameter.requires_grad = False
            frozen += parameter.numel()
    print(f"frozen visual encoder parameters: {frozen}")


def validate_and_split(args):
    records = load_manifest(args.manifest, max_samples=args.max_samples)
    train_records, validation_records = split_by_episode(
        records, validation_ratio=args.validation_ratio, seed=args.seed
    )
    if not train_records:
        raise ValueError("episode split produced no training samples")
    print("all:", json.dumps(dataset_summary(records), sort_keys=True))
    print("train:", json.dumps(dataset_summary(train_records), sort_keys=True))
    if validation_records:
        print(
            "validation:",
            json.dumps(dataset_summary(validation_records), sort_keys=True),
        )
    print("example_prompt:", navigation_prompt(train_records[0]))
    print("example_answer:", assistant_answer(train_records[0]))
    return train_records, validation_records


def run_self_test():
    """Test HPAC, mixed schema validation, and episode-safe splitting."""
    actions = [
        "move_forward",
        "move_forward",
        "move_forward",
        "turn_left",
        "move_forward",
        "stop",
    ]
    chunks_a = hierarchical_probabilistic_action_chunking(actions, seed=7)
    chunks_b = hierarchical_probabilistic_action_chunking(actions, seed=7)
    if chunks_a != chunks_b:
        raise AssertionError("HPAC must be reproducible for a fixed seed")
    if [action for chunk in chunks_a for action in chunk["actions"]] != actions:
        raise AssertionError("HPAC changed the atomic action sequence")

    with tempfile.TemporaryDirectory(prefix="navida_self_test_") as temp_dir:
        root = Path(temp_dir)
        image_paths = []
        for index in range(3):
            image_path = root / f"frame_{index}.jpg"
            Image.new("RGB", (32, 32), color=(40 * index, 80, 120)).save(image_path)
            image_paths.append(image_path.name)
        common = {
            "schema_version": 1,
            "episode_id": "episode-1",
            "scene_id": "self-test-scene",
            "instruction": "Walk to the doorway.",
            "action_chunk": [{"action": "move_forward", "count": 2}],
            "atomic_actions": ["move_forward", "move_forward"],
            "start_step": 0,
            "end_step": 1,
            "hpac": {},
        }
        records = [
            {
                **common,
                "task": "vln",
                "sample_id": "episode-1:vln",
                "images": image_paths[:2],
            },
            {
                **common,
                "task": "ids",
                "sample_id": "episode-1:ids",
                "images": [image_paths[0], image_paths[2]],
            },
        ]
        manifest = root / "manifest.jsonl"
        with manifest.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        loaded = load_manifest(manifest)
        summary = dataset_summary(loaded)
        if summary["tasks"] != {"ids": 1, "vln": 1}:
            raise AssertionError("mixed task counts are incorrect")
        if summary["trajectory_actions"] != {"move_forward": 2}:
            raise AssertionError("trajectory actions must not double-count IDS labels")
        if not all(navigation_prompt(record) for record in loaded):
            raise AssertionError("task prompts must not be empty")
    print("NaVIDA data pipeline self-test passed")


def train(args, train_records, validation_records):
    torch, transformers, peft = require_training_dependencies()
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = transformers.AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model_class = resolve_model_class(transformers)
    model = model_class.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model = peft.prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    target_modules = args.lora_target_modules
    if target_modules == ["all-linear"]:
        target_modules = "all-linear"
    lora_config = peft.LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(model, lora_config)
    freeze_visual_encoder_parameters(model)
    model.config.use_cache = False
    model.print_trainable_parameters()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = transformers.TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch" if validation_records else "no",
        load_best_model_at_end=bool(validation_records),
        metric_for_best_model="eval_loss" if validation_records else None,
        greater_is_better=False if validation_records else None,
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=NaVIDADataset(train_records),
        eval_dataset=(
            NaVIDADataset(validation_records) if validation_records else None
        ),
        data_collator=NaVIDACollator(processor),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"saved final adapter to {final_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "v_proj"],
        help="LoRA module suffixes, or the special value all-linear.",
    )
    parser.add_argument("--min-pixels", type=int, default=56 * 56)
    parser.add_argument("--max-pixels", type=int, default=224 * 224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.manifest:
        raise ValueError("--manifest is required unless --self-test is used")
    train_records, validation_records = validate_and_split(args)
    if args.dry_run:
        print("dry-run passed; model and GPU were not loaded")
        return
    train(args, train_records, validation_records)


if __name__ == "__main__":
    main()
