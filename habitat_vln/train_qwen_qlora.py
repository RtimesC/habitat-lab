"""Validate navigation data and QLoRA-fine-tune Qwen2.5-VL on Oracle actions."""

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image

try:
    from .training_data import (
        assistant_answer,
        dataset_summary,
        load_manifests,
        make_record,
        split_by_episode,
        training_prompt,
    )
except ImportError:
    from training_data import (
        assistant_answer,
        dataset_summary,
        load_manifests,
        make_record,
        split_by_episode,
        training_prompt,
    )


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_OUTPUT_DIR = "habitat_vln/outputs/qwen_qlora"


class NavigationDataset:
    """Small list-backed dataset compatible with Transformers Trainer."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class QwenNavigationCollator:
    """Build multimodal batches and mask the user prompt from the loss."""

    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"

    def _messages(self, record, include_answer):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": record["_image_path"]},
                    {"type": "text", "text": training_prompt(record)},
                ],
            }
        ]
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
        import torch

        images = [Image.open(record["_image_path"]).convert("RGB") for record in records]
        try:
            full_texts = [
                self.processor.apply_chat_template(
                    self._messages(record, include_answer=True),
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for record in records
            ]
            prompt_texts = [
                self.processor.apply_chat_template(
                    self._messages(record, include_answer=False),
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
        del prompt_batch
        return batch


def resolve_model_class(transformers_module):
    for class_name in [
        "Qwen2_5_VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ]:
        model_class = getattr(transformers_module, class_name, None)
        if model_class is not None:
            return model_class
    raise RuntimeError("installed transformers has no compatible Qwen2.5-VL class")


def require_training_dependencies():
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "Training requires peft in addition to the existing torch, transformers, "
            "accelerate, and bitsandbytes packages. Install habitat_vln/"
            "requirements-training.txt first."
        ) from exc
    return torch, transformers, peft


def print_summary(label, records):
    print(f"{label}: {json.dumps(dataset_summary(records), sort_keys=True)}")


def validate_and_split(args):
    records = load_manifests(args.manifest, max_samples=args.max_samples)
    train_records, validation_records = split_by_episode(
        records,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    if not train_records:
        raise ValueError("episode split produced no training samples")
    print_summary("all", records)
    print_summary("train", train_records)
    if validation_records:
        print_summary("validation", validation_records)
    else:
        print("validation: no samples (at least two episodes are needed)")
    example = train_records[0]
    print(f"example_sample_id: {example['sample_id']}")
    print(f"example_answer: {assistant_answer(example)}")
    return train_records, validation_records


def run_self_test():
    """Exercise schema validation, prompt creation, and episode-safe splitting."""
    with tempfile.TemporaryDirectory(prefix="habitat_vln_training_") as temp_dir:
        root = Path(temp_dir)
        image_path = root / "frame.jpg"
        Image.new("RGB", (32, 32), color=(64, 96, 128)).save(image_path)
        manifest_path = root / "manifest.jsonl"
        context = {
            "step": 0,
            "goal_distance_m": 2.0,
            "goal_angle_deg": 0.0,
            "success_distance_m": 0.2,
            "previous_action": "none",
        }
        records = [
            make_record(
                sample_id=f"episode-{episode}:0",
                episode_id=f"episode-{episode}",
                scene_id="self-test-scene",
                step=0,
                instruction="Navigate to the target.",
                image=image_path.name,
                action=action,
                navigation_context=context,
            )
            for episode, action in [(1, "move_forward"), (2, "stop")]
        ]
        with manifest_path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        loaded = load_manifests([manifest_path])
        train_records, validation_records = split_by_episode(loaded, 0.5, seed=42)
        if not train_records or not validation_records:
            raise AssertionError("self-test split must contain train and validation data")
        if not training_prompt(loaded[0]) or not assistant_answer(loaded[0]):
            raise AssertionError("self-test prompt formatting failed")
    print("training pipeline self-test passed")


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
        model,
        use_gradient_checkpointing=True,
    )
    lora_config = peft.LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(model, lora_config)
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
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=NavigationDataset(train_records),
        eval_dataset=(
            NavigationDataset(validation_records) if validation_records else None
        ),
        data_collator=QwenNavigationCollator(processor),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final_adapter"))
    processor.save_pretrained(str(output_dir / "final_adapter"))
    print(f"saved final LoRA adapter to {output_dir / 'final_adapter'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", nargs="+", type=Path)
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
    parser.add_argument("--min-pixels", type=int, default=56 * 56)
    parser.add_argument("--max-pixels", type=int, default=224 * 224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifests and splits without loading Qwen or using the GPU.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Test the data pipeline using temporary synthetic records.",
    )
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
