"""Qwen NaVIDA action-chunk generation and closed-loop policy helpers."""

import json
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    from ..core import NavigationObservation
    from ..data.navida_data import navigation_prompt, uniformly_sample_indices
except ImportError:
    from core import NavigationObservation
    from data.navida_data import navigation_prompt, uniformly_sample_indices

from .vlm_policy import DEFAULT_MODEL_ID, PolicyOutput


VALID_ACTIONS = {"move_forward", "turn_left", "turn_right", "stop"}


@dataclass
class ActionChunkOutput:
    atomic_actions: list
    raw_text: str
    is_valid: bool


def parse_action_chunk(text, max_action_count=20):
    """Parse and expand ``{"actions": [{"action": ..., "count": ...}]}``."""
    cleaned = text.strip()
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match and match.group(0) != cleaned:
        candidates.append(match.group(0))
    payload = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(payload, dict) or not isinstance(payload.get("actions"), list):
        return None

    atomic_actions = []
    for item in payload["actions"]:
        if not isinstance(item, dict):
            return None
        action = str(item.get("action", "")).strip().lower()
        count = item.get("count")
        if action not in VALID_ACTIONS or not isinstance(count, int):
            return None
        if count < 1 or count > max_action_count:
            return None
        atomic_actions.extend([action] * count)
    return atomic_actions or None


class QwenNaVIDAModel:
    """Load Qwen plus an optional adapter and generate structured action chunks."""

    def __init__(
        self,
        model_id=DEFAULT_MODEL_ID,
        adapter_path=None,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        min_pixels=56 * 56,
        max_pixels=128 * 128,
        max_new_tokens=128,
    ):
        import torch
        import transformers

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        processor_path = adapter_path or model_id
        self.processor = transformers.AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        model_class = None
        for class_name in [
            "Qwen2_5_VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        ]:
            model_class = getattr(transformers, class_name, None)
            if model_class is not None:
                break
        if model_class is None:
            raise RuntimeError("installed transformers has no compatible Qwen class")

        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
        }
        if load_in_4bit:
            compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
            model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        self.model = model_class.from_pretrained(model_id, **model_kwargs)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate_action_chunk(self, images, prompt):
        """Generate one action chunk from PIL/NumPy RGB images and a task prompt."""
        pil_images = [
            image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
            for image in images
        ]
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in pil_images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=pil_images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        try:
            with self.torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            output_ids = generated_ids[:, inputs["input_ids"].shape[-1] :]
            raw_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        finally:
            if "generated_ids" in locals():
                del generated_ids
            if "output_ids" in locals():
                del output_ids
            del inputs
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        atomic_actions = parse_action_chunk(raw_text)
        return ActionChunkOutput(
            atomic_actions=atomic_actions or [],
            raw_text=raw_text,
            is_valid=atomic_actions is not None,
        )


class NaVIDAChunkPolicy:
    """Generate chunks at replanning points and expose one Habitat action per step."""

    def __init__(
        self,
        adapter_path,
        model_id=DEFAULT_MODEL_ID,
        fallback_action="move_forward",
        max_history_frames=8,
        max_executed_actions=3,
        **model_kwargs,
    ):
        if fallback_action not in VALID_ACTIONS:
            raise ValueError(f"invalid fallback action: {fallback_action}")
        if max_history_frames < 1 or max_executed_actions < 1:
            raise ValueError("history and execution limits must be positive")
        self.runtime = QwenNaVIDAModel(
            model_id=model_id,
            adapter_path=adapter_path,
            **model_kwargs,
        )
        self.fallback_action = fallback_action
        self.max_history_frames = max_history_frames
        self.max_executed_actions = max_executed_actions
        self.history = []
        self.action_queue = []
        self.last_chunk_text = ""

    def reset(self):
        self.history = []
        self.action_queue = []
        self.last_chunk_text = ""

    def clear_action_queue(self):
        """Force replanning after an external safety override."""
        self.action_queue = []

    def predict(
        self,
        observation,
        instruction=None,
        step=None,
        navigation_context=None,
    ):
        observation = NavigationObservation.from_legacy_inputs(
            observation,
            instruction,
            step,
            navigation_context,
        )
        if observation.step in (None, 0):
            self.reset()
        self.history.append(
            Image.fromarray(np.asarray(observation.rgb).astype(np.uint8))
        )

        if self.action_queue:
            action = self.action_queue.pop(0)
            return PolicyOutput(
                action=action,
                raw_text=(
                    f"{self.last_chunk_text}\n"
                    f'{{"queued_action":"{action}","remaining":{len(self.action_queue)}}}'
                ),
                is_valid=True,
            )

        indices = uniformly_sample_indices(
            len(self.history) - 1, self.max_history_frames
        )
        selected_images = [self.history[index] for index in indices]
        prompt = navigation_prompt(
            {
                "task": "vln",
                "instruction": observation.instruction,
                "navigation_context": observation.navigation_context,
            }
        )
        chunk = self.runtime.generate_action_chunk(selected_images, prompt)
        self.last_chunk_text = chunk.raw_text
        if not chunk.is_valid:
            return PolicyOutput(
                action=self.fallback_action,
                raw_text=chunk.raw_text,
                is_valid=False,
            )

        executable = chunk.atomic_actions[: self.max_executed_actions]
        action = executable[0]
        self.action_queue = executable[1:]
        return PolicyOutput(action=action, raw_text=chunk.raw_text, is_valid=True)
