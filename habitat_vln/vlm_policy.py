import json
import re
from typing import Optional

import numpy as np
from PIL import Image

try:
    from .core import NavigationObservation, PolicyOutput
    from .prompts import (
        ADVISORY_ACTIONS,
        EXPLORATION_ACTIONS,
        VALID_ACTIONS,
        build_navigation_prompt,
    )
except ImportError:
    from core import NavigationObservation, PolicyOutput
    from prompts import (
        ADVISORY_ACTIONS,
        EXPLORATION_ACTIONS,
        VALID_ACTIONS,
        build_navigation_prompt,
    )


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


class MockVLMPolicy:
    """Small deterministic policy for checking the Habitat loop without a VLM."""

    def __init__(self, allowed_actions=None):
        self.allowed_actions = set(allowed_actions or EXPLORATION_ACTIONS)
        if self.allowed_actions <= ADVISORY_ACTIONS:
            self._actions = [
                "follow_goal",
                "follow_goal",
                "turn_left_to_avoid",
                "follow_goal",
                "turn_right_to_avoid",
            ]
        else:
            self._actions = [
                "turn_left",
                "move_forward",
                "move_forward",
                "turn_right",
                "move_forward",
            ]

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
        action = self._actions[(observation.step or 0) % len(self._actions)]
        if action not in self.allowed_actions:
            action = sorted(self.allowed_actions)[0]
        return PolicyOutput(
            action=action,
            raw_text=f'{{"action": "{action}"}}',
            is_valid=True,
        )


class QwenVLMPolicy:
    def __init__(
        self,
        model_id=DEFAULT_MODEL_ID,
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=16,
        fallback_action="turn_left",
        allowed_actions=None,
        load_in_4bit=False,
        bnb_4bit_compute_dtype="float16",
        adapter_path=None,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.allowed_actions = set(allowed_actions or EXPLORATION_ACTIONS)
        self._validate_allowed_actions(self.allowed_actions)
        self.fallback_action = self._validate_fallback(fallback_action)

        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "QwenVLMPolicy requires transformers, torch, pillow, and optionally "
                "qwen-vl-utils. Install them before running without --mock-policy."
            ) from exc

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model_cls = self._resolve_model_class()
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self._resolve_torch_dtype(
                    bnb_4bit_compute_dtype
                ),
                bnb_4bit_use_double_quant=True,
            )
        self.model = model_cls.from_pretrained(model_id, **model_kwargs)
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "Loading --adapter-path requires peft. Install "
                    "habitat_vln/requirements-training.txt first."
                ) from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            process_vision_info = None
        self.process_vision_info = process_vision_info

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
        prompt = build_navigation_prompt(
            observation.instruction,
            self.allowed_actions,
            navigation_context=observation.navigation_context,
        )
        image = Image.fromarray(np.asarray(observation.rgb).astype(np.uint8))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self._build_inputs(messages)
        try:
            with self.torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            input_len = inputs["input_ids"].shape[-1]
            output_ids = generated_ids[:, input_len:]
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

        action = parse_action(raw_text, self.allowed_actions)
        if action is None:
            return PolicyOutput(
                action=self.fallback_action,
                raw_text=raw_text,
                is_valid=False,
            )
        return PolicyOutput(action=action, raw_text=raw_text, is_valid=True)

    def _build_inputs(self, messages):
        if self.process_vision_info is not None:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

        return inputs.to(self.model.device)

    def _resolve_model_class(self):
        import transformers

        class_names = [
            "AutoModelForImageTextToText",
            "AutoModelForMultimodalLM",
            "Qwen2_5_VLForConditionalGeneration",
            "AutoModelForVision2Seq",
        ]
        for class_name in class_names:
            model_cls = getattr(transformers, class_name, None)
            if model_cls is not None:
                return model_cls
        raise RuntimeError(
            "Could not find a compatible Qwen2.5-VL model class in transformers. "
            "Please upgrade transformers."
        )

    def _validate_fallback(self, action):
        if action not in self.allowed_actions:
            raise ValueError(
                f"fallback_action must be one of {sorted(self.allowed_actions)}"
            )
        return action

    def _validate_allowed_actions(self, allowed_actions):
        invalid_actions = set(allowed_actions) - (VALID_ACTIONS | ADVISORY_ACTIONS)
        if invalid_actions:
            raise ValueError(f"Invalid actions: {sorted(invalid_actions)}")
        if not allowed_actions:
            raise ValueError("allowed_actions must not be empty")

    def _resolve_torch_dtype(self, dtype):
        aliases = {
            "float16": "float16",
            "fp16": "float16",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
            "float32": "float32",
            "fp32": "float32",
        }
        attr = aliases.get(str(dtype).lower())
        if attr is None:
            raise ValueError(
                "--bnb-4bit-compute-dtype must be one of "
                "float16, bfloat16, or float32"
            )
        return getattr(self.torch, attr)


def parse_action(text: str, allowed_actions=None) -> Optional[str]:
    allowed_actions = set(allowed_actions or VALID_ACTIONS)
    cleaned = text.strip().lower()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        action = str(payload.get("action", "")).strip().lower()
        if action in allowed_actions:
            return action

    action_pattern = "|".join(re.escape(action) for action in sorted(allowed_actions))
    for action in re.findall(rf"\b({action_pattern})\b", cleaned):
        if action in allowed_actions:
            return action

    return None
