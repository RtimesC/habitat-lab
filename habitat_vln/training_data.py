"""Shared schema and validation helpers for Habitat navigation training data."""

import hashlib
import json
from collections import Counter
from pathlib import Path

try:
    from .prompts import VALID_ACTIONS, build_navigation_prompt
except ImportError:
    from prompts import VALID_ACTIONS, build_navigation_prompt


SCHEMA_VERSION = 1
REQUIRED_FIELDS = {
    "schema_version",
    "sample_id",
    "episode_id",
    "instruction",
    "image",
    "action",
    "navigation_context",
}


def json_value(value):
    """Convert NumPy-like values and tuples into JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def make_record(
    sample_id,
    episode_id,
    scene_id,
    step,
    instruction,
    image,
    action,
    navigation_context,
    next_image=None,
):
    """Create one stable JSONL record from an observation and Oracle action."""
    record = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": str(sample_id),
        "episode_id": str(episode_id),
        "scene_id": str(scene_id),
        "step": int(step),
        "instruction": str(instruction),
        "image": str(image),
        "action": str(action),
        "navigation_context": json_value(navigation_context),
    }
    if next_image is not None:
        record["next_image"] = str(next_image)
    return record


def training_prompt(record):
    """Build the same navigation prompt used by the inference policy."""
    return build_navigation_prompt(
        record["instruction"],
        VALID_ACTIONS,
        navigation_context=record["navigation_context"],
    )


def assistant_answer(record):
    """Return the exact JSON answer the model should learn to emit."""
    return json.dumps({"action": record["action"]}, separators=(",", ":"))


def validate_record(record, manifest_path, line_number, require_image=True):
    """Validate one record and attach its resolved absolute image path."""
    location = f"{manifest_path}:{line_number}"
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        raise ValueError(f"{location}: missing fields {missing}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"{location}: schema_version must be {SCHEMA_VERSION}, "
            f"got {record['schema_version']!r}"
        )
    if not str(record["instruction"]).strip():
        raise ValueError(f"{location}: instruction must not be empty")
    if record["action"] not in VALID_ACTIONS:
        raise ValueError(
            f"{location}: action must be one of {sorted(VALID_ACTIONS)}, "
            f"got {record['action']!r}"
        )
    if not isinstance(record["navigation_context"], dict):
        raise ValueError(f"{location}: navigation_context must be an object")

    image_path = Path(record["image"]).expanduser()
    if not image_path.is_absolute():
        image_path = manifest_path.parent / image_path
    image_path = image_path.resolve()
    if require_image and not image_path.is_file():
        raise ValueError(f"{location}: image does not exist: {image_path}")

    validated = dict(record)
    validated["_image_path"] = str(image_path)
    validated["_manifest_path"] = str(manifest_path)
    return validated


def load_manifests(manifest_paths, max_samples=None, require_images=True):
    """Load and validate one or more JSONL manifests."""
    records = []
    for raw_path in manifest_paths:
        manifest_path = Path(raw_path).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
        with manifest_path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{manifest_path}:{line_number}: {exc}") from exc
                records.append(
                    validate_record(
                        record,
                        manifest_path,
                        line_number,
                        require_image=require_images,
                    )
                )
                if max_samples is not None and len(records) >= max_samples:
                    return records
    if not records:
        raise ValueError("no training samples were found in the manifest")
    return records


def split_by_episode(records, validation_ratio=0.1, seed=42):
    """Split whole episodes, preventing adjacent frames from leaking to validation."""
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1)")

    episode_ids = sorted({record["episode_id"] for record in records})
    validation_episodes = set()
    threshold = int(validation_ratio * 10000)
    for episode_id in episode_ids:
        digest = hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest()
        if int(digest[:8], 16) % 10000 < threshold:
            validation_episodes.add(episode_id)

    if validation_ratio > 0 and len(episode_ids) > 1:
        if not validation_episodes:
            validation_episodes.add(episode_ids[-1])
        if len(validation_episodes) == len(episode_ids):
            validation_episodes.remove(episode_ids[0])

    train_records = [
        record for record in records if record["episode_id"] not in validation_episodes
    ]
    validation_records = [
        record for record in records if record["episode_id"] in validation_episodes
    ]
    return train_records, validation_records


def dataset_summary(records):
    """Return compact counts used by dry-run and collection reports."""
    return {
        "samples": len(records),
        "episodes": len({record["episode_id"] for record in records}),
        "scenes": len({record.get("scene_id", "") for record in records}),
        "actions": dict(sorted(Counter(record["action"] for record in records).items())),
    }
