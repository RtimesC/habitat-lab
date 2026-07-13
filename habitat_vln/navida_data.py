"""NaVIDA-style HPAC and mixed VLN/IDS training-data helpers."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path


SCHEMA_VERSION = 1
VALID_TASKS = {"vln", "ids"}
VALID_ACTIONS = {"move_forward", "turn_left", "turn_right", "stop"}
REQUIRED_FIELDS = {
    "schema_version",
    "task",
    "sample_id",
    "episode_id",
    "images",
    "action_chunk",
    "atomic_actions",
    "start_step",
    "end_step",
}


def hierarchical_probabilistic_action_chunking(
    actions,
    merge_probability=0.7,
    max_level2_chunks=3,
    seed=42,
):
    """Apply the two-level HPAC procedure from the NaVIDA paper."""
    actions = list(actions)
    if not actions:
        return []
    if not 0.0 <= merge_probability <= 1.0:
        raise ValueError("merge_probability must be in [0, 1]")
    if max_level2_chunks < 1:
        raise ValueError("max_level2_chunks must be positive")
    invalid = sorted(set(actions) - VALID_ACTIONS)
    if invalid:
        raise ValueError(f"unsupported actions: {invalid}")

    rng = random.Random(seed)
    level1 = []
    current = [(0, actions[0])]
    for index, action in enumerate(actions[1:], start=1):
        can_merge = (
            action == actions[index - 1]
            and len(current) < 3
            and rng.random() <= merge_probability
        )
        if can_merge:
            current.append((index, action))
        else:
            level1.append(current)
            current = [(index, action)]
    level1.append(current)

    chunks = []
    for offset in range(0, len(level1), max_level2_chunks):
        grouped = level1[offset : offset + max_level2_chunks]
        flattened = [item for subchunk in grouped for item in subchunk]
        chunks.append(
            {
                "start_index": flattened[0][0],
                "end_index": flattened[-1][0],
                "actions": [action for _, action in flattened],
                "level1_subchunks": len(grouped),
            }
        )
    return chunks


def compact_action_chunk(actions):
    """Convert atomic actions into action-type plus count tokens."""
    compact = []
    for action in actions:
        if compact and compact[-1]["action"] == action:
            compact[-1]["count"] += 1
        else:
            compact.append({"action": action, "count": 1})
    return compact


def uniformly_sample_indices(last_index, maximum):
    """Select up to ``maximum`` frames while retaining first and current views."""
    count = last_index + 1
    if count <= maximum:
        return list(range(count))
    if maximum == 1:
        return [last_index]
    return [round(index * last_index / (maximum - 1)) for index in range(maximum)]


def resolve_source_image(record, field, manifest_path):
    image_path = Path(record[field]).expanduser()
    if not image_path.is_absolute():
        image_path = manifest_path.parent / image_path
    return image_path.resolve()


def build_episode_samples(
    records,
    source_manifest,
    merge_probability=0.7,
    max_level2_chunks=3,
    max_history_frames=8,
    seed=42,
):
    """Build paired VLN and IDS samples from one ordered Habitat trajectory."""
    records = sorted(records, key=lambda record: int(record["step"]))
    if not records:
        return []
    steps = [int(record["step"]) for record in records]
    if steps != list(range(steps[0], steps[0] + len(steps))):
        raise ValueError(f"episode {records[0]['episode_id']} has non-contiguous steps")

    episode_id = str(records[0]["episode_id"])
    episode_seed = int(
        hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest()[:8], 16
    )
    hpac_chunks = hierarchical_probabilistic_action_chunking(
        [record["action"] for record in records],
        merge_probability=merge_probability,
        max_level2_chunks=max_level2_chunks,
        seed=episode_seed,
    )

    samples = []
    for chunk_index, chunk in enumerate(hpac_chunks):
        start_index = chunk["start_index"]
        end_index = chunk["end_index"]
        start_record = records[start_index]
        end_record = records[end_index]
        history_indices = uniformly_sample_indices(start_index, max_history_frames)
        history_images = [
            str(resolve_source_image(records[index], "image", source_manifest))
            for index in history_indices
        ]
        common = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "scene_id": str(start_record.get("scene_id", "")),
            "instruction": str(start_record.get("instruction", "")),
            "navigation_context": start_record.get("navigation_context"),
            "action_chunk": compact_action_chunk(chunk["actions"]),
            "atomic_actions": list(chunk["actions"]),
            "start_step": int(start_record["step"]),
            "end_step": int(end_record["step"]),
            "hpac": {
                "merge_probability": merge_probability,
                "max_level2_chunks": max_level2_chunks,
                "level1_subchunks": chunk["level1_subchunks"],
            },
        }
        samples.append(
            {
                **common,
                "task": "vln",
                "sample_id": f"{episode_id}:chunk-{chunk_index}:vln",
                "images": history_images,
            }
        )

        if "next_image" in end_record:
            goal_image = resolve_source_image(end_record, "next_image", source_manifest)
        elif end_index + 1 < len(records):
            goal_image = resolve_source_image(
                records[end_index + 1], "image", source_manifest
            )
        else:
            goal_image = None
        if goal_image is not None and goal_image.is_file():
            samples.append(
                {
                    **common,
                    "task": "ids",
                    "sample_id": f"{episode_id}:chunk-{chunk_index}:ids",
                    "images": [history_images[-1], str(goal_image)],
                }
            )
    return samples


def navigation_prompt(record):
    """Return the task-specific prompt used for mixed NaVIDA training."""
    if record["task"] == "vln":
        context = record.get("navigation_context") or {}
        state_parts = []
        if context.get("goal_distance_m") is not None:
            state_parts.append(
                f"target_distance_m={float(context['goal_distance_m']):.2f}"
            )
        if context.get("goal_angle_deg") is not None:
            state_parts.append(
                f"target_bearing_deg={float(context['goal_angle_deg']):.1f}"
            )
        if context.get("previous_action"):
            state_parts.append(f"previous_action={context['previous_action']}")
        if context.get("collided") is not None:
            state_parts.append(f"collided={bool(context['collided'])}")
        state_text = ""
        if state_parts:
            state_text = (
                " Current navigation state: "
                + ", ".join(state_parts)
                + ". Target bearing uses negative=left and positive=right."
            )
        return (
            "Imagine you are a robot programmed for navigation tasks. You have "
            "been given a sequence of historical RGB observations ending at the "
            "current view. Your assigned task is: "
            f"{record['instruction']}"
            f"{state_text} Analyze the images and predict the next "
            "navigation action chunk."
        )
    return (
        "Imagine you are a robot programmed for navigation tasks. The first image "
        "is the current view and the second image is the goal view. Predict the "
        "navigation action chunk that moves the robot between these viewpoints."
    )


def assistant_answer(record):
    """Serialize the shared action grammar used by VLN and IDS."""
    return json.dumps(
        {"actions": record["action_chunk"]}, separators=(",", ":")
    )


def validate_record(record, manifest_path, line_number, require_images=True):
    """Validate one mixed-task sample and resolve all image paths."""
    location = f"{manifest_path}:{line_number}"
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        raise ValueError(f"{location}: missing fields {missing}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{location}: unsupported schema_version")
    if record["task"] not in VALID_TASKS:
        raise ValueError(f"{location}: task must be one of {sorted(VALID_TASKS)}")
    expected_images = 2 if record["task"] == "ids" else None
    if not isinstance(record["images"], list) or not record["images"]:
        raise ValueError(f"{location}: images must be a non-empty list")
    if expected_images and len(record["images"]) != expected_images:
        raise ValueError(f"{location}: IDS samples require exactly two images")
    if record["task"] == "vln" and not str(record.get("instruction", "")).strip():
        raise ValueError(f"{location}: VLN instruction must not be empty")
    if not record["atomic_actions"]:
        raise ValueError(f"{location}: atomic_actions must not be empty")
    invalid = sorted(set(record["atomic_actions"]) - VALID_ACTIONS)
    if invalid:
        raise ValueError(f"{location}: unsupported actions {invalid}")

    image_paths = []
    for raw_path in record["images"]:
        image_path = Path(raw_path).expanduser()
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        image_path = image_path.resolve()
        if require_images and not image_path.is_file():
            raise ValueError(f"{location}: image does not exist: {image_path}")
        image_paths.append(str(image_path))
    validated = dict(record)
    validated["_image_paths"] = image_paths
    return validated


def load_manifest(manifest_path, max_samples=None, require_images=True):
    """Load and validate a NaVIDA mixed-task JSONL manifest."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    records = []
    with manifest_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(
                validate_record(record, manifest_path, line_number, require_images)
            )
            if max_samples is not None and len(records) >= max_samples:
                break
    if not records:
        raise ValueError("manifest contains no samples")
    return records


def split_by_episode(records, validation_ratio=0.1, seed=42):
    """Keep all VLN/IDS samples from one episode in the same split."""
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1)")
    episode_ids = sorted({record["episode_id"] for record in records})
    threshold = int(validation_ratio * 10000)
    validation_ids = {
        episode_id
        for episode_id in episode_ids
        if int(hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest()[:8], 16)
        % 10000
        < threshold
    }
    if validation_ratio > 0 and len(episode_ids) > 1:
        if not validation_ids:
            validation_ids.add(episode_ids[-1])
        if len(validation_ids) == len(episode_ids):
            validation_ids.remove(episode_ids[0])
    train = [record for record in records if record["episode_id"] not in validation_ids]
    validation = [record for record in records if record["episode_id"] in validation_ids]
    return train, validation


def dataset_summary(records):
    """Return compact task and action statistics."""
    task_counts = Counter(record["task"] for record in records)
    action_labels_by_task = {
        task: dict(
            sorted(
                Counter(
                    action
                    for record in records
                    if record["task"] == task
                    for action in record["atomic_actions"]
                ).items()
            )
        )
        for task in sorted(task_counts)
    }
    trajectory_actions = Counter(
        action
        for record in records
        if record["task"] == "vln"
        for action in record["atomic_actions"]
    )
    return {
        "samples": len(records),
        "episodes": len({record["episode_id"] for record in records}),
        "tasks": dict(sorted(task_counts.items())),
        "unique_chunks": len(
            {
                (record["episode_id"], record["start_step"], record["end_step"])
                for record in records
            }
        ),
        "trajectory_actions": dict(sorted(trajectory_actions.items())),
        "action_labels_by_task": action_labels_by_task,
    }
