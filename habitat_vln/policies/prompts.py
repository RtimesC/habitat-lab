"""Target-free prompts and action vocabularies for indoor semantic navigation."""

import json

VALID_ACTIONS = {
    "turn_left",
    "turn_right",
    "move_forward",
    "stop",
}

EXPLORATION_ACTIONS = {
    "turn_left",
    "turn_right",
    "move_forward",
}

NAVIGATION_PROMPT_TEMPLATE = """You are controlling a robot inside one building.

Task:
{instruction}

Policy-visible state:
{navigation_state}

You have no target coordinate, target distance, target bearing, shortest path,
success radius, or hidden map answer. Use the task language, current image,
allowed building prior, local depth, collision feedback, and action history.

Do not present an unobserved room, connection, or destination as confirmed.
Choose a safe action that either advances the task or reveals useful evidence.
Choose stop only when visible evidence strongly supports that the requested
semantic destination has been reached.

Choose exactly one action:
{action_list}

Return only this JSON object and nothing else:
{{"action": "{example_action}", "location_hypothesis": "short statement",
"confidence": "low|medium|high", "observed_evidence": ["short evidence"],
"topology_hypothesis": "short statement", "next_verification": "short statement"}}
"""


def format_navigation_state(navigation_context):
    """Format the target-free state and allowed building weak prior for Qwen."""
    if not navigation_context:
        return "- step: unknown"

    rows = []
    for key in [
        "step",
        "collided",
        "depth_left_m",
        "depth_center_m",
        "depth_right_m",
        "previous_action",
        "previous_action_count",
        "previous_collision",
        "building_prior",
    ]:
        if key not in navigation_context:
            continue
        value = navigation_context[key]
        if value is None or value == "":
            value = "unknown"
        elif isinstance(value, float):
            value = f"{value:.3f}"
        elif isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        rows.append(f"- {key}: {value}")
    return "\n".join(rows) or "- step: unknown"


def build_navigation_prompt(
    instruction, allowed_actions, navigation_context=None
):
    """Build the single semantic-indoor prompt used by active policies."""
    allowed_actions = set(allowed_actions)
    action_list = "\n".join(
        f"- {action}" for action in sorted(allowed_actions)
    )
    example_action = sorted(allowed_actions)[0]
    return NAVIGATION_PROMPT_TEMPLATE.format(
        instruction=instruction,
        navigation_state=format_navigation_state(navigation_context or {}),
        action_list=action_list,
        example_action=example_action,
    )
