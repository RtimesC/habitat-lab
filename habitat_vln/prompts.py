VALID_ACTIONS = {
    "turn_left",
    "turn_right",
    "move_forward",
    "stop",
}

ADVISORY_ACTIONS = {
    "follow_goal",
    "turn_left_to_avoid",
    "turn_right_to_avoid",
    "stop_if_reached",
}

EXPLORATION_ACTIONS = {
    "turn_left",
    "turn_right",
    "move_forward",
}

NAVIGATION_PROMPT_TEMPLATE = """You are controlling a robot in a Habitat indoor scene.

Task:
{instruction}

Navigation state:
{navigation_state}

You can choose exactly one action:
{action_list}

{mode_rules}

Use the RGB image and navigation state together. The navigation state contains
the true PointNav target geometry. Smaller goal_distance_m is better, and a
negative distance_change_m means the last step made progress. If the previous
action increased distance or collided, change strategy. The goal angle
convention is: negative means the goal is to the left, positive means the goal
is to the right, and 0 degrees means straight ahead. If the goal angle is near 0
and center depth is safe, move_forward. If the goal is left, turn_left. If the
goal is right, turn_right. Depth values are 10th-percentile safe distances in
meters for the left, center, and right image regions. Do not choose stop unless
goal_distance_m is below success_distance_m.
Do not mention bounding boxes, object detector results, or pixel coordinates.
{stop_rule}

Return only this JSON object and nothing else:
{{"action": "{example_action}"}}
"""


def format_navigation_state(navigation_context):
    if not navigation_context:
        return "- step: unknown"

    rows = []
    for key in [
        "step",
        "agent_position",
        "agent_rotation",
        "goal_position",
        "goal_distance_m",
        "goal_angle_deg",
        "success_distance_m",
        "distance_change_m",
        "collided",
        "depth_left_m",
        "depth_center_m",
        "depth_right_m",
        "previous_action",
    ]:
        value = navigation_context.get(key, "unknown")
        if value is None or value == "":
            value = "unknown"
        elif isinstance(value, float):
            value = f"{value:.3f}"
        elif isinstance(value, (tuple, list)):
            value = "(" + ", ".join(f"{float(v):.3f}" for v in value) + ")"
        rows.append(f"- {key}: {value}")
    return "\n".join(rows)


def build_navigation_prompt(instruction, allowed_actions, navigation_context=None):
    allowed_actions = set(allowed_actions)
    action_list = "\n".join(f"- {action}" for action in sorted(allowed_actions))
    example_action = sorted(allowed_actions)[0]

    if allowed_actions <= ADVISORY_ACTIONS:
        mode_rules = (
            "You are a high-level navigation advisor, not the low-level motor "
            "controller. Choose one advisory action:\n"
            "- follow_goal: let the geometric controller steer toward the PointNav target.\n"
            "- turn_left_to_avoid: recommend a left turn to avoid an obstacle or recover.\n"
            "- turn_right_to_avoid: recommend a right turn to avoid an obstacle or recover.\n"
            "- stop_if_reached: recommend stopping only when the goal has been reached.\n"
            "Do not output move_forward, turn_left, turn_right, or stop in advisor mode."
        )
        stop_rule = (
            "Choose stop_if_reached only when goal_distance_m is below "
            "success_distance_m. Otherwise prefer follow_goal unless depth or "
            "collision suggests an avoidance turn."
        )
    elif "stop" in allowed_actions:
        mode_rules = (
            "You are the low-level controller. Choose one executable Habitat action."
        )
        stop_rule = (
            "Choose stop only when goal_distance_m is below success_distance_m.\n"
            "Otherwise keep navigating with turn_left, turn_right, or move_forward."
        )
    else:
        mode_rules = (
            "You are the low-level controller. Choose one executable Habitat action."
        )
        stop_rule = (
            "Continue moving unless movement is impossible. If forward motion is "
            "blocked, choose turn_left or turn_right."
        )

    return NAVIGATION_PROMPT_TEMPLATE.format(
        instruction=instruction,
        navigation_state=format_navigation_state(navigation_context),
        action_list=action_list,
        mode_rules=mode_rules,
        stop_rule=stop_rule,
        example_action=example_action,
    )
