# Legacy navigation experiments

These scripts are retained as small historical experiments and do not use the
current `core -> envs -> policies -> control -> runtime` navigation framework.

- `qwen_habitat_sim_nav.py` drives Habitat-Sim directly with Qwen.
- `pointnav_manual.py` is a keyboard-control example.
- `pointnav_auto.py` is a simple distance-and-angle controller.
- `pointnav_oracle.py` follows Habitat's shortest path.

Use `habitat_vln/habitat_vln_nav.py` for current navigation work.
