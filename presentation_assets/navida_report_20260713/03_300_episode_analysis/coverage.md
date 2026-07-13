# VLN State-Action Coverage

- Samples: 1500
- Episodes: 300
- Scenes: 10
- Observed cells: 54 / 80 (67.5%)
- Under-target observed cells: 22 (target >= 20)
- Core control cells: 20 / 20 (100.0%)
- Under-target core cells: 7
- STOP coverage: 300 chunks containing STOP / 98 first-action STOP / 300 episodes / 10 scenes
- Recovery coverage: 426 samples / 246 episodes / 10 scenes
- Collision / no-progress recovery samples: 0 / 426

| Bearing | Distance | Action | Samples | Episodes | Scenes | Under target |
|---|---|---|---:|---:|---:|---:|
| ahead | arrived | move_forward | 1 | 1 | 1 | true |
| ahead | arrived | stop | 51 | 51 | 10 | false |
| ahead | far | move_forward | 41 | 33 | 9 | false |
| ahead | far | turn_left | 40 | 38 | 10 | false |
| ahead | far | turn_right | 40 | 37 | 10 | false |
| ahead | mid | move_forward | 41 | 39 | 9 | false |
| ahead | mid | turn_left | 40 | 39 | 10 | false |
| ahead | mid | turn_right | 40 | 40 | 10 | false |
| ahead | near | move_forward | 124 | 124 | 10 | false |
| ahead | near | turn_left | 32 | 32 | 10 | false |
| ahead | near | turn_right | 40 | 40 | 9 | false |
| left | arrived | stop | 8 | 8 | 6 | true |
| left | arrived | turn_left | 1 | 1 | 1 | true |
| left | arrived | turn_right | 1 | 1 | 1 | true |
| left | far | move_forward | 40 | 36 | 10 | false |
| left | far | turn_left | 40 | 39 | 10 | false |
| left | far | turn_right | 40 | 39 | 10 | false |
| left | mid | move_forward | 40 | 37 | 10 | false |
| left | mid | turn_left | 40 | 37 | 10 | false |
| left | mid | turn_right | 18 | 17 | 9 | true |
| left | near | move_forward | 3 | 3 | 3 | true |
| left | near | turn_left | 30 | 26 | 9 | false |
| left | near | turn_right | 3 | 3 | 2 | true |
| right | arrived | stop | 12 | 12 | 7 | true |
| right | arrived | turn_right | 1 | 1 | 1 | true |
| right | far | move_forward | 41 | 36 | 10 | false |
| right | far | turn_left | 40 | 38 | 10 | false |
| right | far | turn_right | 41 | 40 | 10 | false |
| right | mid | move_forward | 40 | 37 | 10 | false |
| right | mid | turn_left | 13 | 13 | 7 | true |
| right | mid | turn_right | 40 | 37 | 10 | false |
| right | near | move_forward | 4 | 4 | 4 | true |
| right | near | turn_right | 18 | 15 | 9 | true |
| strong_left | arrived | stop | 13 | 13 | 7 | true |
| strong_left | far | move_forward | 40 | 33 | 10 | false |
| strong_left | far | turn_left | 40 | 34 | 10 | false |
| strong_left | far | turn_right | 40 | 37 | 9 | false |
| strong_left | mid | move_forward | 40 | 27 | 9 | false |
| strong_left | mid | turn_left | 40 | 29 | 10 | false |
| strong_left | mid | turn_right | 18 | 13 | 7 | true |
| strong_left | near | move_forward | 4 | 3 | 3 | true |
| strong_left | near | turn_left | 10 | 9 | 8 | true |
| strong_left | near | turn_right | 1 | 1 | 1 | true |
| strong_right | arrived | move_forward | 1 | 1 | 1 | true |
| strong_right | arrived | stop | 14 | 14 | 7 | true |
| strong_right | far | move_forward | 41 | 35 | 10 | false |
| strong_right | far | turn_left | 40 | 34 | 10 | false |
| strong_right | far | turn_right | 41 | 37 | 10 | false |
| strong_right | mid | move_forward | 40 | 30 | 9 | false |
| strong_right | mid | turn_left | 22 | 19 | 8 | false |
| strong_right | mid | turn_right | 40 | 34 | 10 | false |
| strong_right | near | move_forward | 4 | 4 | 3 | true |
| strong_right | near | turn_left | 2 | 2 | 2 | true |
| strong_right | near | turn_right | 5 | 4 | 3 | true |
