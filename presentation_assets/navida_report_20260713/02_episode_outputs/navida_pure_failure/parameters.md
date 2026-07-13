# Pure NaVIDA Failure 参数说明

- 策略：纯 NaVIDA action-chunk controller，无真实目标距离 STOP guard。
- 基础模型：`Qwen/Qwen2.5-VL-3B-Instruct`。
- adapter：`habitat_vln/outputs/navida_overfit_20ep/qlora_100_all_linear_ep5/final_adapter/`。
- chunk 执行上限：每次 replanning 最多执行 3 个 atomic actions。
- 历史图像上限：8 帧。
- 最大 episode 步数：80。
- 成功半径：0.2 m，仅用于 Habitat 评估；没有用于强制 STOP。
- 场景：HM3D example scene `00337-CFVBbU9Rsyb`，episode 7。
- instruction：`Navigate to the target location and stop when you reach it.`
- 原始视频：224 × 224，10 fps，80 帧。
- 汇报视频：动作顺序不变，2 倍慢放到约 16 秒并放大为 672 × 672。

原始运行目录：`habitat_vln/outputs/navida_overfit_20ep/closed_loop_all_linear_ep5_chunk3/run_20260711_213417/`。原运行 CSV 同时记录两个 episode，本素材包中的 `trajectory.csv` 只保留与 `episode_000.mp4` 对应的 episode 0。

