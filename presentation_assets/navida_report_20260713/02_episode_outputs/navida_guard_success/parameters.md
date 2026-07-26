# NaVIDA Guard Success 参数说明

- 策略：与 pure failure 相同的 NaVIDA action-chunk controller。
- 额外诊断设置：`--force-stop-within-success-radius`。
- guard 使用 Habitat 提供的真实目标距离，在距离小于等于 0.2 m 时清空动作队列并强制 STOP。
- 基础模型：`Qwen/Qwen2.5-VL-3B-Instruct`。
- adapter：`habitat_vln/outputs/03_navida_overfit/navida_overfit_20ep/qlora_100_all_linear_ep5/final_adapter/`。
- chunk 执行上限：3；历史图像上限：8；最大 episode 步数：80。
- 场景与 pure failure 相同：HM3D episode 7。
- 原始视频：224 × 224，10 fps，59 帧。
- 汇报视频：动作顺序不变，3 倍慢放到约 18 秒并放大为 672 × 672。

重要：最后一步模型仍给出 `move_forward`，实际 STOP 来自 privileged guard。因此这个案例只能作为诊断对照，不能计入纯模型性能。

原始运行目录：`habitat_vln/outputs/03_navida_overfit/navida_overfit_20ep/closed_loop_guarded_chunk3/run_20260711_213607/`。素材包中的 `trajectory.csv` 只保留 episode 0。

