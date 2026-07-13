# NaVIDA 汇报素材包

这是一份可直接提交到 Git 并同步到 Mac 的 PPT 素材包。素材来自已有 HM3D 实验，不包含新下载或新增长时仿真。

## 推荐使用顺序

1. `01_videos/advisor_success.mp4`：主成果演示，Qwen advisor + geometric controller 成功接近目标并 STOP。
2. `01_videos/navida_pure_failure.mp4`：纯 NaVIDA 已进入 0.2 m 成功半径，但没有及时 STOP，随后偏离并发生碰撞。
3. `01_videos/navida_guard_success.mp4`：同一 pure failure episode 加入真实目标距离 guard 后，在成功半径内停止。

## 关键指标

| 案例 | Success | SPL | 碰撞次数 | 最低距离 | 最终距离 | 说明 |
|---|---:|---:|---:|---:|---:|---|
| advisor success | 1 | 0.9995 | 0 | 0.087 m | 0.087 m | Qwen advisor 给高层建议，几何控制器执行并 STOP |
| NaVIDA pure failure | 0 | 0.0000 | 11 | 0.172 m | 0.375 m | 进入成功半径但没有执行 STOP |
| NaVIDA guard success | 1 | 0.5086 | 8 | 0.172 m | 0.172 m | guard 使用真实目标距离强制 STOP |

碰撞次数按 `trajectory.csv` 中 `collision=True` 的 step 数统计。guard 是诊断结果，不代表纯模型性能。

## 目录

- `01_videos/`：适合直接插入 PPT 的慢放版视频，约 21.0、15.9、17.6 秒。
- `02_episode_outputs/`：每个案例的原始视频、episode 专属 `trajectory.csv`、逐帧图片、指标和参数说明。
- `03_300_episode_analysis/`：300-episode mixed coverage、mixed summary 和 Oracle episode 原始结果。
- `04_training_sample/`：真实 HPAC 动作块的开始/结束图像和成对 VLN/IDS manifest 行。
- `MAC_CODEX_PROMPT.md`：同步到 Mac 后可直接交给 Codex 的 PPT 制作指令。

## 结果边界

这些结果属于 HM3D engineering/smoke protocol，不是 R2R/RxR benchmark 成绩。PPT 中不要将其写成论文 benchmark reproduction。
