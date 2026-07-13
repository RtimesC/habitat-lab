# 给 Mac 上 Codex 的指令

请使用当前仓库 `presentation_assets/navida_report_20260713/` 中的素材制作或更新 NaVIDA/Habitat 项目汇报 PPT。

要求：

1. 把 `01_videos/advisor_success.mp4` 作为主成果视频，标题写成“Qwen Advisor + Geometric Controller：成功接近并停止”。旁边展示：Success=1、SPL=0.9995、碰撞=0、最终距离=0.087 m。
2. 用 `01_videos/navida_pure_failure.mp4` 解释当前瓶颈：纯 NaVIDA 最低距离达到 0.172 m，已经进入 0.2 m 成功半径，但没有及时 STOP，随后最终距离变为 0.375 m，并累计 11 个 collision steps。
3. 可选加入 `01_videos/navida_guard_success.mp4` 作为同 episode 的诊断对照。必须明确写：guard 使用 Habitat 提供的真实目标距离，在 0.2 m 成功半径内强制 STOP；该结果不代表纯模型性能。指标：Success=1、SPL=0.5086、碰撞=8、最终距离=0.172 m。
4. 用 `04_training_sample/action_chunk_start.jpg` 和 `action_chunk_end.jpg` 替换抽象 NaVIDA 示意图；标注 instruction 为 “Find a safe route to the target point and stop when you reach it.”，HPAC chunk 为 `forward × 3 → left × 1`。`manifest_rows.jsonl` 中第一行是 VLN 样本，第二行是 IDS 样本。
5. 数据分析页直接读取 `03_300_episode_analysis/coverage.json`、`coverage_cells.csv`、`coverage.md`、`summary.json` 和 `episodes.csv`，不要从截图手抄数据。重点数字：300 episodes、10 scenes、Oracle 300/300 success、20,165 atomic samples、3,000 mixed samples（1,500 VLN + 1,500 IDS）、core control cells 20/20。
6. 所有结论注明“HM3D engineering protocol, not R2R/RxR benchmark”。不要把 guard 指标与 pure-policy 指标混合，也不要把 HM3D smoke 结果写成论文 benchmark reproduction。
7. 如果需要核对细节，使用 `02_episode_outputs/*/trajectory.csv`、`metrics.json` 和 `parameters.md`；不要修改这些原始证据文件。

输出更新后的 `.pptx`，并同时给出一页简短的素材来源/实验边界说明。

