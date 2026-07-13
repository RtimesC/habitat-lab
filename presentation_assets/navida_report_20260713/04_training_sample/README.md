# 真实训练样本示例

- episode：`00806-tQ5s4ShP627-000008`
- sample：`chunk-3`
- instruction：`Find a safe route to the target point and stop when you reach it.`
- HPAC action chunk：`move_forward × 3 → turn_left × 1`
- atomic actions：`move_forward, move_forward, move_forward, turn_left`
- chunk 开始：step 17，见 `action_chunk_start.jpg`
- chunk 执行完成后的观测：step 21，见 `action_chunk_end.jpg`
- `manifest_rows.jsonl`：第一行是对应 VLN 样本，第二行是配对 IDS 样本。

VLN 行中的 `images` 是到 chunk 开始时为止的 8 帧历史；IDS 行中的两张图分别是 chunk 开始和执行完成后的图像。JSON 内保留了原始 Linux 绝对路径用于来源追踪，PPT 应使用本目录中复制出的两张 JPG。

