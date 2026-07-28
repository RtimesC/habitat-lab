# Habitat Indoor Semantic Navigation

本目录是项目自己的室内语义导航应用层。第一阶段只研究**单栋楼宇内部**：机器人
根据语言目标、经过校验的建筑弱先验和连续 RGB-D 观察，辨认自身区域、形成可验证
的连通假设，并在走错或不确定时恢复。

完整目标、边界和使用体验见 [`PROJECT_DIRECTION.md`](PROJECT_DIRECTION.md)；模块分工
见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。在设计数据、训练和实验前，必须先阅读这两份
文件。

## 唯一活动任务契约

活动策略和控制器可以使用：

- 用户语言目标；
- RGB/RGB-D、碰撞、局部深度和动作历史；
- 经过字段白名单校验的建筑弱先验。

它们不能使用终点坐标、终点距离、终点角度、成功半径、最短路径，或任何由隐藏终点
派生的进度/自动 STOP/自动转向逻辑。旧 PointNav、Oracle、NaVIDA、R2R 和 HM3D 路径
已退出活动框架，历史文件的边界见 [`legacy/README.md`](legacy/README.md)。

## 活动入口

`habitat_vln_nav.py` 只运行目标无关的直接动作闭环。它要求显式提供未来的室内语义
任务适配器配置，避免默认回落到 PointNav：

```bash
conda run -n habitat_vlm python habitat_vln/habitat_vln_nav.py \
  --task-config benchmark/nav/your_semantic_indoor_task.yaml \
  --instruction "带我去二层电梯旁的小教室区" \
  --building-prior-file path/to/building_prior.json
```

`--building-prior-file` 是一个小型 JSON 对象，只允许以下字段：

```json
{
  "building_id": "teaching-building-a",
  "start_region": "main entrance",
  "known_entrances": ["south entrance"],
  "floor_labels": ["1", "2"],
  "footprint_summary": "building extends east into a second wing",
  "public_structure_summary": "upper floors may connect the two wings",
  "known_regions": ["lecture hall", "elevator lobby"]
}
```

未知字段会被拒绝，防止目标坐标或路线答案以其他名称混入策略输入。

## 当前可验证的框架能力

在专用楼宇语义数据和 Habitat 任务适配器完成前，唯一可运行的无模型回归配置是：

```bash
conda run -n habitat_vlm python habitat_vln/habitat_vln_nav.py \
  --experiment-config habitat_vln/configs/runtime/semantic_indoor_mock.yaml \
  --mock-policy
```

该 Mock 配置只验证输入过滤、直接动作、局部深度避障、轨迹记录、帧和视频产物；它不
代表区域理解、拓扑推理或真实导航性能。

每个 `trajectory.csv` 记录模型动作、局部控制动作、碰撞、深度、动作历史和任务适配器
返回的通用环境指标。它不再记录或依赖 PointNav 目标距离、角度、`Success`、`SPL` 或
最短路径进度。

## 下一步的活动实现

1. 定义楼宇语义 episode：建筑弱先验、起始区域、语言目标、可验证语义终点和恢复标注。
2. 实现 Habitat 室内语义任务适配器，而不是复用 PointNav episode。
3. 将模型输出扩展为位置假设、证据、拓扑假设、下一验证点和动作。
4. 构造相似区域、跨楼层连通、错误恢复三类场景与相应评价器。
