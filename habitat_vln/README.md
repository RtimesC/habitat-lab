# habitat_vln：团队仿真实验层

`habitat_vln/` 是 Habitat-Lab 仓库中的团队实验层，用于 VLN 仿真、任务复现和离线评测。

## 当前优先任务

先在 Habitat-HM3D 中复现论文原始任务和动作空间：

```text
场景/episode
  -> 语言与视觉观测
  -> 原始动作接口
  -> STOP/结束条件
  -> 轨迹与指标
```

复现成功后，才开展 3D Gaussian Map、空间记忆和连续控制适配实验。

## 模块职责

- `envs/`：Habitat 场景、传感器和任务适配；
- `core/`：观察、任务和实验配置契约；
- `policies/`：策略接口和 baseline policy；
- `control/`：动作执行与局部安全控制；
- `runtime/`：闭环运行、日志和产物记录；
- `data/`：数据检查与适配；
- `evaluation/`：指标和失败分析；
- `configs/`：运行配置；
- `legacy/`：历史实验代码，不代表当前活动方法。

## 边界

这里是仿真与离线评测层，不是 VLN 主算法仓库。官方 Habitat-Lab、Habitat-Baselines 和 Habitat-HITL 代码保持稳定。不得把终点坐标、最短路径、Nav2 路径、SLAM 位姿或评价真值作为主策略输入。

详细定位、重构顺序和验收标准见：

[`docs/research/HABITAT_SIMULATION_PLATFORM_POSITIONING.md`](../docs/research/HABITAT_SIMULATION_PLATFORM_POSITIONING.md)


## 当前模块定位

| 状态 | 目录/文件 | 用途 | 当前是否属于 HM3D 原始 VLN 复现主线 |
|---|---|---|---|
| active | `core/` | 任务、观测、策略输出和实验配置契约 | 是 |
| active | `envs/` | Habitat 场景、传感器和动作适配 | 是 |
| active | `runtime/` | reset/step 闭环、轨迹、视频和运行产物 | 是 |
| active | `baselines/reactive_doornav/` | 目标无关的局部视觉动作 baseline 与契约测试 | 仅作接口 smoke test |
| active | `control/` | 动作执行和局部安全控制 | 是，限于执行层 |
| supporting | `policies/` | Mock、NaVIDA、VLM 等策略接口 | 逐项接入，不能自动视为原始论文复现 |
| supporting | `data/` | 数据检查、episode 生成和训练数据工具 | 仅在对应实验需要时使用 |
| supporting | `evaluation/` | 指标和失败分析 | 是，但必须与任务契约一致 |
| supporting | `configs/` | 运行配置 | 是，新增配置需注明任务和动作空间 |
| supporting | `tests/` | 仿真接口和回归测试 | 是 |
| supporting | `pipelines/` | 较长的 NaVIDA 训练/评测流水线 | 暂不作为第一阶段入口 |
| legacy | `legacy/` | PointNav、旧 Qwen/Habitat 脚本和历史探索 | 否 |
| legacy | `training/` | 旧训练入口，除非实验明确启用 | 否 |
| legacy | `demos/` | Grounded/NaVIDA 演示脚本 | 否，作为演示材料保留 |

### 推荐入口

第一阶段只从以下入口开始检查：

```text
PROJECT_DIRECTION.md
  -> core/
  -> envs/
  -> habitat_vln_nav.py
  -> runtime/
  -> configs/runtime/
  -> tests/test_habitat_vln_*.py
```

`pipelines/`、`training/`、`demos/`、`legacy/` 不作为 HM3D 原始任务复现的默认入口。移动这些目录前必须先更新引用和测试；当前先用文档标记职责，保持导入路径稳定。
