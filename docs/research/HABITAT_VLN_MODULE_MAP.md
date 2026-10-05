# habitat_vln 模块整理记录

## 整理决定

当前先采用“文档定位、代码不搬动”的整理方式。原因是测试、流水线和历史入口仍直接引用现有 Python import path；贸然移动目录会改变实验入口，不能证明任务复现正确。

## active：复现闭环的最小核心

- `core/`：定义 `NavigationObservation`、`PolicyOutput`、实验配置和任务契约；
- `envs/`：把 Habitat 观测和动作适配到团队接口；
- `runtime/`：运行 episode、执行动作、写入轨迹/视频/日志；
- `habitat_vln_nav.py`：活动命令入口；
- `control/`：动作执行和局部安全控制；
- `baselines/reactive_doornav/`：当前可测试的局部动作 baseline，不代表完整建筑级 VLN；
- `configs/runtime/`：活动运行配置；
- `test/test_habitat_vln_*.py`：仓库级契约和回归测试。

## supporting：按实验启用

- `policies/`：策略适配层；
- `data/`：数据检查和 episode/训练数据工具；
- `evaluation/`：指标与失败分析；
- `tests/`：`habitat_vln` 内部测试；
- `pipelines/`：较长的 NaVIDA 训练/评测链路。

这些模块需要在任务契约明确后启用，不能因为文件存在就视为第一阶段复现完成。

## legacy：历史材料

- `legacy/`：PointNav、旧导航脚本和历史实验；
- `training/`：旧 QLoRA/训练入口；
- `demos/`：Grounded/NaVIDA 演示。

legacy 代码保留用于追溯，不作为当前 HM3D 原始任务与动作空间复现入口。

## 下一步整理顺序

1. 在 `core/`、`envs/`、`runtime/` 中补齐 HM3D/VLN 任务契约；
2. 将 active 配置和 smoke test 与论文原始动作空间对应起来；
3. 为 supporting 模块增加“适用任务”和“输入是否含特权信息”的说明；
4. 复现闭环稳定后，再决定是否把 NaVIDA/旧训练链路移到单独的历史包；
5. 任何实际移动都必须先更新 import、测试和运行文档。
