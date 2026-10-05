# Habitat VLN 项目核心上下文

## 项目定位

本仓库是团队的 Habitat 仿真与离线评测平台，不是 VLN 主算法仓库。

官方平台目录保持稳定：
habitat-lab/
habitat-baselines/
habitat-hitl/

团队实验代码集中在 habitat_vln/。

## 当前主任务

先在 Habitat-HM3D 中复现 VLN 论文原始任务和动作空间：

HM3D 场景和 VLN episode
→ RGB/RGB-D 与语言观测
→ 原始动作空间
→ STOP/结束条件
→ 轨迹、指标和失败分析

在这个闭环稳定前，不加入 3DGS、空间记忆或新的连续控制方法。

## 研究边界

- 不把终点坐标、最短路径、Nav2 路径、SLAM 位姿或评价真值输入主策略；
- 不自动下载 HM3D 等大型数据集；
- 不擅自运行长时间仿真或训练；
- 不把 Habitat 仿真成功表述为真实小车部署成功；
- 原始离散或 waypoint 动作和小车 v、omega、stop 控制分开记录。

## 当前代码入口

优先查看：

habitat_vln/core/
habitat_vln/envs/
habitat_vln/runtime/
habitat_vln/control/
habitat_vln/configs/runtime/
habitat_vln/habitat_vln_nav.py
test/test_habitat_vln_*.py

habitat_vln/policies/、data/、evaluation/、pipelines/ 是按实验启用的支撑模块。
legacy/、training/、demos/ 是历史或辅助材料。

## 新对话启动方式

先读取本文件、AGENTS.md、habitat_vln/README.md 和相关任务计划，再回答：

1. 当前任务属于仿真平台、原始 VLN 复现还是研究方法实验；
2. 需要查看哪些文件；
3. 如何做最小验证；
4. 哪些信息不能进入策略输入。
