# Advisor Success 参数说明

- 策略：Qwen advisor + geometric controller。
- 基础模型：`Qwen/Qwen2.5-VL-3B-Instruct`。
- 环境：`habitat_vlm`，Habitat-Sim 0.3.3。
- 场景：HM3D example scene `00337-CFVBbU9Rsyb`。
- instruction：`Navigate to the target location and stop when you reach it.`
- advisor 输出：`stop_if_reached`；低层动作由几何控制器根据目标方向与距离决定。
- 成功半径：0.2 m。
- 原始视频：224 × 224，10 fps，20 帧。
- 汇报视频：动作顺序不变，放慢到 21 秒并放大为 672 × 672。

原始运行目录：`habitat_vln/outputs/run_20260709_174058/`。该旧运行没有保存完整 CLI 命令，因此这里仅记录能从代码、轨迹和视频中确认的参数，不补猜未保存的选项。
