[![Ubuntu 22.04](https://img.shields.io/badge/Ubuntu-22.04_LTS-brightgreen.svg)](https://releases.ubuntu.com/22.04/)
[![Isaac Sim 6.0.1](https://img.shields.io/badge/Isaac_Sim-6.0.1-brightgreen.svg)](https://developer.nvidia.com/isaac-sim)
[![Isaac Lab 3.0.0 Beta 2](https://img.shields.io/badge/Isaac_Lab-3.0.0_Beta_2-orange.svg)](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/)
[![ROS 2 Humble](https://img.shields.io/badge/ROS_2-Humble-blue.svg)](https://docs.ros.org/en/humble/)

# Rflyarm

Rflyarm 是基于 Isaac Sim 6.0.1 和 Isaac Lab 3.0.0 Beta 2 的六旋翼飞行机械臂室内仿真平台。

## 安装

按 [Isaac Lab 3.0.0 Beta 2 官方二进制安装教程](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/setup/installation/binaries_installation.html) 安装 Isaac Sim 6.0.1 和 Isaac Lab，然后克隆本项目：

```bash
git clone https://github.com/robotswang/Rflyarm.git
```

## 运行

```bash
cd ~/Rflyarm
./run_simulation.py
```

平台启动后会闭环飞往 `map` 坐标系下的 `(0, 0, 1.5)` m 并悬停。
运行 `./replace_bulb.py` 可启动自动换灯泡演示。
拆卸演示输出完成提示后保持其进程运行，再在另一终端运行 `./install_bulb.py`，
将夹爪中已经取下的灯泡重新安装。

运行 `./attach_inclined.py` 可启动斜面附着测试。

## ROS 2 控制

仿真端由 `run_simulation.py` 自动切换到 Isaac Lab，并使用 Isaac Sim 自带的
ROS 2 Humble Python 3.12 环境。

| 方向 | 话题 | 类型 | 说明 |
|---|---|---|---|
| 输入 | `/drone/cmd_pose` | `geometry_msgs/msg/PoseStamped` | `map` 下的飞行目标位姿 |
| 输入 | `/joint_command` | `sensor_msgs/msg/JointState` | `joint_1..6` 或 `gripper` 位置命令 |
| 输入 | `/arm/cmd_pose` | `geometry_msgs/msg/PoseStamped` | `base_link` 下的末端目标位姿 |
| 输出 | `/clock` | `rosgraph_msgs/msg/Clock` | 由物理步长推进的 ROS 仿真时间 |
| 输出 | `/drone/pose` | `geometry_msgs/msg/PoseStamped` | `map` 下的无人机位置与四元数姿态 |
| 输出 | `/joint_states` | `sensor_msgs/msg/JointState` | 关节位置、速度和力/力矩 |
| 输出 | `/arm/ee_pose` | `geometry_msgs/msg/PoseStamped` | Lula FK 计算的 `tool_center` 位姿 |
| 输出 | `/depth_camera/{color,depth}/image_raw` | `sensor_msgs/msg/Image` | 320×240 RGB 与 32FC1 深度图（15 Hz） |
| 输出 | `/depth_camera/{color,depth}/camera_info` | `sensor_msgs/msg/CameraInfo` | 相机内参 |
| 输出 | `/tf`、`/tf_static` | `tf2_msgs/msg/TFMessage` | `map → body → depth_camera_optical_frame` |

需要使用 ROS 定时器的外部节点应设置参数 `use_sim_time:=true`；状态消息的
`header.stamp` 与 `/clock` 均使用仿真时间，界面运行速度则由墙钟时间单独衡量。

## 项目架构

```text
Rflyarm/
├── run_simulation.py             # 唯一启动入口、环境自举与 Isaac Lab 仿真主循环
├── replace_bulb.py               # Python ROS 2 自动换灯泡状态机
├── install_bulb.py               # 连接现有仿真的灯泡重新安装状态机
├── attach_inclined.py            # Python ROS 2 斜面附着状态机
├── simulation/
│   ├── scene.py                  # 场景、机器人与深度相机配置
│   ├── flight_controller.py      # 六旋翼飞行控制器
│   ├── aerial_manipulator.py     # 推力与力矩动力学
│   ├── arm_controller.py         # 机械臂和夹爪控制
│   ├── arm_kinematics.py         # Lula 正逆运动学
│   ├── rotor_visualizer.py       # 旋翼视觉旋转同步
│   └── ros2_interface.py         # ROS 2 话题接口
├── assets/
│   ├── Collected_World/
│   │   ├── World0.usd            # Collect 生成的 Kit 场景快照
│   │   └── SubUSDs/
│   │       ├── World.usd         # 仿真实际加载的组合场景
│   │       ├── simple_room.usd   # SimpleRoom 环境
│   │       ├── inclined_panel.usda # 斜面附着测试板
│   │       ├── target_bulb.usd   # 目标灯泡
│   │       └── rflyarm.usda      # 飞行机械臂
│   └── kinematics/
│       ├── arm.urdf              # 机械臂运动学模型
│       └── robot_description.yaml
├── tests/
│   ├── flight_response.py        # 飞控响应测试
│   └── ros2_smoke.py             # ROS 2 接口测试
└── README.md
```
