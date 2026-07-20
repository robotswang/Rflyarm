[![Ubuntu 22.04](https://img.shields.io/badge/Ubuntu-22.04LTS-brightgreen.svg)](https://releases.ubuntu.com/22.04/)
[![Isaac Sim 5.1.0](https://img.shields.io/badge/IsaacSim-5.1.0-brightgreen.svg)](https://developer.nvidia.com/isaac-sim)
[![ROS 2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)

# Rflyarm

Rflyarm 是基于 PegasusSimulator 和 Isaac Sim 5.1.0 的六旋翼机械臂仿真平台。

## 安装

请先按照 [Pegasus Simulator 官方文档](https://pegasussimulator.github.io/PegasusSimulator/source/setup/installation.html) 完成安装。

```bash
git clone https://github.com/robotswang/Rflyarm.git ~/Rflyarm
cd ~/Rflyarm
bash install.sh
```

`install.sh` 将运行资产和六旋翼机型类安装到 PegasusSimulator。

## 运行

```bash
isaac_run ~/Rflyarm/run_simulation.py
```

平台启动后闭环爬升到 `map` 坐标系下的 `(0, 0, 1.5)` m 并悬停。

```bash
# 查看飞行平台和机械臂状态
ros2 topic echo /drone/pose
ros2 topic echo /joint_states
ros2 topic echo /arm/ee_pose

# 飞行目标：位置单位 m，姿态为四元数
ros2 topic pub --once /drone/cmd_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: "map"}, pose: {position: {x: 0.0, y: 0.0, z: 1.5}, orientation: {w: 1.0}}}'

# 六个机械臂关节：单位 rad
ros2 topic pub --once /joint_command sensor_msgs/msg/JointState \
  '{name: ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"], position: [0.0, -0.5, 0.5, 0.0, 0.5, 0.0]}'

# 单自由度夹爪：0 rad 张开，0.5 rad 闭合
ros2 topic pub --once /joint_command sensor_msgs/msg/JointState \
  '{name: ["gripper"], position: [0.5]}'

# 机械臂末端位姿控制
ros2 topic pub --once /arm/cmd_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: "base_link"}, pose: {position: {x: 0.0, y: 0.0, z: 0.6}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}'
```

## ROS 2 话题

| 方向 | 话题 | 类型 | 说明 |
|---|---|---|---|
| 输入 | `/drone/cmd_pose` | `geometry_msgs/msg/PoseStamped` | `map` 坐标系下的飞行目标位姿 |
| 输入 | `/joint_command` | `sensor_msgs/msg/JointState` | `joint_1..6` 或 `gripper` 位置命令，单位 rad |
| 输入 | `/arm/cmd_pose` | `geometry_msgs/msg/PoseStamped` | `base_link` 坐标系下的末端目标全位姿 |
| 输出 | `/drone/pose` | `geometry_msgs/msg/PoseStamped` | 约 60 Hz；`map` 坐标系下的无人机位置与四元数姿态 |
| 输出 | `/joint_states` | `sensor_msgs/msg/JointState` | 约 60 Hz；关节位置、速度和力/力矩 |
| 输出 | `/arm/ee_pose` | `geometry_msgs/msg/PoseStamped` | 约 60 Hz；Lula FK 计算的 `tool_center` 当前位姿 |

## 项目架构

`run_simulation.py` 是 GUI 仿真入口，`simulation/` 存放飞行、机械臂和 ROS 2 运行逻辑，`assets/` 存放 PhysX/USD 模型与 Lula 运动学资产。

```text
Rflyarm/
├── README.md
├── install.sh
├── run_simulation.py                 # GUI 仿真入口
├── simulation/
│   ├── __init__.py
│   ├── hexrotor.py                   # Pegasus 六旋翼机型
│   ├── geometric_controller.py       # 底层几何飞行控制
│   ├── flight_controller.py          # ROS 2 飞行位姿控制
│   ├── arm_controller.py             # 关节、夹爪和笛卡尔控制
│   ├── arm_kinematics.py             # Lula FK/IK
│   └── pose_publisher.py             # /drone/pose 位姿发布
└── assets/
    ├── rflyarm.usda                  # 整机 USD 入口
    ├── arm.usda                      # 机械臂 USD 资产
    ├── propeller.usd                 # 螺旋桨模型
    └── kinematics/
        ├── arm.urdf                  # Lula 运动学 URDF
        ├── robot_description.yaml    # Lula 配置
        └── meshes/                   # URDF 网格
```
