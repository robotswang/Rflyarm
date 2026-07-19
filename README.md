[![Ubuntu 22.04](https://img.shields.io/badge/Ubuntu-22.04LTS-brightgreen.svg)](https://releases.ubuntu.com/22.04/)
[![IsaacSim 5.1.0](https://img.shields.io/badge/IsaacSim-5.1.0-brightgreen.svg)](https://developer.nvidia.com/isaac-sim)
[![PX4-Autopilot 1.14.3](https://img.shields.io/badge/PX4--Autopilot-1.14.3-brightgreen.svg)](https://github.com/PX4/PX4-Autopilot)

# Rflyarm

Rflyarm 是基于 [PegasusSimulator](https://github.com/PegasusSimulator/PegasusSimulator) 的六旋翼机械臂仿真平台，运行于 Isaac Sim 5.1.0。整机作为单一 articulation：6 螺旋桨 + 6 臂关节 + 6 夹爪关节（其中 5 个通过 mimic 联动到 `Gripper_r1`），共 18 DOF；臂 base_link 通过 fixed joint 焊接到机身。

## 安装

```bash
git clone https://github.com/robotswang/Rflyarm.git ~/Rflyarm
cd ~/Rflyarm
bash ./install.sh
```

## 运行

启动仿真后，平台自动爬升到 1.5 m 悬停；不发任何指令时相当于开环悬停。三终端：

```bash
# 终端 1：仿真
isaac_run ~/Rflyarm/code/examples/rflyarm/13_ros2_pose_control_rflyarm.py

# 终端 2（可选）：看平台实时位姿；机械臂状态在 /arm/joint_states 和 /arm/ee_pose
ros2 topic echo /drone/pose
# ros2 topic echo /arm/joint_states   # 7 字段：Joint1..6 + Gripper_r1（rad）
# ros2 topic echo /arm/ee_pose        # 末端 tool_center 在 arm base_link 系的位姿

# 终端 3：发目标位姿（ENU / 米 / 度）
# 飞到 (0, 0, 1.5)，偏航 0
ros2 topic pub --once /drone/cmd_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: "map"}, pose: {position: {x: 0.0, y: 0.0, z: 1.5}, orientation: {w: 1.0}}}'

# 飞到 (2, 3, 2)，机头转到 yaw=90°（z=sin(45°), w=cos(45°)）
ros2 topic pub --once /drone/cmd_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: "map"}, pose: {position: {x: 2.0, y: 3.0, z: 2.0}, orientation: {z: 0.7071, w: 0.7071}}}'

# 终端 4（可选）：发臂末端笛卡尔目标（Lula IK，arm base_link 系，米）
ros2 topic pub --once /arm/target_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: "base_link"}, pose: {position: {x: 0.2, y: -0.3, z: 0.4}, orientation: {w: 1.0}}}'

# 终端 5（可选）：发臂关节命令（rad，直接跳过 IK）
# 只发 Gripper_r1 → 夹爪对称闭合（mimic 联动 5 个从动关节 r2/r3/l1/l2/l3）
# 夹爪有效开合区间约 [0, 0.5] rad：0=完全张开，0.5≈完全闭合。
# 负值或 >0.5 会顶到硬限位（USD 内 ±179.9° 是关节机械极限，不是实用范围）。
ros2 topic pub --once /arm/joint_command sensor_msgs/msg/JointState \
  '{name: ["Gripper_r1"], position: [0.5]}'    # 闭合
ros2 topic pub --once /arm/joint_command sensor_msgs/msg/JointState \
  '{name: ["Gripper_r1"], position: [0.0]}'    # 张开

# 命令 6 臂关节到位（每个关节实用区间通常 ±1.5 rad 内即可，别贴 ±π）
ros2 topic pub --once /arm/joint_command sensor_msgs/msg/JointState \
  '{name: ["Joint1","Joint2","Joint3","Joint4","Joint5","Joint6"], position: [0.0, -0.5, 0.5, 0.0, 0.5, 0.0]}'
```

坐标系 ENU 世界系（X 东、Y 北、Z 上），单位米，yaw 单位度。`z` 保持 ≥1 m 以免触地。姿态四元数 `(x, y, z, w)`，绕 Z 旋转 yaw：`z=sin(yaw/2)`、`w=cos(yaw/2)`。

话题：

| 方向 | 话题 | 类型 | QoS | 说明 |
|---|---|---|---|---|
| 外部 → 平台 | `/drone/cmd_pose` | `geometry_msgs/PoseStamped` | default | 目标飞行位姿；`ROS2PoseControllerRflyarm` 驱动 6 桨 |
| 外部 → 平台 | `/arm/target_pose` | `geometry_msgs/PoseStamped` | default | 末端（`tool_center`）目标笛卡尔位姿，**arm base_link 系**；Lula 解 IK 后驱动 `Joint1..6`（不动夹爪） |
| 外部 → 平台 | `/arm/joint_command` | `sensor_msgs/JointState` | default | 直接关节角命令（`Joint1..6`、`Gripper_r1`；发 `Gripper_r1` 联动 5 个从动关节；未知关节名会被警告并忽略） |
| 平台 → 外部 | `/drone/pose` | `geometry_msgs/PoseStamped` | sensor_data | 平台当前位姿（世界系） |
| 平台 → 外部 | `/arm/joint_states` | `sensor_msgs/JointState` | sensor_data | 可命令关节的实时角度，共 7 个字段：`Joint1..6` + `Gripper_r1`（6 桨和 5 个 mimic 从动关节不外露） |
| 平台 → 外部 | `/arm/ee_pose` | `geometry_msgs/PoseStamped` | sensor_data | 末端 `tool_center` 当前位姿，**arm base_link 系**（用 Lula FK 每 tick 算，启动即发） |

三条 backend 并行：`ROS2PoseControllerRflyarm`（backends[0]）算 6 桨转速；`ROS2ArmController`（backends[1]）接 `/arm/joint_command`，缓存到 `_targets` dict（保留式：未出现的键保持旧值），每物理步 drain 订阅队列后统一 `set_dof_position_target` 下发到 PhysX drive，PhysX 内建 PD 用 USD 里的 `stiffness/damping` 拉过去（backend 层无 PID / 无轨迹），并回读 7 关节角发布 `/arm/joint_states`；`ROS2ArmIKController`（backends[2]）接 `/arm/target_pose`，Lula 解一次 IK（HOME 种子，稳定收敛，不重解），把 6 主关节角写回 articulation，同时每 tick 用 Lula FK 发布 `/arm/ee_pose`。不发命令时臂保持零位。

## 绘图

```bash
python3 ~/Rflyarm/code/examples/rflyarm/plot_force_torque.py
```

## 目录结构

```
Rflyarm/
├── README.md
├── install.sh                        # 只装 USD 资产 + hexrotor.py + params.py 注册
├── assets/
│   ├── Robots/Rflyarm/               # USD 资产（入口 rflyarm.usda，含 robot_arm_flat.usda）
│   └── robot_arm_urdf/               # Lula IK 用的 URDF + STL meshes + robot_description.yaml
└── code/
    ├── examples/
    │   ├── rflyarm/                  # 启动脚本（isaac_run 直接从这里跑）
    │   │   ├── 13_ros2_pose_control_rflyarm.py
    │   │   └── plot_force_torque.py
    │   └── utils/                    # 控制器（脚本用 sys.path 相对 __file__ 加载）
    │       ├── nonlinear_controller_arm.py
    │       ├── ros2_pose_controller_rflyarm.py
    │       ├── ros2_arm_controller.py         # /arm/joint_command
    │       └── ros2_arm_ik_controller.py      # /arm/target_pose (Lula IK)
    └── logic_vehicles/hexrotor.py    # 机型类（→ 装进 Pegasus）
```

## 常见坑

- **`ros2` 找不到 / `_rclpy_pybind11` 报错**：终端 2/3 用之前 `unset PYTHONPATH && source /opt/ros/humble/setup.bash`，避免 conda Python 串进来。
- **平台完全不动**：`ROS2PoseControllerRflyarm` 必须是 `backends[0]`；`multirotor.py` 只用第一个后端驱动桨。
- **臂关节收不到命令**：`ROS2ArmController` 首次 `update` 时通过 `dc.get_articulation("/World/rflyarm")` 拿句柄；若 stage 里 prim 路径不同（`Hexrotor(stage_prefix=...)` 改过）就要同步改 `articulation_path` 参数。
- **平台起飞后立刻下坠**：臂增加了约 6.25 kg，总质量约 10 kg；若飞行不稳，把 `code/examples/utils/nonlinear_controller_arm.py` 里 `self.m = 4.30` 上调（脚本从仓库跑，改完直接生效）。
- **突然乱飞**：目标高度太低触地发散；先给 `0 0 1.5` 确认能稳，再逐步移动。
