[![Ubuntu 22.04](https://img.shields.io/badge/Ubuntu-22.04LTS-brightgreen.svg)](https://releases.ubuntu.com/22.04/)
[![IsaacSim 5.1.0](https://img.shields.io/badge/IsaacSim-5.1.0-brightgreen.svg)](https://developer.nvidia.com/isaac-sim)
[![PX4-Autopilot 1.14.3](https://img.shields.io/badge/PX4--Autopilot-1.14.3-brightgreen.svg)](https://github.com/PX4/PX4-Autopilot)

# Rflyarm

Rflyarm 是基于 [PegasusSimulator](https://github.com/PegasusSimulator/PegasusSimulator) 的六旋翼机械臂仿真平台，运行于 Isaac Sim 5.1.0。

## 安装

```bash
git clone https://github.com/robotswang/Rflyarm.git ~/Rflyarm
cd ~/Rflyarm
bash install.sh
```

## 运行

```bash
isaac_run ~/Rflyarm/code/examples/rflyarm/5_aerial_arm_hover.py
```

## 绘图

```bash
python3 ~/Rflyarm/code/examples/rflyarm/plot_force_torque.py
```
