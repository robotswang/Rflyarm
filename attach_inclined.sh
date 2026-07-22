#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

unset PYTHONPATH ROS_DISTRO AMENT_PREFIX_PATH COLCON_PREFIX_PATH
unset RMW_IMPLEMENTATION LD_LIBRARY_PATH VIRTUAL_ENV CONDA_PREFIX
set +u
source /opt/ros/humble/setup.bash
set -u

"$PROJECT_DIR/run_simulation.sh" --device cpu --render-hz 30 &
sim_pid=$!
trap 'kill "$sim_pid" 2>/dev/null || true' EXIT INT TERM

wait_sim_time() {
    ros2 topic echo /clock rosgraph_msgs/msg/Clock --once \
        --filter "m.clock.sec + m.clock.nanosec / 1e9 >= $1" >/dev/null
}

wait_sim_time 1
ros2 topic pub --once /drone/cmd_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: "map"}, pose: {position: {x: 0.04, y: -5.03, z: 7.05}, orientation: {w: 1.0}}}'
ros2 topic pub --once /joint_command sensor_msgs/msg/JointState \
    '{name: ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"], position: [-1.527817011, -1.492991328, 1.358749747, -0.897547543, -1.027395606, -0.012855814]}'

wait_sim_time 15
ros2 topic pub --once /drone/cmd_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: "map"}, pose: {position: {x: 0.04, y: -5.03, z: 100.0}, orientation: {w: 1.0}}}'

wait_sim_time 15.5
ros2 topic pub --once /drone/cmd_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: "map"}, pose: {position: {x: 0.04, y: -50.0, z: 100.0}, orientation: {w: 1.0}}}'

wait_sim_time 16
ros2 topic pub --once /arm/cmd_pose geometry_msgs/msg/PoseStamped \
    '{header: {frame_id: "base_link"}, pose: {position: {x: 0.03, y: 0.0, z: 0.28}, orientation: {x: 0.0, y: 0.0, z: 0.707, w: 0.707}}}'

wait_sim_time 16.5
ros2 topic pub --once /joint_command sensor_msgs/msg/JointState \
    '{name: ["gripper"], position: [0.0]}'

wait_sim_time 17
ros2 topic pub --once /drone/cmd_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: "map"}, pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'

wait "$sim_pid"
