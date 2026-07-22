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

# Move below and in front of the inclined panel while opening the gripper.
wait_sim_time 1
ros2 topic pub --once /drone/cmd_pose geometry_msgs/msg/PoseStamped \
    '{header: {frame_id: "map"}, pose: {position: {x: 0.0, y: -4.46, z: 6.20}, orientation: {w: 1.0}}}' &
drone_pub_pid=$!
ros2 topic pub --once /joint_command sensor_msgs/msg/JointState \
    '{name: ["gripper"], position: [1.0]}' &
gripper_pub_pid=$!
wait "$drone_pub_pid" "$gripper_pub_pid"

# Align the tool Z axis with the 45-degree panel normal before approaching.
wait_sim_time 10
ros2 topic pub --once /arm/cmd_pose geometry_msgs/msg/PoseStamped \
    '{header: {frame_id: "base_link"}, pose: {position: {x: 0.0, y: -0.44, z: 0.46}, orientation: {x: 0.3826834324, y: 0.0, z: 0.0, w: 0.9238795325}}}'

# Advance until the tool center reaches the underside center of the panel.
wait_sim_time 13
ros2 topic pub --once /drone/cmd_pose geometry_msgs/msg/PoseStamped \
    '{header: {frame_id: "map"}, pose: {position: {x: 0.0, y: -4.937, z: 6.657}, orientation: {w: 1.0}}}'

wait_sim_time 17
ros2 topic pub --once /joint_command sensor_msgs/msg/JointState \
    '{name: ["gripper"], position: [0.0]}'

wait "$sim_pid"
