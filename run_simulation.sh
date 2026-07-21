#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
ISAACLAB_DIR="${ISAACLAB_PATH:-$WORKSPACE_DIR/IsaacLab}"
ISAACSIM_DIR="${ISAACSIM_PATH:-$WORKSPACE_DIR/isaacsim}"

if [[ ! -x "$ISAACLAB_DIR/isaaclab.sh" ]]; then
    echo "Isaac Lab launcher not found: $ISAACLAB_DIR/isaaclab.sh" >&2
    exit 2
fi
if [[ ! -f "$ISAACSIM_DIR/setup_ros_env.sh" ]]; then
    echo "Isaac Sim ROS environment not found: $ISAACSIM_DIR/setup_ros_env.sh" >&2
    exit 2
fi

# Isaac Sim 6 runs Python 3.12 and ships a matching ROS 2 backend.  Remove any
# system ROS/Python 3.10 paths inherited from the interactive shell first.
unset PYTHONPATH ROS_DISTRO AMENT_PREFIX_PATH COLCON_PREFIX_PATH
unset RMW_IMPLEMENTATION LD_LIBRARY_PATH VIRTUAL_ENV CONDA_PREFIX
set +u
source "$ISAACSIM_DIR/setup_ros_env.sh"
set -u

visualizer_selected=false
for arg in "$@"; do
    case "$arg" in
        --viz | --visualizer | --viz=* | --visualizer=* | --headless)
            visualizer_selected=true
            break
            ;;
    esac
done

if [[ "$visualizer_selected" == false ]]; then
    set -- --viz kit "$@"
fi

exec "$ISAACLAB_DIR/isaaclab.sh" -p "$PROJECT_DIR/run_simulation.py" "$@"
