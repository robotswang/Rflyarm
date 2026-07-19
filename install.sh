#!/usr/bin/env bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PEGASUS="${PEGASUS_ROOT:-$HOME/PegasusSimulator}"
PEGASUS_EXT="$PEGASUS/extensions/pegasus.simulator/pegasus/simulator"
PARAMS="$PEGASUS_EXT/params.py"

echo "==> Rflyarm installer"
echo "    repo:    $REPO_DIR"
echo "    pegasus: $PEGASUS"
echo ""

if [ ! -d "$PEGASUS" ]; then
    echo "[ERROR] PegasusSimulator not found at: $PEGASUS"
    echo "        Set PEGASUS_ROOT to override, e.g.:"
    echo "        PEGASUS_ROOT=/path/to/PegasusSimulator bash install.sh"
    exit 1
fi

# 只把 Pegasus 包 import 必须找到的东西装进去；启动脚本/utils/客户端留在仓库里跑。

# ── Step 1：USD 资产（ROBOTS 字典解析成 $PEGASUS_EXT/assets/... 绝对路径） ──
echo "[1/3] Copying Rflyarm USD assets..."
cp -r "$REPO_DIR/assets/Robots/Rflyarm" \
      "$PEGASUS_EXT/assets/Robots/"

# ── Step 2：Hexrotor 机型类（被 examples 里 `from pegasus.simulator.logic.vehicles.hexrotor import` 引用） ──
echo "[2/3] Copying hexrotor vehicle logic..."
cp "$REPO_DIR/code/logic_vehicles/hexrotor.py" \
   "$PEGASUS_EXT/logic/vehicles/hexrotor.py"

# ── Step 3：在 params.py 注册 Rflyarm 机型 ──────────────────────────
echo "[3/3] Registering Rflyarm in params.py..."

if grep -q '"Rflyarm"' "$PARAMS"; then
    echo "      already registered, skipping."
else
    python3 - <<PYEOF
import pathlib

path = pathlib.Path("$PARAMS")
text = path.read_text()

text = text.replace(
    '"Pegasus": ROBOTS_ASSETS + "/Pegasus/pegasus.usd"',
    '"Pegasus": ROBOTS_ASSETS + "/Pegasus/pegasus.usd",\n    "Rflyarm": ROBOTS_ASSETS + "/Rflyarm/rflyarm.usda"'
)

text = text.replace(
    '"Pegasus": ROBOTS_ASSETS + "/Pegasus/pegasus_thumbnail.png"',
    '"Pegasus": ROBOTS_ASSETS + "/Pegasus/pegasus_thumbnail.png",\n    "Rflyarm": ROBOTS_ASSETS + "/Rflyarm/rflyarm_thumbnail.png"'
)

path.write_text(text)
print("      params.py updated.")
PYEOF
fi

echo ""
echo "[OK] Installation complete."
echo ""
echo "Run the example directly from the repo (utils/ resolved via sys.path relative to script):"
echo "  isaac_run $REPO_DIR/code/examples/rflyarm/13_ros2_pose_control_rflyarm.py   # ROS2 pose control"
echo ""
echo "Command via ROS2 (system py3.10, after 'unset PYTHONPATH && source /opt/ros/humble/setup.bash'):"
echo "  ros2 topic pub --once /drone/cmd_pose  geometry_msgs/msg/PoseStamped   ..."
echo "  ros2 topic pub --once /arm/joint_command sensor_msgs/msg/JointState  ..."
