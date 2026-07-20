#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PEGASUS_DIR="${PEGASUS_ROOT:-$HOME/PegasusSimulator}"
SIMULATOR_DIR="$PEGASUS_DIR/extensions/pegasus.simulator/pegasus/simulator"
ASSET_DIR="$SIMULATOR_DIR/assets/Robots/Rflyarm"
PARAMS_FILE="$SIMULATOR_DIR/params.py"

if [[ ! -f "$PARAMS_FILE" ]]; then
    echo "[ERROR] PegasusSimulator not found: $PEGASUS_DIR" >&2
    exit 1
fi

install -Dm644 "$REPO_DIR/assets/rflyarm.usda" "$ASSET_DIR/rflyarm.usda"
install -Dm644 "$REPO_DIR/assets/arm.usda" "$ASSET_DIR/arm.usda"
install -Dm644 "$REPO_DIR/assets/propeller.usd" "$ASSET_DIR/propeller.usd"
install -Dm644 "$REPO_DIR/simulation/hexrotor.py" "$SIMULATOR_DIR/logic/vehicles/hexrotor.py"

python3 - "$PARAMS_FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
entry = '    "Rflyarm": ROBOTS_ASSETS + "/Rflyarm/rflyarm.usda"'
anchor = '    "Pegasus": ROBOTS_ASSETS + "/Pegasus/pegasus.usd"'

if entry not in text:
    if anchor not in text:
        raise RuntimeError("Pegasus ROBOTS registration anchor not found")
    path.write_text(text.replace(anchor, anchor + ",\n" + entry, 1))
PY

echo "[OK] Installed Rflyarm into: $SIMULATOR_DIR"
