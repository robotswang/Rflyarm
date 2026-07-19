#!/usr/bin/env python3
"""Scale the whole flight stack around a target body mass.

Usage:
    python3 tools/set_body_mass.py               # default 50 kg
    python3 tools/set_body_mass.py 50            # explicit
    python3 tools/set_body_mass.py 4             # restore original
    python3 tools/set_body_mass.py 100

Scaling factor  s = mass / BASE_MASS  (BASE_MASS = 4 kg).  Everything is
rewritten from the fixed baseline every run, so the script is idempotent —
running it twice with the same argument leaves the files identical, and any
argument can undo any previous run.

What gets touched (5 files):

 1. assets/Robots/Rflyarm/rflyarm.usda
      body physics:mass         = 4       * s
      body physics:diagonalInertia = (0.5, 0.5, 1.0) * s
    (rotor mass 0.05 left alone -- matched by value 4 only)

 2. code/logic_vehicles/hexrotor.py
      rotor_constant             = 5e-5   * s    (thrust per omega^2 scales with total weight)
      rolling_moment_coefficient = 1e-6   * s

 3. code/examples/utils/nonlinear_controller_arm.py
      self.m                     = 4.30   * s + 6.25  (approx: scaled body + fixed arm mass)
      (This keeps the arm's real mass in the model. If you want a "body-dominated"
      assumption, set FLAG_INCLUDE_ARM_IN_M = False below.)

 4. code/examples/rflyarm/13_ros2_pose_control_rflyarm.py
      Kp                = 6.0    * s
      Kd                = 9.0    * s
      Ki                = 0.3    * s
      Kr                = 12.0   * s
      Kw                = 5.0    * s
    (Position outer loop: Kp/Kd/Ki scale linearly with mass -> maintain natural freq.
     Attitude inner loop: Kr/Kw scale with inertia which also grows ~linearly here.)

If a matching copy exists under $PEGASUS_ROOT (default ~/PegasusSimulator),
that copy is patched too so Isaac picks the change on next launch without
re-running install.sh.
"""
import os
import re
import sys
from pathlib import Path

DEFAULT_MASS = 50.0
BASE_MASS = 4.0             # body mass in the original USD
ARM_FIXED_MASS = 6.25       # robot_arm_flat.usda base_link mass (approx)
FLAG_INCLUDE_ARM_IN_M = True

REPO = Path(__file__).resolve().parents[1]
PEG = Path(os.environ.get(
    "PEGASUS_ROOT",
    str(Path.home() / "PegasusSimulator")
)) / "extensions/pegasus.simulator/pegasus/simulator"

# ---- baseline values (never edit these; the script rebuilds from them) ------
BASE_INERTIA = (0.5, 0.5, 1.0)
BASE_ROTOR_K = 5e-5
BASE_ROLLING_M = 1e-6
BASE_KP, BASE_KD, BASE_KI = 6.0, 9.0, 1.2  # Ki bumped 4x from 0.3: needed to null out the arm's ~44 Nm lateral moment in reasonable time (see ros2_pose_controller_rflyarm.py _int_band notes)
BASE_KR, BASE_KW = 12.0, 5.0


def _rewrite(path: Path, patterns):
    """Apply an ordered list of (regex, replacement, description) to path."""
    if not path.is_file():
        print(f"[skip] {path}")
        return
    txt = path.read_text()
    for rx, repl, desc in patterns:
        new_txt, n = re.subn(rx, repl, txt, count=1)
        if n == 0:
            print(f"[warn] {desc}: no match in {path.name}")
        txt = new_txt
    path.write_text(txt)
    print(f"[ok]  {path}")


def patch_usd(path: Path, mass: float, inertia):
    ix, iy, iz = inertia
    _rewrite(path, [
        (
            r"(def Xform \"body\"[\s\S]*?float physics:mass = )[0-9.eE+\-]+",
            rf"\g<1>{mass}",
            "body mass",
        ),
        (
            r"(def Xform \"body\"[\s\S]*?float3 physics:diagonalInertia = )"
            r"\([0-9., eE+\-]+\)",
            rf"\g<1>({ix}, {iy}, {iz})",
            "body inertia",
        ),
    ])


def patch_hexrotor(path: Path, rotor_k: float, rolling_m: float):
    _rewrite(path, [
        (
            r'("rotor_constant":\s*)\[[^]]*\]',
            rf"\g<1>[{rotor_k}, {rotor_k}, {rotor_k}, {rotor_k}, {rotor_k}, {rotor_k}]",
            "rotor_constant",
        ),
        (
            r'("rolling_moment_coefficient":\s*)\[[^]]*\]',
            rf"\g<1>[{rolling_m}, {rolling_m}, {rolling_m}, {rolling_m}, {rolling_m}, {rolling_m}]",
            "rolling_moment_coefficient",
        ),
    ])


def patch_controller(path: Path, m_effective: float):
    _rewrite(path, [
        (
            r"(self\.m\s*=\s*)[0-9.eE+\-]+",
            rf"\g<1>{m_effective}",
            "controller mass",
        ),
    ])


def patch_launch(path: Path, Kp, Kd, Ki, Kr, Kw):
    def _kvec(v):
        return f"[{v}, {v}, {v}]"
    _rewrite(path, [
        (r"(Kp=)\[[^]]*\]", rf"\g<1>{_kvec(Kp)}", "Kp"),
        (r"(Kd=)\[[^]]*\]", rf"\g<1>{_kvec(Kd)}", "Kd"),
        (r"(Ki=)\[[^]]*\]", rf"\g<1>{_kvec(Ki)}", "Ki"),
        (r"(Kr=)\[[^]]*\]", rf"\g<1>{_kvec(Kr)}", "Kr"),
        (r"(Kw=)\[[^]]*\]", rf"\g<1>{_kvec(Kw)}", "Kw"),
    ])


def main():
    mass = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MASS
    s = mass / BASE_MASS
    inertia = tuple(round(v * s, 6) for v in BASE_INERTIA)
    rotor_k = round(BASE_ROTOR_K * s, 10)
    rolling_m = round(BASE_ROLLING_M * s, 12)
    Kp, Kd, Ki = round(BASE_KP * s, 4), round(BASE_KD * s, 4), round(BASE_KI * s, 4)
    Kr, Kw = round(BASE_KR * s, 4), round(BASE_KW * s, 4)
    m_eff = round(mass + (ARM_FIXED_MASS if FLAG_INCLUDE_ARM_IN_M else 0.0), 3)

    print(f"target body mass = {mass} kg (scale s = {s})")
    print(f"  body inertia -> {inertia}")
    print(f"  rotor_constant -> {rotor_k}")
    print(f"  rolling_moment -> {rolling_m}")
    print(f"  controller m   -> {m_eff}  (arm mass {'included' if FLAG_INCLUDE_ARM_IN_M else 'ignored'})")
    print(f"  Kp/Kd/Ki       -> {Kp}/{Kd}/{Ki}")
    print(f"  Kr/Kw          -> {Kr}/{Kw}")
    print()

    usd = REPO / "assets/Robots/Rflyarm/rflyarm.usda"
    hex_ = REPO / "code/logic_vehicles/hexrotor.py"
    ctrl = REPO / "code/examples/utils/nonlinear_controller_arm.py"
    launch = REPO / "code/examples/rflyarm/13_ros2_pose_control_rflyarm.py"

    patch_usd(usd, mass, inertia)
    patch_hexrotor(hex_, rotor_k, rolling_m)
    patch_controller(ctrl, m_eff)
    patch_launch(launch, Kp, Kd, Ki, Kr, Kw)

    # Mirror to Pegasus install
    peg_usd = PEG / "assets/Robots/Rflyarm/rflyarm.usda"
    peg_hex = PEG / "logic/vehicles/hexrotor.py"
    if peg_usd.is_file():
        patch_usd(peg_usd, mass, inertia)
    if peg_hex.is_file():
        patch_hexrotor(peg_hex, rotor_k, rolling_m)


if __name__ == "__main__":
    main()
