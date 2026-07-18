#!/usr/bin/env python
"""
| File: plot_force_torque.py
| Description: Reads the statistics .npz saved by NonlinearController and plots the
| control effort over time: total thrust u_1 [N] and body torques tau [Nm].
| Usage:  isaac_run examples/utils/plot_force_torque.py [path_to_npz]
| Default npz: examples/results/single_statistics.npz
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe: write to file instead of opening a window
import matplotlib.pyplot as plt

# Resolve the npz path (arg 1 or default results file)
here = os.path.dirname(os.path.realpath(__file__))
default_npz = os.path.join(here, "..", "results", "single_statistics.npz")
npz_path = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(default_npz)

if not os.path.isfile(npz_path):
    print(f"[ERROR] npz not found: {npz_path}")
    print("Run the simulation first (it saves on stop), then run this script.")
    sys.exit(1)

data = np.load(npz_path)
print("Keys in npz:", list(data.keys()))

if "thrust" not in data or "torque" not in data:
    print("[ERROR] This npz has no 'thrust'/'torque' arrays.")
    print("It was produced before the controller was patched to record them.")
    print("Re-run the simulation once with the updated nonlinear_controller.py.")
    sys.exit(1)

t = data["time"]
thrust = data["thrust"]          # shape (N,)
torque = data["torque"]          # shape (N, 3)

# Guard against length mismatch (time vs effort can differ by one sample)
n = min(len(t), len(thrust), len(torque))
t, thrust, torque = t[:n], thrust[:n], torque[:n]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# --- Total thrust ---
ax1.plot(t, thrust, color="tab:blue", linewidth=1.5)
ax1.set_ylabel("Total thrust $u_1$ [N]")
ax1.set_title("Control effort over time")
ax1.grid(True, alpha=0.3)

# --- Body torques ---
ax2.plot(t, torque[:, 0], label=r"$\tau_x$ (roll)", color="tab:red", linewidth=1.2)
ax2.plot(t, torque[:, 1], label=r"$\tau_y$ (pitch)", color="tab:green", linewidth=1.2)
ax2.plot(t, torque[:, 2], label=r"$\tau_z$ (yaw)", color="tab:purple", linewidth=1.2)
ax2.set_ylabel("Body torque $\\tau$ [Nm]")
ax2.set_xlabel("Time [s]")
ax2.legend(loc="best")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
out_path = os.path.splitext(npz_path)[0] + "_force_torque.png"
fig.savefig(out_path, dpi=130)
print(f"[OK] Saved plot to: {out_path}")
print(f"     samples={n}  t=[{t[0]:.2f}, {t[-1]:.2f}]s  "
      f"thrust=[{thrust.min():.2f},{thrust.max():.2f}]N")
