"""
Batch runner for DRN_4_GSF_rate.py.

Generates N_PERTURB perturbed initial conditions around a nominal
[x_init, y_init] = [-3.6, -3.8] (z_init held fixed), then for EACH
perturbed initial condition, runs DRN_4_GSF_rate.py once per weight_U
value in WEIGHT_U_LOCATIONS (obstacle center and radius held fixed).

Total runs = N_PERTURB * len(WEIGHT_U_LOCATIONS).
"""

import subprocess
import sys
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
SCRIPT = "DRN_4_GSF_rate.py"
N_PERTURB = 10           # number of perturbed initial conditions
PERTURB_STD = 0.1        # std dev of the (x_init, y_init) perturbation
SEED = 0                 # set to None for a different draw every time

X_INIT_NOMINAL = np.array([-3.6, -3.8])   # [x_init, y_init] # -3.5, -4.0
Z_INIT = 3.0                              # held fixed across all runs

OBS_CENTER_X = -2.0         # obs_center_x, held fixed across all runs
OBS_CENTER_Y = -2.0         # obs_center_y, held fixed across all runs
OBS_RADIUS = 0.5            # obs_radius, held fixed across all runs

WEIGHT_U_LOCATIONS = [10, 100, 1000]   # swept instead of obstacle radius

# ── 1. Generate perturbation samples ───────────────────────────────
rng = np.random.default_rng(SEED)
samples = rng.normal(0, PERTURB_STD, size=(N_PERTURB, 2))   # (10, 2)

run_id = 0

# ── 3. Loop over weight_U values ──────────────────────────────────
for weight_u in WEIGHT_U_LOCATIONS:

    # ── 2. Loop over perturbed initial conditions ──────────────────
    for xi, yi in X_INIT_NOMINAL + samples:

        args = [
            sys.executable, SCRIPT,
            "-ocx", str(OBS_CENTER_X),
            "-ocy", str(OBS_CENTER_Y),
            "-or", str(OBS_RADIUS),
            "-xi", str(xi),
            "-yi", str(yi),
            "-zi", str(Z_INIT),
            "-wu", str(weight_u),
            "-rid", str(run_id),
        ]

        print()
        print("=" * 44)
        print(f"Run {run_id}: {' '.join(args[2:])}")
        print("=" * 44)
        subprocess.run(args, check=True)

        run_id += 1

print()
print(f"All {run_id} runs complete.")