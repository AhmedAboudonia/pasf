"""
Batch runner for PSF_quad_linearized_full_state.py.

Generates N_PERTURB perturbed initial conditions around a nominal
[x_init, y_init] = [-4, -4] (z_init held fixed), then for EACH perturbed
initial condition, runs PSF_quad_linearized_full_state.py once per
obstacle radius in OBS_RADIUS_LOCATIONS (obstacle center held fixed at
(-2, -2)).

Total runs = N_PERTURB * len(OBS_RADIUS_LOCATIONS).
"""

import subprocess
import sys
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
SCRIPT = "DRN_4_GSF_rate.py"
N_PERTURB = 10           # number of perturbed initial conditions
PERTURB_STD = 0.1        # std dev of the (x_init, y_init) perturbation
SEED = 42                 # set to None for a different draw every time

X_INIT_NOMINAL = np.array([-3.6, -3.8])   # [x_init, y_init] # -3.5, -4.0
Z_INIT = 3.0                              # held fixed across all runs

OBS_CENTER_X = -2.0         # obs_center_x, held fixed across all runs
OBS_CENTER_Y = -2.0         # obs_center_y, held fixed across all runs
OBS_RADIUS_LOCATIONS = [0.5, 0.75, 1.0]

# ── 1. Generate perturbation samples ───────────────────────────────
rng = np.random.default_rng(SEED)
samples = rng.normal(0, PERTURB_STD, size=(N_PERTURB, 2))   # (10, 2)

run_id = 0

# ── 3. Loop over obstacle radii ──────────────────────────────────
for obs_radius in OBS_RADIUS_LOCATIONS:

    # ── 2. Loop over perturbed initial conditions ──────────────────────
    for xi, yi in X_INIT_NOMINAL + samples:

        args = [
            sys.executable, SCRIPT,
            "-ocx", str(OBS_CENTER_X),
            "-ocy", str(OBS_CENTER_Y),
            "-or", str(obs_radius),
            "-xi", str(xi),
            "-yi", str(yi),
            "-zi", str(Z_INIT),
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