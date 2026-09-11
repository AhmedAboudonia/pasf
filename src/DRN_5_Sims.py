"""
Batch runner for DRN_4_SF_rate.py -- SWEEP OVER 4 NOMINAL INITIAL
CONDITIONS, EACH WITH N_PERTURB PERTURBED INITIAL CONDITIONS AROUND IT.

For EACH nominal initial condition (x_nom, y_nom) in

    (-4, -4), (-4, 4), (4, -4), (4, 4)

generates N_PERTURB perturbed [x_init, y_init] samples around it (z_init
held fixed), then runs DRN_4_SF_rate.py once per perturbed sample, with:
    - obs_radius FIXED at 0.5
    - obs_center = 0.5 * (x_nom, y_nom)   (tied to the NOMINAL IC of that
                                            group, not the perturbed sample
                                            -- so all perturbations sharing
                                            a nominal IC share one obstacle)

Total runs = len(NOMINAL_ICS) * N_PERTURB = 4 * 10 = 40.
"""

import subprocess
import sys
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
SCRIPT = "DRN_4_SF_rate.py"
N_PERTURB = 10            # number of perturbed initial conditions per nominal IC
PERTURB_STD = 0.1         # std dev of the (x_init, y_init) perturbation
SEED = 42                  # set to None for a different draw every time

Z_INIT = 3.0               # held fixed across all runs
OBS_RADIUS = 0.5           # FIXED across all runs

NOMINAL_ICS = [(-4.0, -4.0), (-4.0, 4.0), (4.0, -4.0), (4.0, 4.0)]

rng = np.random.default_rng(SEED)

run_id = 0

# ── Loop over the 4 nominal initial conditions ─────────────────────
for x_nom, y_nom in NOMINAL_ICS:

    x_nom_arr = np.array([x_nom, y_nom])
    obs_center_x = 0.5 * x_nom
    obs_center_y = 0.5 * y_nom

    # 10 perturbed [x_init, y_init] samples around this nominal IC
    samples = rng.normal(0, PERTURB_STD, size=(N_PERTURB, 2))

    # ── Loop over the N_PERTURB perturbed initial conditions ────────
    for xi, yi in x_nom_arr + samples:

        args = [
            sys.executable, SCRIPT,
            "-ocx", str(obs_center_x),
            "-ocy", str(obs_center_y),
            "-or", str(OBS_RADIUS),
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