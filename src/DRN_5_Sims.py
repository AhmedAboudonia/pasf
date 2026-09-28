import subprocess
import sys
import numpy as np

SCRIPT = "DRN_4_PSF.py" # or DRN_4_SSF.py
N_PERTURB = 10
PERTURB_STD = 0.1
SEED = 42

X_INIT_NOMINAL = np.array([-3.6, -3.8])
Z_INIT = 3.0

OBS_CENTER_X = -2.0
OBS_CENTER_Y = -2.0
OBS_RADIUS = [0.5, 0.75, 1.0]

rng = np.random.default_rng(SEED)
samples = rng.normal(0, PERTURB_STD, size=(N_PERTURB, 2))   # (10, 2)

run_id = 0

for obs_radius in OBS_RADIUS:

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