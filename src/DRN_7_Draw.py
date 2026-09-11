"""
Plot two closed-loop drone top-view (x, y) trajectories on the same plot,
alongside a shared obstacle, start, and goal marker.

Expects two .npz files produced by the quadrotor PSF simulation script's
trajectory-saving step, each containing (at least):
    x_hist      : (T+1, 12) true-plant physical states
                  [x,y,z,vx,vy,vz,phi,theta,psi,p,q,r]
    obs_center  : (2,)      obstacle center [ox, oy]  (x-y projection)
    obs_radius  : scalar    obstacle radius
    x0_phys     : (12,)     initial condition
    x_goal      : (3,)      goal position [gx, gy, gz]

Usage:
    python plot_two_drone_trajectories.py traj_A.npz traj_B.npz
    python plot_two_drone_trajectories.py traj_A.npz traj_B.npz -l1 "run 0" -l2 "run 1"
    python plot_two_drone_trajectories.py traj_A.npz traj_B.npz -o combined.png
"""

# python DRN_7_Draw.py Ex4_pasf.npz Ex4_std.npz -l1 "Proposed" -l2 "Standard"

import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Plot two drone closed-loop top-view trajectories together.")
parser.add_argument('traj1', type=str, help='Path to first trajectory .npz file')
parser.add_argument('traj2', type=str, help='Path to second trajectory .npz file')
parser.add_argument('-l1', '--label1', type=str, default='trajectory 1', help='Legend label for trajectory 1')
parser.add_argument('-l2', '--label2', type=str, default='trajectory 2', help='Legend label for trajectory 2')
parser.add_argument('-o', '--output', type=str, default='psf_quad_two_trajectories.png',
                     help='Output image filename')
args = parser.parse_args()

# ── Load both trajectory files ──────────────────────────────────────
data1 = np.load(args.traj1)
data2 = np.load(args.traj2)

x_hist_1 = data1['x_hist']       # (T1+1, 12)
x_hist_2 = data2['x_hist']       # (T2+1, 12)

# Shared obstacle, start, and goal -- taken from traj1 (warn if traj2 disagrees)
obs_center = data1['obs_center']
obs_radius = float(data1['obs_radius'])
x0_phys = data1['x0_phys']
x_goal = data1['x_goal']

if not np.allclose(obs_center, data2['obs_center']) or not np.isclose(obs_radius, float(data2['obs_radius'])):
    print("WARNING: traj1 and traj2 have different obstacles -- plot only shows traj1's obstacle.")
if not np.allclose(x0_phys, data2['x0_phys']):
    print("WARNING: traj1 and traj2 have different initial conditions -- plot only shows traj1's start.")
if not np.allclose(x_goal, data2['x_goal']):
    print("WARNING: traj1 and traj2 have different goals -- plot only shows traj1's goal.")


# ── Plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
theta_c = np.linspace(0, 2 * np.pi, 100)

# Obstacle (shared, x-y projection)
ax.plot(obs_center[0] + obs_radius * np.cos(theta_c),
        obs_center[1] + obs_radius * np.sin(theta_c), 'r-', label='obstacle')

# Trajectories (top-view, x-y)
ax.plot(x_hist_1[:, 0], x_hist_1[:, 1], 'b-', linewidth=2, label=args.label1)
ax.plot(x_hist_2[:, 0], x_hist_2[:, 1], color='darkorange', linewidth=2, label=args.label2)

# Start marker (shared)
ax.plot(x0_phys[0], x0_phys[1], 'go', markersize=8, label='start')

# Goal marker (shared)
ax.plot(x_goal[0], x_goal[1], 'k*', markersize=12, label='goal')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Top-view (x-y) trajectories')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()

plt.savefig(args.output, dpi=150)
print(f"Saved plot to '{args.output}'")
plt.show()