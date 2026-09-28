import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Plot two groups of drone trajectories in solid red / blue.")
parser.add_argument('--dir', type=str, default='.', help='Directory containing the .npz files')
parser.add_argument('--prefix1', type=str, default='Sam_1', help='Filename prefix for group 1 (files <prefix1>_0.npz .. <prefix1>_9.npz)')
parser.add_argument('--prefix2', type=str, default='Sam_3', help='Filename prefix for group 2 (files <prefix2>_0.npz .. <prefix2>_9.npz)')
parser.add_argument('--n', type=int, default=10, help='Number of trajectories per group (X in {0,...,n-1})')
parser.add_argument('-l1', '--label1', type=str, default='Standard', help='Legend label for group 1')
parser.add_argument('-l2', '--label2', type=str, default='Performance Aware', help='Legend label for group 2')
parser.add_argument('-o', '--output', type=str, default='psf_quad_groups.png',
                     help='Output image filename')
args = parser.parse_args()


def load_group(directory, prefix, n):
    """Load n trajectory files named <prefix>_0.npz ... <prefix>_{n-1}.npz."""
    datasets = []
    for i in range(n):
        path = os.path.join(directory, f"{prefix}_{i}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected trajectory file not found: {path}")
        datasets.append(np.load(path))
    return datasets


group1 = load_group(args.dir, args.prefix1, args.n)
group2 = load_group(args.dir, args.prefix2, args.n)

xy1_list = [d['x_hist'][:, :2] for d in group1]
xy2_list = [d['x_hist'][:, :2] for d in group2]


ref = group1[0]
obs_center = ref['obs_center']
obs_radius = float(ref['obs_radius'])
x0_phys = ref['x0_phys']
x_goal = ref['x_goal']


all_others = group1[1:] + group2
for idx, d in enumerate(all_others):
    if not np.allclose(obs_center, d['obs_center']) or not np.isclose(obs_radius, float(d['obs_radius'])):
        print(f"WARNING: file #{idx} has a different obstacle -- plot only shows the reference obstacle.")
    if not np.allclose(x0_phys, d['x0_phys']):
        print(f"WARNING: file #{idx} has a different initial condition -- start dots still reflect each file's own trajectory.")
    if not np.allclose(x_goal, d['x_goal']):
        print(f"WARNING: file #{idx} has a different goal -- plot only shows the reference goal.")


# Plot
fig, ax = plt.subplots(figsize=(8, 5.5))
theta_c = np.linspace(0, 2 * np.pi, 100)

LINE_COLOR_1 = 'red'
LINE_COLOR_2 = 'blue'
LINE_WIDTH = 1.5
START_MARKER_SIZE = 6

_start_label_added = False


def plot_group(xy_list, line_color, label):
    global _start_label_added
    for i, xy in enumerate(xy_list):
        ax.plot(xy[:, 0], xy[:, 1], color=line_color, linewidth=LINE_WIDTH,
                zorder=4, label=label if i == 0 else None)

        start_label = None
        if not _start_label_added:
            start_label = 'Start'
            _start_label_added = True
        ax.plot(xy[0, 0], xy[0, 1], 'go', markersize=START_MARKER_SIZE,
                zorder=6, label=start_label)


# Obstacle
obstacle_x = obs_center[0] + obs_radius * np.cos(theta_c)
obstacle_y = obs_center[1] + obs_radius * np.sin(theta_c)
ax.fill(obstacle_x, obstacle_y, facecolor='lightgreen', edgecolor='green',
        linewidth=1.5, zorder=3, label='Obstacle')

# Trajectories
plot_group(xy1_list, LINE_COLOR_1, args.label1)
plot_group(xy2_list, LINE_COLOR_2, args.label2)

# Goal
ax.plot(x_goal[0], x_goal[1], 'k*', markersize=12, zorder=5, label='Goal')

ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('y', fontsize=16)
ax.tick_params(axis='both', labelsize=13)
ax.legend(loc='upper left', fontsize=13)
ax.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(args.output, dpi=150)
print(f"Saved plot to '{args.output}'")
plt.show()