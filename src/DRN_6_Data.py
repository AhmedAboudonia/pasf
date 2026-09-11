import json
import numpy as np
import matplotlib.pyplot as plt

GROUP_SIZE = 10

# Actual obstacle-radius values swept in your batch runner, one per group,
# in the same order the runs were generated (must match the number of groups).
# See batch_run_quad_psf.py's OBS_RADIUS_LOCATIONS.
# OBS_LOCATIONS = ["NN", "NP", "PN", "PP"]
OBS_LOCATIONS = [0.5, 0.75, 1.0]

# ── Config: one entry per METHOD being compared ─────────────────────
# Each method's results should live in its own .jsonl file (e.g. produced by
# running your batch sweep once per method / POLICY_SOURCE / controller variant).
METHODS = [
    {"path": "psf_quad_results_GenPASF_org.jsonl", "label": "Performance-aware"},
    {"path": "psf_quad_results_baseline_org.jsonl", "label": "Standard"},
    #{"path": "psf_quad_results_2_FT_rate.jsonl", "label": "Approximate"},
]

# ── Font size config ─────────────────────────────────────────────────
FONT_LEGEND = 12
FONT_AXIS_LABEL = 13
FONT_TICK = 11


def load_rewards(path):
    with open(path) as f:
        results = [json.loads(line) for line in f]
    return np.array([r['episode_reward'] for r in results])

def grouped_mean_std(rewards, group_size):
    n_runs = len(rewards)
    n_full_groups = n_runs // group_size
    remainder = n_runs % group_size
    if remainder:
        print(f"Warning: {n_runs} runs is not a multiple of {group_size}; "
              f"last {remainder} run(s) will form a smaller final group.")
    n_groups = n_full_groups + (1 if remainder else 0)

    means = np.zeros(n_groups)
    stds = np.zeros(n_groups)
    labels = []
    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, n_runs)
        chunk = rewards[start:end]
        means[g] = chunk.mean()
        stds[g] = chunk.std()
        labels.append(f"{start}-{end-1}")
    return means, stds, labels


# ── Load and group each file ────────────────────────────────────────
all_means, all_stds, all_labels = [], [], []
for entry in METHODS:
    rewards = load_rewards(entry["path"])
    means, stds, labels = grouped_mean_std(rewards, GROUP_SIZE)
    all_means.append(means)
    all_stds.append(stds)
    all_labels.append(labels)
    print(f"\n{entry['label']} ({entry['path']}):")
    for lbl, m, s in zip(labels, means, stds):
        print(f"  Runs {lbl}: mean={m:.3f}  std={s:.3f}")

# Use the group count from whichever file has the most groups.
n_groups = max(len(m) for m in all_means)
x = np.arange(n_groups)

if len(OBS_LOCATIONS) == n_groups:
    x_labels = OBS_LOCATIONS
else:
    print(f"Warning: OBS_LOCATIONS has {len(OBS_LOCATIONS)} entries but there are "
          f"{n_groups} groups; falling back to run-range labels.")
    x_labels = max(all_labels, key=len)   # labels from the longest file

n_files = len(METHODS)
bar_width = 0.8 / n_files

fig, ax = plt.subplots(figsize=(1.5 * n_groups + 2, 4))

for i, entry in enumerate(METHODS):
    means = all_means[i]
    stds = all_stds[i]
    # Pad shorter files with NaN so bars simply don't appear for missing groups
    if len(means) < n_groups:
        pad = n_groups - len(means)
        means = np.concatenate([means, np.full(pad, np.nan)])
        stds = np.concatenate([stds, np.full(pad, np.nan)])

    offset = (i - (n_files - 1) / 2) * bar_width
    ax.bar(x + offset, means, width=bar_width, yerr=stds, capsize=4,
           label=entry["label"], alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=FONT_TICK)
ax.tick_params(axis='y', labelsize=FONT_TICK)
ax.set_xlabel('Obstacle Radius', fontsize=FONT_AXIS_LABEL)
ax.set_ylabel('Episode Return', fontsize=FONT_AXIS_LABEL)
# ax.set_title(f'Mean ± std per group of {GROUP_SIZE} consecutive runs, by method')
ax.legend(fontsize=FONT_LEGEND)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('psf_quad_mean_std_grouped_compare.png', dpi=100)
plt.show()
print("\nDone")