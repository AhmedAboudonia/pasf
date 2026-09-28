"""
Requirements:
    pip install do-mpc casadi scipy matplotlib numpy torch
"""

import numpy as np
import scipy.linalg
import scipy.signal
from scipy.special import erf as erf_np
import matplotlib.pyplot as plt
import do_mpc
from casadi import *

import time

from DRN_1_Env import NonlinearQuadrotorEnv

IC = np.array([[ 0.03047171, -0.10399841],
                [ 0.07504512,  0.09405647],
                [-0.19510352, -0.13021795],
                [ 0.01278404, -0.03162426],
                [-0.00168012, -0.08530439],
                [ 0.0879398 ,  0.07777919],
                [ 0.00660307,  0.11272412],
                [ 0.04675093, -0.08592925],
                [ 0.03687508, -0.09588826],
                [ 0.08784503, -0.00499259]])

import argparse
parser = argparse.ArgumentParser(description="Run the Quadrotor PSF simulation (12-state linearized closed-loop filter, with rate constraints).")
parser.add_argument('-ocx', '--obs-center-x', type=float, default=-2.0, help='X coordinate of the obstacle center')
parser.add_argument('-ocy', '--obs-center-y', type=float, default=-2.0, help='Y coordinate of the obstacle center')
parser.add_argument('-or', '--obs-radius', type=float, default=0.5, help='Radius of the obstacle')
parser.add_argument('-xi', '--x-init', type=float, default=-3.6+IC[9,0], help='Initial X position')
parser.add_argument('-yi', '--y-init', type=float, default=-3.8+IC[9,1], help='Initial Y position')
parser.add_argument('-zi', '--z-init', type=float, default=3.0, help='Initial Z position')
parser.add_argument('-rid', '--run-id', type=str, default='', help='Optional run identifier')
args = parser.parse_args()
obs_center = np.array([args.obs_center_x, args.obs_center_y])
obs_radius = args.obs_radius
x0_phys = np.array([args.x_init, args.y_init, args.z_init, 0, 0, 0, 0, 0, 0, 0, 0, 0])


env = NonlinearQuadrotorEnv()

dt      = env.dt
h       = env.h
n_sub   = env.n_substeps

N     = 10
T_sim = 600

nx_phys = 12
n_aug   = 3
nx      = nx_phys + n_aug
nu      = 3

# Input bounds
u_min = env.vref_min.copy()
u_max = env.vref_max.copy()

# Rate bounds
a_min = np.array([-0.1, -0.1, -0.1])
a_max = np.array([ 0.1,  0.1,  0.1])


# State bounds
x_min = np.concatenate([env.x_min.copy(), u_min])
x_max = np.concatenate([env.x_max.copy(), u_max])

# goal
x_goal  = env.goal_pos.copy()

SLACK_PEN  = 400


A_cl   = env.A - env.B @ env.K
B_cl_v = env.B @ env.K[:, :3]

Ac_phys = np.zeros((nx_phys, nx_phys))
Ac_phys[0, 3] = 1.0
Ac_phys[1, 4] = 1.0
Ac_phys[2, 5] = 1.0
Ac_phys[3:12, 3:12] = A_cl

Bc_phys = np.zeros((nx_phys, nu))
Bc_phys[3:12, :] = B_cl_v


Ad_phys, Bd_phys, *_ = scipy.signal.cont2discrete(
    (Ac_phys, Bc_phys, np.eye(nx_phys), np.zeros((nx_phys, nu))), dt
)


Ad = np.zeros((nx, nx))
Ad[:nx_phys, :nx_phys] = Ad_phys

Bd = np.zeros((nx, nu))
Bd[:nx_phys, :] = Bd_phys
Bd[nx_phys:, :] = np.eye(n_aug)


class MLP:
    def __init__(self, npz_path, prefix, hidden_activation='tanh', out_activation='linear'):
        data = np.load(npz_path)
        self.weights, self.biases = [], []
        i = 0
        while f'{prefix}_W{i}' in data:
            self.weights.append(data[f'{prefix}_W{i}'])
            self.biases.append(data[f'{prefix}_b{i}'])
            i += 1
        if not self.weights:
            raise ValueError(
                f"No layers found for prefix '{prefix}_' in {npz_path}. "
                f"Expected keys like '{prefix}_W0', '{prefix}_b0', ..."
            )
        self.n_layers = len(self.weights)
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        self.in_dim = self.weights[0].shape[1]
        self.out_dim = self.weights[-1].shape[0]

    @staticmethod
    def _apply_np(name, z):
        if name == 'tanh': return np.tanh(z)
        if name == 'relu': return np.maximum(0.0, z)
        if name == 'gelu': return 0.5 * z * (1.0 + erf_np(z / np.sqrt(2.0)))
        if name == 'linear': return z
        raise ValueError(f"Unknown activation '{name}'")

    @staticmethod
    def _apply_ca(name, z):
        if name == 'tanh': return tanh(z)
        if name == 'relu': return fmax(0.0, z)
        if name == 'gelu': return 0.5 * z * (1.0 + erf(z / sqrt(2.0)))
        if name == 'linear': return z
        raise ValueError(f"Unknown activation '{name}'")

    def forward_numpy(self, x):
        z = np.asarray(x).flatten()
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ z + b
            z = self._apply_np(self.hidden_activation if i < self.n_layers - 1 else self.out_activation, z)
        return z

    def forward_casadi(self, x_sym):
        z = x_sym
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = mtimes(DM(W), z) + DM(b.reshape(-1, 1))
            z = self._apply_ca(self.hidden_activation if i < self.n_layers - 1 else self.out_activation, z)
        return z


NN_WEIGHTS_PATH = 'nn_weights_quad_2_effort_prcp_300.npz'
NN_HIDDEN_ACTIVATION = 'gelu'

policy_net = MLP(NN_WEIGHTS_PATH, 'policy', NN_HIDDEN_ACTIVATION, out_activation='tanh')
critic_net = MLP(NN_WEIGHTS_PATH, 'critic', NN_HIDDEN_ACTIVATION, out_activation='linear')

if policy_net.in_dim != nx_phys or policy_net.out_dim != nu:
    raise ValueError(
        f"policy net in/out dims {policy_net.in_dim}/{policy_net.out_dim} != nx_phys/nu = {nx_phys}/{nu}. "
        f"Did you train with the full-state DRN_2_Train.py (POLICY_STATE_DIM=12)?"
    )
if critic_net.in_dim != nx_phys + nu or critic_net.out_dim != 1:
    raise ValueError(f"critic net in/out dims {critic_net.in_dim}/{critic_net.out_dim} != (nx_phys+nu)/1 = {nx_phys+nu}/1")

u_center = 0.5 * (u_max + u_min)
u_half   = 0.5 * (u_max - u_min)


def policy_numpy(x_phys):
    u_tanh = policy_net.forward_numpy(x_phys)
    return u_center + u_half * u_tanh


def Q_numeric_fn(x_phys, u):
    xu = np.concatenate([np.asarray(x_phys).flatten(), np.asarray(u).flatten()])
    return float(critic_net.forward_numpy(xu)[0])


def policy_casadi(x_sym):
    u_tanh = policy_net.forward_casadi(x_sym)
    return DM(u_center.reshape(-1, 1)) + DM(u_half.reshape(-1, 1)) * u_tanh


def Q_casadi_fn(x_sym, u_sym):
    xu_sym = vertcat(x_sym, u_sym)
    return critic_net.forward_casadi(xu_sym)


model = do_mpc.model.Model('discrete')

phys_state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r']
aug_state_names  = ['vxref_prev', 'vyref_prev', 'vzref_prev']
state_names = phys_state_names + aug_state_names

state_syms = [model.set_variable('_x', name) for name in state_names]
(x_, y_, z_, vx_, vy_, vz_, phi_, theta_, psi_, p_, q_, r_,
 vxref_prev_, vyref_prev_, vzref_prev_) = state_syms

vxref = model.set_variable('_u', 'vxref')
vyref = model.set_variable('_u', 'vyref')
vzref = model.set_variable('_u', 'vzref')

cost_weight = model.set_variable('_tvp', 'cost_weight')
term_switch = model.set_variable('_tvp', 'term_switch')

x_aug_rhs = vertcat(*state_syms)
u_vec_rhs = vertcat(vxref, vyref, vzref)
x_next_rhs = mtimes(DM(Ad), x_aug_rhs) + mtimes(DM(Bd), u_vec_rhs)

for i, name in enumerate(state_names):
    model.set_rhs(name, x_next_rhs[i])

model.setup()


x_full = vertcat(x_, y_, z_, vx_, vy_, vz_, phi_, theta_, psi_, p_, q_, r_)
u_vec = vertcat(vxref, vyref, vzref)


# Safety Filter
mpc = do_mpc.controller.MPC(model)
mpc.set_param(
    n_horizon=N, t_step=dt, n_robust=0, store_full_solution=True,
    nlpsol_opts={
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.sb': 'yes',
        'ipopt.max_wall_time': 0.08,   # seconds -- tune to your real-time budget
    }
)

u_theta_sym = policy_casadi(x_full)
Q_theta_sym = Q_casadi_fn(x_full, u_theta_sym)
Q_val_sym = Q_casadi_fn(x_full, u_vec)
du = u_vec - u_theta_sym

lterm = cost_weight * mtimes(du.T, du)

mterm = SX(0)
mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(vxref=1e-6, vyref=1e-6, vzref=1e-6)

for i, name in enumerate(['vxref', 'vyref', 'vzref']):
    mpc.bounds['lower', '_u', name] = u_min[i]
    mpc.bounds['upper', '_u', name] = u_max[i]

bounded_state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'vxref_prev', 'vyref_prev', 'vzref_prev']
for name in bounded_state_names:
    i = state_names.index(name)
    mpc.bounds['lower', '_x', name] = x_min[i]
    mpc.bounds['upper', '_x', name] = x_max[i]

mpc.set_nl_cons('rate_vxref_ub',  vxref - vxref_prev_, ub=a_max[0])
mpc.set_nl_cons('rate_vxref_lb', -(vxref - vxref_prev_), ub=-a_min[0])
mpc.set_nl_cons('rate_vyref_ub',  vyref - vyref_prev_, ub=a_max[1])
mpc.set_nl_cons('rate_vyref_lb', -(vyref - vyref_prev_), ub=-a_min[1])
mpc.set_nl_cons('rate_vzref_ub',  vzref - vzref_prev_, ub=a_max[2])
mpc.set_nl_cons('rate_vzref_lb', -(vzref - vzref_prev_), ub=-a_min[2])

mpc.set_nl_cons('obstacle',
    (0.2 + obs_radius) ** 2
    - (obs_center[0] - x_) ** 2
    - (obs_center[1] - y_) ** 2,
    ub=0,
    soft_constraint=True,
    penalty_term_cons=SLACK_PEN,
)

tvp_template = mpc.get_tvp_template()


def tvp_fun(t_now):
    for k in range(N + 1):
        tvp_template['_tvp', k, 'cost_weight'] = 1.0 if k == 0 else 0.0
        tvp_template['_tvp', k, 'term_switch'] = 1.0 if k == N - 1 else 0.0
    return tvp_template


mpc.set_tvp_fun(tvp_fun)
mpc.setup()


env.state = x0_phys.astype(np.float32).copy()  # bypass reset()'s random sampling

x0_aug = np.concatenate([x0_phys.copy(), np.zeros(n_aug)])
mpc.x0 = x0_aug.reshape(-1, 1)
mpc.set_initial_guess()


# Simulation loop
x_hist = [x0_phys.copy()]
u_hist = []
u_theta_hist = []
Q_theta_hist = []
Q_safe_hist = []
episode_reward = 0.0
timings = []

x0_aug_now = x0_aug.reshape(-1, 1).copy()

print(f"\nRunning Quadrotor PSF simulation (12-state linearized closed-loop filter "
      f"+ rate constraints, dt={dt}s, rate_max={a_max}, plant inner tick h={h}s, "
      f"n_substeps={n_sub}) ...")
print(f"{'t':>4}  {'x':>6}  {'y':>6}  {'z':>6}  "
      f"{'vxr_ref':>8}  {'vyr_ref':>8}  {'vzr_ref':>8}  "
      f"{'vxr_safe':>8}  {'vyr_safe':>8}  {'vzr_safe':>8}")
print("-" * 100)

for t in range(T_sim):
    x_phys_now = env.state.astype(np.float64) 

    u_theta = policy_numpy(x_phys_now)
    Q_theta_hist.append(Q_numeric_fn(x_phys_now, u_theta))

    start = time.perf_counter()
    u_safe = mpc.make_step(x0_aug_now)
    elapsed = time.perf_counter() - start  # seconds
    timings.append(elapsed)

    return_status = mpc.S.stats()['return_status']
    if return_status == 'Infeasible_Problem_Detected':
        raise RuntimeError(
            f"MPC safety filter genuinely infeasible at step {t} "
            f"(x0={x0_aug_now.flatten()}, return_status={return_status}). "
            f"Refusing to apply an unverified action."
        )

    Q_safe_hist.append(Q_numeric_fn(x_phys_now, u_safe.flatten()))

    # TRUE PLANT
    obs, reward, terminated, truncated, info = env.step(u_safe.flatten())
    episode_reward += reward

    x0_aug_now = np.concatenate([
        env.state.astype(np.float64),
        u_safe.flatten(),
    ]).reshape(-1, 1)

    x_hist.append(env.state.astype(np.float64).copy())
    u_hist.append(u_safe.flatten())
    u_theta_hist.append(u_theta.copy())

    if t % 1 == 0:
        p = env.state[0:3]
        print(f"{t:>4}  {p[0]:>6.2f}  {p[1]:>6.2f}  {p[2]:>6.2f}  "
              f"{u_theta[0]:>8.3f}  {u_theta[1]:>8.3f}  {u_theta[2]:>8.3f}  "
              f"{float(u_safe[0,0]):>8.3f}  {float(u_safe[1,0]):>8.3f}  {float(u_safe[2,0]):>8.3f}")

    if terminated or truncated:
        print(f"Episode ended at t={t}: terminated={terminated}, truncated={truncated}")
        break

print("\nSimulation complete.")

import json

results_path = 'psf_quad_results.jsonl'
result_record = {
    'run_id':         args.run_id,
    'obs_center_x':   args.obs_center_x,
    'obs_center_y':   args.obs_center_y,
    'obs_radius':     args.obs_radius,
    'x_init':         args.x_init,
    'y_init':         args.y_init,
    'z_init':         args.z_init,
    'episode_reward': float(episode_reward),
    'timings_mean_ms': float(np.mean(timings)*1000),
    'timings_std_ms': float(np.std(timings)*1000),
    'timings_max_ms':  float(np.max(timings)*1000),
    'timings_min_ms':  float(np.min(timings)*1000),
    'timings_q1_ms':  float(np.quantile(timings, 0.01)*1000),
    'timings_q5_ms':  float(np.quantile(timings, 0.05)*1000),
    'timings_med_ms':  float(np.quantile(timings, 0.5)*1000),
    'timings_q95_ms':  float(np.quantile(timings, 0.95)*1000),
    'timings_q99_ms':  float(np.quantile(timings, 0.99)*1000),
    'timing':         [float(t*1000) for t in timings],
    'n_steps':        len(u_hist),
}
with open(results_path, 'a') as f:
    f.write(json.dumps(result_record) + '\n')

print(f"EPISODE_REWARD={episode_reward:.6f}")
print(f"Mean planning time: {np.mean(timings)*1000:.3f} ms")
print(f"Max planning time:  {np.max(timings)*1000:.3f} ms")

# Plot results
Q_theta_hist = np.array(Q_theta_hist)
Q_safe_hist = np.array(Q_safe_hist)
x_hist = np.vstack(x_hist)  # full 12-dim (true plant)
u_hist = np.vstack(u_hist)
u_theta_hist = np.vstack(u_theta_hist)
n_steps = len(u_hist)
t_axis = np.arange(n_steps) * dt
t_axis_x = np.arange(n_steps + 1) * dt

traj_path = f'psf_quad_trajectory_PASF_{args.run_id}.npz' if args.run_id else 'psf_quad_trajectory.npz'
np.savez(
    traj_path,
    t_axis=t_axis,
    t_axis_x=t_axis_x,
    x_hist=x_hist,
    u_hist=u_hist,
    u_theta_hist=u_theta_hist,
    Q_theta_hist=Q_theta_hist,
    Q_safe_hist=Q_safe_hist,
    obs_center=obs_center,
    obs_radius=obs_radius,
    x0_phys=x0_phys,
    x_goal=x_goal,
    episode_reward=episode_reward,
)
print(f"Trajectories saved to '{traj_path}'")

# # -- Critic value: proposed vs. filtered --
# fig, ax = plt.subplots(figsize=(8, 4))
# ax.plot(t_axis, Q_theta_hist, label=r'$Q(s, u_\theta)$ (proposed)', linewidth=2)
# ax.plot(t_axis, Q_safe_hist, label=r'$Q(s, u_{safe})$ (applied)', linewidth=2, linestyle='--')
# ax.set_xlabel('time [s]'); ax.set_ylabel('Q value')
# ax.set_title("Quadrotor NN Q-function (12-state linearized closed-loop filter + rate constraints): proposed vs. filtered action")
# ax.legend(); ax.grid(True, alpha=0.3)
# plt.tight_layout()

# # -- Position, velocity, and inputs over time --
# fig2, axs = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
# pos_labels = ['x', 'y', 'z']
# vel_labels = ['vx', 'vy', 'vz']
# for i in range(3):
#     axs[0, i].plot(t_axis_x, x_hist[:, i]); axs[0, i].axhline(x_goal[i], color='gold', linestyle='--', linewidth=1)
#     axs[0, i].set_ylabel(pos_labels[i]); axs[0, i].grid(alpha=0.3)
#     axs[1, i].plot(t_axis_x, x_hist[:, 3 + i])
#     axs[1, i].set_ylabel(vel_labels[i]); axs[1, i].grid(alpha=0.3)
#     axs[2, i].step(t_axis, u_hist[:, i], where='post', label='safe')
#     axs[2, i].step(t_axis, u_theta_hist[:, i], where='post', label='ref', linestyle='--', alpha=0.7)
#     axs[2, i].set_ylabel(f'v{pos_labels[i]}_ref'); axs[2, i].set_xlabel('time [s]')
#     axs[2, i].legend(); axs[2, i].grid(alpha=0.3)
# fig2.suptitle('Quadrotor States and Inputs (true plant, 12-state linearized closed-loop filter + rate constraints)')
# plt.tight_layout()

# # -- Top-view (x-y) trajectory with obstacle projection --
# fig3, ax3 = plt.subplots(figsize=(6, 6))
# theta_c = np.linspace(0, 2 * np.pi, 100)
# ax3.plot(obs_center[0] + obs_radius * np.cos(theta_c),
#          obs_center[1] + obs_radius * np.sin(theta_c), 'r-', label='obstacle (x-y projection)')
# ax3.plot(x_hist[:, 0], x_hist[:, 1], 'b-', linewidth=2, label='trajectory')
# ax3.plot(x0_phys[0], x0_phys[1], 'go', markersize=8, label='start')
# ax3.plot(x_goal[0], x_goal[1], 'k*', markersize=12, label='goal')
# ax3.set_xlabel('x'); ax3.set_ylabel('y')
# ax3.set_title('Top-view (x-y) trajectory')
# ax3.legend(); ax3.grid(alpha=0.3); ax3.set_aspect('equal')
# plt.tight_layout()

# plt.savefig('psf_quad_result.png', dpi=100)
# plt.show()
print("Done!")