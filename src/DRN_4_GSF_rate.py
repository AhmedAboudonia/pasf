"""
Predictive Safety Filter (PSF) for the Nonlinear Quadrotor -- 12-STATE
LINEARIZED CLOSED-LOOP MODEL, WITH RATE CONSTRAINTS + AUGMENTED STATES
=============================================================================
System: full nonlinear 6-DOF rigid-body quadrotor, cascaded with an inner
LQR controller (see DRN_1_Env.py):

    RL/filter velocity ref [vx_ref,vy_ref,vz_ref] --> LQR --> [T,tau] --> plant

THE FILTER'S INTERNAL MODEL IS THE FULL 12-STATE SYSTEM, LINEARIZED ABOUT
HOVER, WITH THE INNER LQR LOOP CLOSED ANALYTICALLY -- NOT an identified
reduced-order model:

    Physical filter state:  [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
                             (nx_phys = 12)
    Input:                  v_ref = [vx_ref, vy_ref, vz_ref]   (nu = 3)

The env already builds and stores the ingredients we need for this at
construction time (see NonlinearQuadrotorEnv.__init__ / _linearize):

    z = [vx,vy,vz,phi,theta,psi,p,q,r]   (9-state reduced system)
    A = df/dz, B = df/du   (Jacobians of the reduced nonlinear dynamics
                             about hover, z=0, u=u_hover)
    K  s.t.  u = u_hover - K @ (z - z_ref)   (continuous LQR gain)

Substituting the control law into the linearized reduced dynamics and using
that z_ref only ever has nonzero entries in its first 3 components
(z_ref = [vx_ref,vy_ref,vz_ref,0,0,0,0,0,0], since attitude/rate references
are always zero) gives the CLOSED-LOOP linear system:

    z_dot = A @ z + B @ (u - u_hover)
          = A @ z + B @ (-K @ (z - z_ref))
          = A @ z - B @ K @ z
            + B @ K @ z_ref
          = (A - B@K) @ z + (B @ K[:, :3]) @ v_ref

Augmenting with the (trivial) position integrators pos_dot = vel gives the
12-state physical linear model used here:

    x_dot   = vx                     (and similarly y_dot, z_dot)
    z_dot   = (A - B@K) @ z + (B @ K[:, :3]) @ v_ref

This is exactly the "linearized closed loop" -- no system identification
of any kind is used. It's also a strictly richer model than the earlier
6-state ROM filter: attitude and body rates are genuine filter states here,
not collapsed into a lumped velocity lag.

This 12-dim physical state also matches what the (full-state) DDPG
policy/critic were trained on -- see DRN_2_Train.py's POLICY_STATE_DIM=12 --
so the policy/critic networks ALWAYS see exactly this 12-dim physical
state (never the augmented state below), matching the loaded network's
input dimension directly.

RATE CONSTRAINTS + AUGMENTED STATES (new in this version):
    We now enforce  a_min <= u_k - u_{k-1} <= a_max  on the three inputs
    [vxref, vyref, vzref]. Since do_mpc's stage constraints only see the
    CURRENT state/input pair (_x, _u) at each node, "u_{k-1}" has to be
    made available as part of the state itself. We do this the standard
    way: augment the filter state with three extra "previous input" states

        vxref_prev, vyref_prev, vzref_prev

    whose trivial dynamics are simply

        vxref_prev_{k+1} = vxref_k   (and similarly for vyref_prev, vzref_prev)

    i.e. "what I apply now becomes 'previous' at the next node." This is
    affine in u and totally decoupled from the physical 12-state block, so
    it drops cleanly into the SAME discrete-time linear state-space form
    used for the physical states -- we just block-augment Ad/Bd:

        Ad_aug = [[Ad_phys,   0   ],      Bd_aug = [[Bd_phys],
                  [   0   ,   0   ]]                 [  I_3  ]]

    giving a (nx_phys+3) x (nx_phys+3) = 15-state augmented filter model.
    The rate constraint itself is then a plain nonlinear (in do_mpc's
    sense -- actually affine) stage constraint:

        a_min <= vxref - vxref_prev <= a_max   (and similarly y, z)

    IMPORTANT: this augmentation is PURELY an MPC/filter-internal
    bookkeeping trick. The policy/critic networks are untouched and still
    only ever see the 12-dim PHYSICAL state (x_full below) -- they were
    trained with POLICY_STATE_DIM=12 and have no notion of "previous
    input," so we never feed them the augmented 15-dim vector.

TWO DIFFERENT TIMESCALES, TWO DIFFERENT MODELS, same as before:

  - The SAFETY FILTER (this MPC) plans using the 15-state augmented
    linearized closed-loop model above (12 physical + 3 previous-input),
    discretized at the OUTER rate dt = env.dt (the rate the filter itself
    runs at -- same rate an RL policy would be queried at).

  - The TRUE PLANT -- what actually happens when a filtered action is
    applied -- is STILL the full nonlinear NonlinearQuadrotorEnv, unchanged,
    which internally re-runs its LQR inner loop and RK4 physics integration
    at h = env.h (n_substeps ticks per outer dt). Nothing about the
    simulator changes; only the filter's internal prediction model does.
    The "previous input" part of the augmented state is tracked outside
    the env, in the simulation loop, from the actually-applied u_safe.

"Reference policy" u_theta(x) and "critic" Q(x,u) come from a pretrained
DDPG actor/critic (see DRN_2_Train.py, full-state variant), loaded from an
.npz weight file via the same dimension-agnostic MLP class as before --
nx_phys=12, nu=3, matching the trained network's actual input size.

Cost:    min  (Q(x0,u0) - Q(x0,u_theta(x0)))^2     (only at k=0)

Constraints:
    - Input bounds:        u_min <= u <= u_max            (velocity ref bounds)
    - Rate bounds:          a_min <= u_k - u_{k-1} <= a_max  (NEW: via the
                             3 augmented "previous input" states)
    - Obstacle avoidance:   keep (x,y) outside a circle of radius r around
                             obs_center (top-view / x-y projection, as before)

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

# IC = np.array([[ 0.03047171, -0.10399841],
#                 [ 0.07504512,  0.09405647],
#                 [-0.19510352, -0.13021795],
#                 [ 0.01278404, -0.03162426],
#                 [-0.00168012, -0.08530439],
#                 [ 0.0879398 ,  0.07777919],
#                 [ 0.00660307,  0.11272412],
#                 [ 0.04675093, -0.08592925],
#                 [ 0.03687508, -0.09588826],
#                 [ 0.08784503, -0.00499259]])


IC = np.array([[ 0.01257302, -0.01321049],
                [ 0.06404227,  0.01049001],
                [-0.05356694,  0.03615951],
                [ 0.1304    ,  0.0947081 ],
                [-0.07037352, -0.12654215],
                [-0.06232745,  0.0041326 ],
                [-0.23250308, -0.02187917],
                [-0.12459109, -0.07322674],
                [-0.0544259 , -0.03163002],
                [ 0.04116305,  0.10425134]])

import argparse
parser = argparse.ArgumentParser(description="Run the Quadrotor PSF simulation (12-state linearized closed-loop filter, with rate constraints).")
parser.add_argument('-ocx', '--obs-center-x', type=float, default=-2.0, help='X coordinate of the obstacle center')
parser.add_argument('-ocy', '--obs-center-y', type=float, default=-2.0, help='Y coordinate of the obstacle center')
parser.add_argument('-or', '--obs-radius', type=float, default=0.5, help='Radius of the obstacle')
parser.add_argument('-xi', '--x-init', type=float, default=-3.6+IC[2,0], help='Initial X position')
parser.add_argument('-yi', '--y-init', type=float, default=-3.8+IC[2,1], help='Initial Y position')
parser.add_argument('-zi', '--z-init', type=float, default=3.0, help='Initial Z position')
parser.add_argument('-rid', '--run-id', type=str, default='', help='Optional run identifier')
parser.add_argument('-wu', '--weight-u', type=float, default=1000.0, help='Weight on the ||u - u_theta||^2 term in the MPC cost (default: 10)')
args = parser.parse_args()
obs_center = np.array([args.obs_center_x, args.obs_center_y])
obs_radius = args.obs_radius
x0_phys = np.array([args.x_init, args.y_init, args.z_init, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# running single file from command line, e.g.:
# python .\PSF_quad_linearized_full_state_rate.py -ocx -1 -ocy -1 -or 0.25 -xi 1 -yi 1 -rmax 0.75


# ─────────────────────────────────────────────
# 1. True plant setup + filter's linearized closed-loop model
#    (no system identification -- A, B, K come straight from the env's
#    own analytic linearization about hover)
# ─────────────────────────────────────────────
env = NonlinearQuadrotorEnv()

dt      = env.dt         # outer step / filter rate
h       = env.h          # inner LQR/RK4 tick (handled INSIDE env.step)
n_sub   = env.n_substeps

N     = 10       # filter prediction horizon (in outer dt steps)
T_sim = 600     # simulation steps

nx_phys = 12    # PHYSICAL filter/policy/critic states: x,y,z,vx,vy,vz,phi,theta,psi,p,q,r
n_aug   = 3     # augmented "previous input" states: vxref_prev, vyref_prev, vzref_prev
nx      = nx_phys + n_aug   # total augmented filter/MPC state dim = 15
nu      = 3     # inputs: vx_ref, vy_ref, vz_ref

# Input bounds: same velocity-reference bounds the env/DDPG action space uses
u_min = env.vref_min.copy()
u_max = env.vref_max.copy()

# Rate bounds: symmetric |Δv_ref| <= rate_max per outer dt step, per axis
a_min = np.array([-0.1, -0.1, -0.1])
a_max = np.array([ 0.1,  0.1,  0.1])


# State bounds: env's own physical 12-state box, plus a box for the 3
# "previous input" augmented states -- these just track a past control
# value, so they're bounded by the same u_min/u_max as the inputs.
x_min = np.concatenate([env.x_min.copy(), u_min])
x_max = np.concatenate([env.x_max.copy(), u_max])

# Initial condition / goal
x_goal  = env.goal_pos.copy()   # (0, 0, goal_z) -- NOT the origin, see env docstring

USE_Q_COST = True
USE_SQP    = False
SLACK_PEN  = 400+4*args.weight_u
weight_Q = 1
weight_U = args.weight_u # 10, 100, 1000


# ─────────────────────────────────────────────
# 2. Filter's internal prediction model: 15-state augmented linearized
#    closed loop (12 physical + 3 "previous input" for rate constraints)
#        x_dot = vx,  y_dot = vy,  z_dot = vz            (position integrators)
#        z_dot = (A - B@K) @ z + (B @ K[:, :3]) @ v_ref   (closed-loop reduced dyn.)
#        [vxref_prev,vyref_prev,vzref_prev]_{k+1} = v_ref_k   (rate-constraint bookkeeping)
#    where z = [vx,vy,vz,phi,theta,psi,p,q,r], and A, B, K are exactly the
#    env's own hover-linearization / LQR gain (env.A, env.B, env.K) -- no
#    identification of any kind. Discretized at the OUTER rate dt.
# ─────────────────────────────────────────────
A_cl   = env.A - env.B @ env.K          # (9,9) closed-loop reduced dynamics
B_cl_v = env.B @ env.K[:, :3]            # (9,3) -- only the vref columns of
                                          # K@z_ref survive, since z_ref's
                                          # attitude/rate entries are always 0

Ac_phys = np.zeros((nx_phys, nx_phys))
Ac_phys[0, 3] = 1.0   # x_dot = vx
Ac_phys[1, 4] = 1.0   # y_dot = vy
Ac_phys[2, 5] = 1.0   # z_dot = vz
Ac_phys[3:12, 3:12] = A_cl

Bc_phys = np.zeros((nx_phys, nu))
Bc_phys[3:12, :] = B_cl_v

# Discretize the PHYSICAL 12-state block first (this block never depends on
# the augmented "previous input" states, so its discretization is
# unaffected by the augmentation).
Ad_phys, Bd_phys, *_ = scipy.signal.cont2discrete(
    (Ac_phys, Bc_phys, np.eye(nx_phys), np.zeros((nx_phys, nu))), dt
)

# Block-augment with the 3 trivial "previous input" states:
#   vxref_prev_{k+1} = vxref_k   (and similarly y, z)
# This is exactly-affine (already in discrete-time form, no ODE to
# discretize), so it drops straight into the augmented Ad/Bd as an
# identity block, decoupled from everything else.
Ad = np.zeros((nx, nx))
Ad[:nx_phys, :nx_phys] = Ad_phys
# bottom-right (n_aug, n_aug) block stays 0: prev-input states don't
# depend on their own previous value, only on the CURRENT input (via Bd)

Bd = np.zeros((nx, nu))
Bd[:nx_phys, :] = Bd_phys
Bd[nx_phys:, :] = np.eye(n_aug)   # vref_prev_{k+1} = vref_k


# ─────────────────────────────────────────────
# 2b. NN policy / critic (loaded from the trained DDPG weights)
#     Unchanged MLP class. Policy/critic ALWAYS see the 12-dim PHYSICAL
#     state only (nx_phys=12), matching DRN_2_Train.py's full-state
#     POLICY_STATE_DIM=12 -- the augmented rate-constraint states are
#     never passed to these networks.
# ─────────────────────────────────────────────
class MLP:
    """
    Small feedforward MLP giving BOTH a NumPy forward pass (numeric, for
    logging/plots) and a CasADi symbolic forward pass (for the optimizer)
    from the SAME weights, loaded from an .npz file.

    Expected .npz keys (PyTorch nn.Linear convention: W shape (out, in)):
        <prefix>_W0, <prefix>_b0, <prefix>_W1, <prefix>_b1, ...
    """
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


NN_WEIGHTS_PATH = 'nn_weights_quad_2_effort_prcp_300.npz' #  nn_weights_quad_2_effort_prcp_300.npz
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

u_center = 0.5 * (u_max + u_min)   # = 0, since vref bounds are symmetric
u_half   = 0.5 * (u_max - u_min)   # = u_max


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


# ─────────────────────────────────────────────
# 3. do_mpc model -- the FILTER's own 15-state augmented linearized
#    closed-loop model (12 physical + 3 "previous input" for rate
#    constraints)
# ─────────────────────────────────────────────
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

# Fresh vertcat objects post-setup (avoids do_mpc/CasADi "free variable" error)
# NOTE: x_full is the 12-dim PHYSICAL state only -- this is what gets passed
# to the policy/critic networks below, matching their trained input dim.
x_full = vertcat(x_, y_, z_, vx_, vy_, vz_, phi_, theta_, psi_, p_, q_, r_)
u_vec = vertcat(vxref, vyref, vzref)


# ─────────────────────────────────────────────
# 4. Safety Filter (MPC)
# ─────────────────────────────────────────────
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

if USE_Q_COST:
    if USE_SQP:
        u_free = SX.sym('u_free', nu)
        Q_free_sym = Q_casadi_fn(x_full, u_free)
        Q_u_expr = jacobian(Q_free_sym, u_free).T
        Q_uu_expr = hessian(Q_free_sym, u_free)[0]
        Q_u_sym = substitute(Q_u_expr, u_free, u_theta_sym)
        Q_uu_sym = substitute(Q_uu_expr, u_free, u_theta_sym)
        lterm = cost_weight * -(mtimes(Q_u_sym.T, du) + 0.5 * mtimes([du.T, Q_uu_sym, du]))
    else:
        lterm = cost_weight * weight_Q * (Q_val_sym - Q_theta_sym) ** 2 + cost_weight * weight_U * mtimes(du.T, du)
else:
    lterm = cost_weight * mtimes(du.T, du)

mterm = SX(0)
mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(vxref=1e-6, vyref=1e-6, vzref=1e-6)

# Input bounds
for i, name in enumerate(['vxref', 'vyref', 'vzref']):
    mpc.bounds['lower', '_u', name] = u_min[i]
    mpc.bounds['upper', '_u', name] = u_max[i]

# # State bounds (all 15 states -- 12 physical + 3 augmented "previous input")
# for i, name in enumerate(state_names):
#     mpc.bounds['lower', '_x', name] = x_min[i]
#     mpc.bounds['upper', '_x', name] = x_max[i]

bounded_state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'vxref_prev', 'vyref_prev', 'vzref_prev']
for name in bounded_state_names:
    i = state_names.index(name)
    mpc.bounds['lower', '_x', name] = x_min[i]
    mpc.bounds['upper', '_x', name] = x_max[i]

# # Terminal heuristic terms: unchanged from before -- driving the terminal
# # velocity/inputs toward zero on the last horizon stage.
# mpc.set_nl_cons('term_vxref_ub',  term_switch * vxref, ub=1e-4)
# mpc.set_nl_cons('term_vxref_lb', -term_switch * vxref, ub=1e-4)
# mpc.set_nl_cons('term_vyref_ub',  term_switch * vyref, ub=1e-4)
# mpc.set_nl_cons('term_vyref_lb', -term_switch * vyref, ub=1e-4)
# mpc.set_nl_cons('term_vzref_ub',  term_switch * vzref, ub=1e-4)
# mpc.set_nl_cons('term_vzref_lb', -term_switch * vzref, ub=1e-4)
# mpc.set_nl_cons('term_vx_ub',  term_switch * vx_, ub=1e-4)
# mpc.set_nl_cons('term_vx_lb', -term_switch * vx_, ub=1e-4)
# mpc.set_nl_cons('term_vy_ub',  term_switch * vy_, ub=1e-4)
# mpc.set_nl_cons('term_vy_lb', -term_switch * vy_, ub=1e-4)
# mpc.set_nl_cons('term_vz_ub',  term_switch * vz_, ub=1e-4)
# mpc.set_nl_cons('term_vz_lb', -term_switch * vz_, ub=1e-4)

# NEW: Rate constraints on the three inputs, via the augmented
# "previous input" states -- a_min <= u_k - u_{k-1} <= a_max.
mpc.set_nl_cons('rate_vxref_ub',  vxref - vxref_prev_, ub=a_max[0])
mpc.set_nl_cons('rate_vxref_lb', -(vxref - vxref_prev_), ub=-a_min[0])
mpc.set_nl_cons('rate_vyref_ub',  vyref - vyref_prev_, ub=a_max[1])
mpc.set_nl_cons('rate_vyref_lb', -(vyref - vyref_prev_), ub=-a_min[1])
mpc.set_nl_cons('rate_vzref_ub',  vzref - vzref_prev_, ub=a_max[2])
mpc.set_nl_cons('rate_vzref_lb', -(vzref - vzref_prev_), ub=-a_min[2])

# Obstacle constraint: keep (x,y) outside the circle (obs_center, obs_radius)
# -- top-view / x-y projection, same convention as before.
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


# ─────────────────────────────────────────────
# 5. Initial conditions.
#    NOTE: no do_mpc Simulator here -- the TRUE plant is the full nonlinear
#    env itself (env.step), unchanged. The filter's PHYSICAL 12-dim state
#    is exactly the true plant's reported state; the 3 augmented
#    "previous input" states are tracked separately (outside the env) from
#    whatever u_safe was actually applied last.
#
#    At t=0 there is no prior applied input, so we initialize the
#    "previous input" augmented states to zero (hover) -- i.e. we assume
#    no rate-limit violation relative to a hovering start.
# ─────────────────────────────────────────────
env.state = x0_phys.astype(np.float32).copy()  # bypass reset()'s random sampling

x0_aug = np.concatenate([x0_phys.copy(), np.zeros(n_aug)])
mpc.x0 = x0_aug.reshape(-1, 1)
mpc.set_initial_guess()


# ─────────────────────────────────────────────
# 6. Simulation loop
# ─────────────────────────────────────────────
x_hist = [x0_phys.copy()]  # full 12-dim true-plant state, for diagnostics/plots
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
    x_phys_now = env.state.astype(np.float64)  # true 12-dim nonlinear plant state
                                                 # -- policy/critic see this directly

    u_theta = policy_numpy(x_phys_now)
    Q_theta_hist.append(Q_numeric_fn(x_phys_now, u_theta))

    start = time.perf_counter()
    u_safe = mpc.make_step(x0_aug_now)
    elapsed = time.perf_counter() - start  # seconds
    timings.append(elapsed)

    # print(mpc.data.prediction(('_u', 'vxref', 0))[0, :, 0])
    # print(mpc.data.prediction(('_u', 'vyref', 0))[0, :, 0])

    return_status = mpc.S.stats()['return_status']
    if return_status == 'Infeasible_Problem_Detected':
        raise RuntimeError(
            f"MPC safety filter genuinely infeasible at step {t} "
            f"(x0={x0_aug_now.flatten()}, return_status={return_status}). "
            f"Refusing to apply an unverified action."
        )

    Q_safe_hist.append(Q_numeric_fn(x_phys_now, u_safe.flatten()))

    # TRUE PLANT: full nonlinear env, internally runs LQR + RK4 at h=env.h
    # (n_substeps ticks) for this single outer dt step -- unchanged.
    obs, reward, terminated, truncated, info = env.step(u_safe.flatten())
    episode_reward += reward

    # Advance the augmented state: physical part from the true plant,
    # "previous input" part from whatever was actually applied this step
    # (so the NEXT step's rate constraint is relative to the real action).
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

# ─────────────────────────────────────────────
# 7. Plot results
# ─────────────────────────────────────────────
Q_theta_hist = np.array(Q_theta_hist)
Q_safe_hist = np.array(Q_safe_hist)
x_hist = np.vstack(x_hist)  # full 12-dim (true plant)
u_hist = np.vstack(u_hist)
u_theta_hist = np.vstack(u_theta_hist)
n_steps = len(u_hist)
t_axis = np.arange(n_steps) * dt
t_axis_x = np.arange(n_steps + 1) * dt

# ── Save closed-loop state and input trajectories to file ──────────
traj_path = f'psf_quad_trajectory_PASF_{args.run_id}.npz' if args.run_id else 'psf_quad_trajectory.npz'
np.savez(
    traj_path,
    t_axis=t_axis,               # (n_steps,)   time at each input step
    t_axis_x=t_axis_x,           # (n_steps+1,) time at each state step
    x_hist=x_hist,                # (n_steps+1, 12)  true-plant physical states
                                   #   [x,y,z,vx,vy,vz,phi,theta,psi,p,q,r]
    u_hist=u_hist,                 # (n_steps, 3)  [vxref,vyref,vzref] SAFE inputs applied
    u_theta_hist=u_theta_hist,     # (n_steps, 3)  [vxref,vyref,vzref] REFERENCE (unfiltered) inputs
    Q_theta_hist=Q_theta_hist,     # (n_steps,)    Q(s, u_theta) -- proposed action value
    Q_safe_hist=Q_safe_hist,       # (n_steps,)    Q(s, u_safe)  -- applied action value
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

# -- Top-view (x-y) trajectory with obstacle projection --
fig3, ax3 = plt.subplots(figsize=(6, 6))
theta_c = np.linspace(0, 2 * np.pi, 100)
ax3.plot(obs_center[0] + obs_radius * np.cos(theta_c),
         obs_center[1] + obs_radius * np.sin(theta_c), 'r-', label='obstacle (x-y projection)')
ax3.plot(x_hist[:, 0], x_hist[:, 1], 'b-', linewidth=2, label='trajectory')
ax3.plot(x0_phys[0], x0_phys[1], 'go', markersize=8, label='start')
ax3.plot(x_goal[0], x_goal[1], 'k*', markersize=12, label='goal')
ax3.set_xlabel('x'); ax3.set_ylabel('y')
ax3.set_title('Top-view (x-y) trajectory')
ax3.legend(); ax3.grid(alpha=0.3); ax3.set_aspect('equal')
plt.tight_layout()

plt.savefig('psf_quad_result.png', dpi=100)
plt.show()
print("Done!")