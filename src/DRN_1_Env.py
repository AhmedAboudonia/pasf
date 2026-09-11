import gymnasium as gym
from gymnasium import spaces
import numpy as np
import scipy.linalg


def rotation_matrix(phi, theta, psi):
    """Body-to-world rotation matrix, ZYX Euler convention (yaw-pitch-roll)."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    Rz = np.array([[cpsi, -spsi, 0.0],
                   [spsi,  cpsi, 0.0],
                   [0.0,   0.0,  1.0]])
    Ry = np.array([[cth, 0.0, sth],
                   [0.0, 1.0, 0.0],
                   [-sth, 0.0, cth]])
    Rx = np.array([[1.0, 0.0,  0.0],
                   [0.0, cphi, -sphi],
                   [0.0, sphi,  cphi]])
    return Rz @ Ry @ Rx


class NonlinearQuadrotorEnv(gym.Env):
    """
    Full nonlinear 6-DOF rigid-body quadrotor, controlled through a cascaded
    architecture:

        RL / outer loop  -->  velocity reference  -->  LQR inner loop  -->
        low-level [T, tau_x, tau_y, tau_z]  -->  nonlinear rigid-body plant

    Full state (12):  [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        x,y,z         - position, world frame
        vx,vy,vz      - velocity, world frame
        phi,theta,psi - roll, pitch, yaw (ZYX Euler angles, body attitude)
        p,q,r         - body-frame angular rates

    Action (3), the ONLY thing an RL policy has to output:
        [vx_ref, vy_ref, vz_ref]   -- desired world-frame velocity

    Everything else -- attitude, rates, and the low-level thrust/torques
    needed to realize that velocity -- is handled internally by an LQR
    controller designed on the system LINEARIZED about hover:

        z = [vx, vy, vz, phi, theta, psi, p, q, r]        (9 states)
        u = [T, tau_x, tau_y, tau_z]                       (4 inputs)
        z_dot = A @ (z - 0) + B @ (u - u_hover)             (about trim)

    A, B are obtained by finite-difference linearization of the same
    nonlinear f(x, u) used for simulation (see _reduced_dynamics /
    _linearize), so the LQR is consistent with the plant it controls.
    Note: position doesn't enter z_dot at all (gravity is constant), so the
    9-state reduced system is fully decoupled from x, y, z -- that's what
    makes this linearization valid regardless of where the drone actually is.

    Control law each inner-loop tick:
        u = u_hover - K @ (z - z_ref),   z_ref = [vx_ref,vy_ref,vz_ref,0,0,0,0,0,0]

    Because there's linear drag opposing velocity, a proportional-only LQR
    (no integral action) will show a small steady-state velocity-tracking
    error at nonzero vref -- the controller needs a persistent tilt to fight
    drag, and pure state feedback under-provides it slightly. This is
    expected, not a bug.

    Reward: negative distance of (x, y, z) to the origin -- the RL's only
    job is to pick velocity references that drive the drone home.

    Timing:
        self.dt          - outer step (RL / env.step), 0.2 s -> 5 Hz
        self.n_substeps  - inner LQR/physics ticks per outer step, 4
        self.h           - inner tick size = dt / n_substeps = 0.05 s -> 20 Hz
    Each call to step() runs the LQR + one RK4 physics integration step,
    n_substeps times, holding the RL's velocity reference constant
    throughout (zero-order hold), exactly like a real cascaded
    guidance/attitude control stack.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ── Physical parameters (roughly a small quadrotor) ──
        self.m = 1.0                                   # mass, kg
        self.g = 9.81                                   # gravity, m/s^2
        self.J = np.diag([0.02, 0.02, 0.04])            # body inertia, kg*m^2
        self.J_inv = np.linalg.inv(self.J)
        self.drag = 0.2                                 # linear translational drag

        self.dt = 0.1                                      # outer step, 5 Hz
        # NOTE on inner-loop rate: the attitude/rate LQR has closed-loop
        # poles as fast as ~-50 rad/s (time constant ~20 ms) -- see the
        # eigenvalues of A - B@K. A zero-order-hold sampled implementation
        # needs h well below that time constant (5-10x) to stay stable.
        # h = 0.05 s (20 Hz) is SLOWER than that pole and destabilizes the
        # horizontal (vx, vy) loop into a sustained oscillation, verified
        # empirically -- so the inner loop is kept fast here even though
        # the outer RL step dt was slowed to 0.2 s.
        self.n_substeps = 10                                # inner LQR/RK4 ticks per step
        self.h = self.dt / self.n_substeps                  # inner tick, 0.01 s -> 100 Hz

        # Hover trim: thrust that exactly balances gravity, level attitude
        self.hover_thrust = self.m * self.g
        self.u_hover = np.array([self.hover_thrust, 0.0, 0.0, 0.0])

        # Low-level actuator saturation (thrust can't go negative; torques
        # are physically limited) -- the LQR output is clipped to this
        self.u_min = np.array([0.0, -0.5, -0.5, -0.2])
        self.u_max = np.array([2.0 * self.hover_thrust, 0.5, 0.5, 0.2])

        # ── Design the inner-loop LQR by linearizing about hover ──
        self.A, self.B = self._linearize()
        Q = np.diag([2.0, 2.0, 2.0,      # vx, vy, vz tracking error
                     50.0, 50.0, 20.0,   # phi, theta, psi -- keep attitude tight
                     5.0, 5.0, 5.0])     # p, q, r -- damp rates
        R = np.diag([0.05, 5.0, 5.0, 2.0])  # thrust cheap to use, torques penalized more
        P = scipy.linalg.solve_continuous_are(self.A, self.B, Q, R)
        self.K = np.linalg.inv(R) @ self.B.T @ P
        self.Q_lqr, self.R_lqr = Q, R  # kept around for reference/inspection

        # RL action: desired world-frame velocity. These bounds also define
        # the outer state box for velocity, along with a position box and
        # attitude/rate box used only for reset sampling / sanity, not
        # enforced as hard constraints inside step() (same convention as
        # the earlier envs -- a wrapper or safety filter would do that).
        self.vref_min = np.array([-0.5, -0.5, -0.1])
        self.vref_max = np.array([ 0.5,  0.5,  0.1])

        pos_lim, ang_lim, rate_lim = 5.0, np.pi / 2, 5.0
        self.x_min = np.array([-pos_lim, -pos_lim, 0.0, *self.vref_min,
                                -ang_lim, -ang_lim, -np.pi, -rate_lim, -rate_lim, -rate_lim])
        self.x_max = np.array([ pos_lim,  pos_lim, 2*pos_lim, *self.vref_max,
                                 ang_lim,  ang_lim,  np.pi, rate_lim, rate_lim, rate_lim])

        # Goal: hover at (0, 0, goal_z) -- deliberately NOT ground level
        # (z=0), since that would coincide with the crash condition in
        # step() and make "reaching the goal" indistinguishable from
        # "crashing"
        self.goal_pos = np.array([0.0, 0.0, 3.0])

        self.action_space = spaces.Box(
            low=self.vref_min.astype(np.float32),
            high=self.vref_max.astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        self.state = None

    # ------------------------------------------------------------------ #
    # Dynamics
    # ------------------------------------------------------------------ #
    def _dynamics(self, state, u):
        """Continuous-time nonlinear state derivative, f(x, u), full 12-state."""
        vel = state[3:6]
        phi, theta, psi = state[6:9]
        omega = state[9:12]
        T, tau = u[0], u[1:4]

        R = rotation_matrix(phi, theta, psi)

        # Translational dynamics: rotated thrust - gravity - drag
        thrust_world = R @ np.array([0.0, 0.0, T])
        vel_dot = thrust_world / self.m - np.array([0.0, 0.0, self.g]) - (self.drag / self.m) * vel

        # Euler-rate kinematics (nonlinear map from body rates to angle rates)
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth = np.cos(theta)
        cth = cth if abs(cth) > 1e-3 else 1e-3 * np.sign(cth if cth != 0 else 1.0)
        tth = np.tan(theta)
        W = np.array([
            [1.0, sphi * tth, cphi * tth],
            [0.0, cphi,       -sphi],
            [0.0, sphi / cth, cphi / cth],
        ])
        eta_dot = W @ omega

        # Rigid-body rotational dynamics (Euler's equation): the
        # omega x (J omega) term is the nonlinear gyroscopic coupling
        omega_dot = self.J_inv @ (tau - np.cross(omega, self.J @ omega))

        return np.concatenate([vel, vel_dot, eta_dot, omega_dot])

    def _reduced_dynamics(self, z, u):
        """z_dot for the 9-state subsystem [vx,vy,vz,phi,theta,psi,p,q,r].
        Position doesn't affect these dynamics, so we can plug z into the
        full 12-state f(x,u) with a dummy zero position and read off the
        last 9 components."""
        state = np.concatenate([np.zeros(3), z])
        return self._dynamics(state, u)[3:12]

    def _linearize(self, eps=1e-6):
        """Finite-difference linearization of the reduced dynamics about
        hover (z=0, u=u_hover), giving A = df/dz, B = df/du."""
        z0 = np.zeros(9)
        u0 = np.array([self.hover_thrust, 0.0, 0.0, 0.0])
        f0 = self._reduced_dynamics(z0, u0)

        n, m = 9, 4
        A = np.zeros((n, n))
        B = np.zeros((n, m))
        for i in range(n):
            dz = np.zeros(n); dz[i] = eps
            A[:, i] = (self._reduced_dynamics(z0 + dz, u0) - f0) / eps
        for j in range(m):
            du = np.zeros(m); du[j] = eps
            B[:, j] = (self._reduced_dynamics(z0, u0 + du) - f0) / eps
        return A, B

    def _rk4_step(self, state, u, h):
        k1 = self._dynamics(state, u)
        k2 = self._dynamics(state + 0.5 * h * k1, u)
        k3 = self._dynamics(state + 0.5 * h * k2, u)
        k4 = self._dynamics(state + h * k3, u)
        return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pos = self.np_random.uniform(low=[-4.0, -4.0, 2.0], high=[4.0, 4.0, 4.0])
        # att = self.np_random.uniform(low=-0.1, high=0.1, size=3)  # near-level start
        self.state = np.concatenate([pos, np.zeros(3), np.zeros(3), np.zeros(3)]).astype(np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        return self.state.astype(np.float32)

    def step(self, action):
        v_ref = np.clip(np.asarray(action, dtype=np.float64), self.vref_min, self.vref_max)
        z_ref = np.concatenate([v_ref, np.zeros(6)])  # attitude/rate refs always 0
        state = self.state.astype(np.float64)

        # Inner loop: n_substeps LQR + RK4 ticks per outer env.step(), the
        # RL's velocity reference held constant (zero-order hold)
        for _ in range(self.n_substeps):
            z = state[3:12]
            u = self.u_hover - self.K @ (z - z_ref)
            u = np.clip(u, self.u_min, self.u_max)
            state = self._rk4_step(state, u, self.h)

        state[6:9] = (state[6:9] + np.pi) % (2 * np.pi) - np.pi  # wrap Euler angles
        self.state = state.astype(np.float32)

        pos = state[0:3]
        att = state[6:9]

        # Reward: negative distance of (x,y,z) to the origin
        pos_error = np.linalg.norm(pos - self.goal_pos)
        reward = - pos_error - 0.1 * np.linalg.norm(v_ref) - 4 * np.maximum(0, -pos[1])

        # Done conditions
        terminated = bool(pos_error < 0.05)
        # A crash (hit the ground) or exceeding the LQR's valid linearization
        # region (large tilt) ends the episode -- physical events, handled
        # here rather than left to an external wrapper
        crashed = bool(pos[2] <= 0.0 or np.any(np.abs(att[:2]) > np.pi / 2 - 0.05))
        truncated = crashed

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        x, y, z = self.state[0:3]
        phi, theta, psi = np.degrees(self.state[6:9])
        print(f"Pos: ({x:5.2f},{y:5.2f},{z:5.2f}) | "
              f"Att[deg]: (roll={phi:5.1f}, pitch={theta:5.1f}, yaw={psi:5.1f})")


if __name__ == "__main__":
    env = NonlinearQuadrotorEnv()
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()