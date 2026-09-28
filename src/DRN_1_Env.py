import gymnasium as gym
from gymnasium import spaces
import numpy as np
import scipy.linalg


def rotation_matrix(phi, theta, psi):

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

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Physical parameters
        self.m = 1.0
        self.g = 9.81
        self.J = np.diag([0.02, 0.02, 0.04])
        self.J_inv = np.linalg.inv(self.J)
        self.drag = 0.2

        self.dt = 0.1
        self.n_substeps = 10
        self.h = self.dt / self.n_substeps

        self.hover_thrust = self.m * self.g
        self.u_hover = np.array([self.hover_thrust, 0.0, 0.0, 0.0])

        self.u_min = np.array([0.0, -0.5, -0.5, -0.2])
        self.u_max = np.array([2.0 * self.hover_thrust, 0.5, 0.5, 0.2])

        self.A, self.B = self._linearize()
        Q = np.diag([2.0, 2.0, 2.0,
                     50.0, 50.0, 20.0,
                     5.0, 5.0, 5.0])
        R = np.diag([0.05, 5.0, 5.0, 2.0])
        P = scipy.linalg.solve_continuous_are(self.A, self.B, Q, R)
        self.K = np.linalg.inv(R) @ self.B.T @ P
        self.Q_lqr, self.R_lqr = Q, R

        self.vref_min = np.array([-0.5, -0.5, -0.1])
        self.vref_max = np.array([ 0.5,  0.5,  0.1])

        pos_lim, ang_lim, rate_lim = 5.0, np.pi / 2, 5.0
        self.x_min = np.array([-pos_lim, -pos_lim, 0.0, *self.vref_min,
                                -ang_lim, -ang_lim, -np.pi, -rate_lim, -rate_lim, -rate_lim])
        self.x_max = np.array([ pos_lim,  pos_lim, 2*pos_lim, *self.vref_max,
                                 ang_lim,  ang_lim,  np.pi, rate_lim, rate_lim, rate_lim])

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

        # Translational dynamics
        thrust_world = R @ np.array([0.0, 0.0, T])
        vel_dot = thrust_world / self.m - np.array([0.0, 0.0, self.g]) - (self.drag / self.m) * vel

        # Euler-rate kinematics
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

        # Rigid-body rotational dynamics
        omega_dot = self.J_inv @ (tau - np.cross(omega, self.J @ omega))

        return np.concatenate([vel, vel_dot, eta_dot, omega_dot])

    def _reduced_dynamics(self, z, u):
        state = np.concatenate([np.zeros(3), z])
        return self._dynamics(state, u)[3:12]

    def _linearize(self, eps=1e-6):
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
        self.state = np.concatenate([pos, np.zeros(3), np.zeros(3), np.zeros(3)]).astype(np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        return self.state.astype(np.float32)

    def step(self, action):
        v_ref = np.clip(np.asarray(action, dtype=np.float64), self.vref_min, self.vref_max)
        z_ref = np.concatenate([v_ref, np.zeros(6)])  # attitude/rate refs always 0
        state = self.state.astype(np.float64)

        for _ in range(self.n_substeps):
            z = state[3:12]
            u = self.u_hover - self.K @ (z - z_ref)
            u = np.clip(u, self.u_min, self.u_max)
            state = self._rk4_step(state, u, self.h)

        state[6:9] = (state[6:9] + np.pi) % (2 * np.pi) - np.pi  # wrap Euler angles
        self.state = state.astype(np.float32)

        pos = state[0:3]
        att = state[6:9]

        # Reward
        pos_error = np.linalg.norm(pos - self.goal_pos)
        reward = - pos_error - 0.1 * np.linalg.norm(v_ref) - 4 * np.maximum(0, -pos[1])

        terminated = bool(pos_error < 0.05)
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