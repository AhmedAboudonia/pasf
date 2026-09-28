import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Plotting Functions
POLICY_STATE_DIM = 12  # x, y, z, vx, vy, vz, phi, theta, psi, p, q, r


def plot_quad_learning_curve(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history, color='teal')
    plt.title("DDPG Learning Curve (Quadrotor)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_quad_trajectory(env, agent, max_steps=300):
    print("Running test episode for trajectory...")
    state, _ = env.reset()
    x_coords, y_coords = [], []

    agent.actor.eval()
    with torch.no_grad():
        for t in range(max_steps):
            policy_state = state[:POLICY_STATE_DIM]
            state_t = torch.FloatTensor(policy_state).unsqueeze(0)
            action = agent.actor(state_t).numpy()[0]

            next_state, _, terminated, truncated, _ = env.step(action)
            x_coords.append(state[0])
            y_coords.append(state[1])

            state = next_state
            if terminated or truncated:
                break

    plt.figure(figsize=(7, 7))
    plt.plot(x_coords, y_coords, label='Path', color='blue', zorder=1)
    plt.scatter(x_coords[0], y_coords[0], color='green', label='Start', zorder=2)
    plt.scatter(x_coords[-1], y_coords[-1], color='red', marker='X', label='End', zorder=2)
    plt.scatter(0, 0, color='gold', marker='*', s=200, label='Goal', zorder=3)

    plt.title("Quadrotor Trajectory — Top View (x-y plane)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend(); plt.grid(True, alpha=0.3); plt.axis('equal')
    plt.show()


def plot_quad_Qfunction(agent, range_limit=5.0, z_fixed=3.0):
    num_points = 100
    x_range = np.linspace(-range_limit, range_limit, num_points)
    y_range = np.linspace(-range_limit, range_limit, num_points)
    q_values = np.zeros((num_points, num_points))

    fixed_rest = [z_fixed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    agent.actor.eval(); agent.critic.eval()
    with torch.no_grad():
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                state_t = torch.FloatTensor([x, y] + fixed_rest).unsqueeze(0)
                action_t = agent.actor(state_t)
                q_values[j, i] = agent.critic(state_t, action_t).item()

    plt.figure(figsize=(9, 7))
    cp = plt.pcolormesh(x_range, y_range, q_values, shading='auto', cmap='magma')
    plt.colorbar(cp, label='Q-Value')
    plt.scatter(0, 0, color='white', marker='*', s=200, label='Goal')
    plt.title(f"Critic Q-Value Map (z={z_fixed}, zero velocity, level attitude)")
    plt.xlabel("x"); plt.ylabel("y"); plt.axis('equal')
    plt.show()


def animate_quad(env, agent, max_steps=300):
    print("Running interactive test...")
    state, _ = env.reset()
    x_coords, y_coords = [], []

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))

    agent.actor.eval()
    with torch.no_grad():
        for t in range(max_steps):
            policy_state = state[:POLICY_STATE_DIM]
            state_t = torch.FloatTensor(policy_state).unsqueeze(0)
            action = agent.actor(state_t).numpy()[0]

            px, py = state[0], state[1]
            vx, vy = state[3], state[4]

            arrow_len = 0.3
            ux = arrow_len * vx
            uy = arrow_len * vy

            ax.clear()
            ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect('equal')

            x_coords.append(px); y_coords.append(py)
            ax.plot(x_coords, y_coords, 'b--')

            ax.scatter(px, py, color='blue')
            ax.quiver(px, py, ux, uy, color='red', scale=1, scale_units='xy')
            ax.scatter(0, 0, color='gold', marker='*', s=200)  # Goal

            plt.pause(0.1)

            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
            if terminated or truncated:
                break

    plt.ioff()
    plt.show()