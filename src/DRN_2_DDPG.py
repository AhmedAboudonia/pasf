import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

from DRN_1_Env import NonlinearQuadrotorEnv
from DRN_3_Plot import plot_quad_learning_curve, plot_quad_trajectory, plot_quad_Qfunction, animate_quad

# --- Hyperparameters ---
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 0.001
LR_CRITIC = 0.002
BATCH_SIZE = 64
BUFFER_SIZE = 100_000
TOTAL_EPISODES = 2_000
MAX_STEPS = 300  # 150 * dt(0.1s) = 15s of simulated flight per episode

# --- Policy/critic observability ---
# The env's full observation is [x,y,z,vx,vy,vz,phi,theta,psi,p,q,r] (12-dim).
# Unlike the translational-only variant of this script, here the policy
# (actor) and Q-function (critic) are given the FULL state -- position,
# velocity, attitude, and body rates -- rather than being restricted to
# the 6-dim [x,y,z,vx,vy,vz] slice the cascaded LQR design would otherwise
# make sufficient. This lets pi and Q condition on everything the env
# reports, at the cost of a larger input dimension for both networks.
POLICY_STATE_DIM = 12  # x, y, z, vx, vy, vz, phi, theta, psi, p, q, r


def extract_policy_state(obs):
    """Identity pass-through: the policy/critic now see the full env
    observation, so no slicing is needed. Kept as a function (rather than
    just using obs directly) so the training loop below doesn't need to
    change shape depending on which state representation is in use."""
    return obs


# --- Neural Networks ---
# NOTE: architecture (Linear -> gelu -> Linear -> gelu -> Linear) unchanged.
# Only the input dimensions differ from the translational-only variant:
# state_dim is now POLICY_STATE_DIM (12) instead of 6, and the Critic's
# input is state(12) + action(3) = 15 instead of state(6) + action(3) = 9.
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, action_dim)
        # Quad env's vref_min == -vref_max (symmetric), so a plain
        # tanh * max_action squashing is exact here -- no need for an
        # asymmetric-bound split.
        self.max_action = torch.FloatTensor(max_action)

    def forward(self, state):
        x = torch.nn.functional.gelu(self.l1(state))
        x = torch.nn.functional.gelu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.nn.functional.gelu(self.l1(x))
        x = torch.nn.functional.gelu(self.l2(x))
        return self.l3(x)

# --- The Agent ---
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.max_action = max_action  # numpy array, used for the clip below

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        action = action + np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE: return

        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state))
        action = torch.FloatTensor(np.array(action))
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(np.array(done)).unsqueeze(1)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * GAMMA * target_Q

        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

    def save_checkpoint(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        # Also sync target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        print(f"Checkpoint {filename} loaded!")

    def export_to_npz(self, filepath="nn_weights.npz"):
        """
        Dimension-agnostic export (policy_W0,b0,... / critic_W0,b0,...);
        works for any Actor/Critic regardless of state/action dims.
        """
        def linear_layers(module):
            return [m for m in module.children() if isinstance(m, nn.Linear)]

        out = {}
        for i, layer in enumerate(linear_layers(self.actor)):
            out[f"policy_W{i}"] = layer.weight.detach().cpu().numpy()
            out[f"policy_b{i}"] = layer.bias.detach().cpu().numpy()
        for i, layer in enumerate(linear_layers(self.critic)):
            out[f"critic_W{i}"] = layer.weight.detach().cpu().numpy()
            out[f"critic_b{i}"] = layer.bias.detach().cpu().numpy()

        np.savez(filepath, **out)
        print(f"Exported weights to {filepath}")


# --- Training Loop ---
env = NonlinearQuadrotorEnv(render_mode="human")
# NOTE: agent is now built with POLICY_STATE_DIM (12), which matches
# env.observation_space.shape[0] exactly -- pi and Q see the full state.
agent = DDPGAgent(POLICY_STATE_DIM, env.action_space.shape[0], env.action_space.high)
rewards_history = []

print("Starting Training...")
for episode in range(TOTAL_EPISODES):
    state, _ = env.reset()
    policy_state = extract_policy_state(state)
    episode_reward = 0

    for t in range(MAX_STEPS):
        action = agent.select_action(policy_state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_policy_state = extract_policy_state(next_state)
        done = terminated or truncated

        agent.replay_buffer.append((policy_state, action, reward, next_policy_state, done))
        agent.train()

        policy_state = next_policy_state
        episode_reward += reward
        if done: break

    rewards_history.append(episode_reward)
    if episode % 10 == 0:
        status = "CRASHED" if truncated else ("REACHED GOAL" if terminated else "TIMED OUT")
        print(f"Episode: {episode} | Reward: {episode_reward:.2f} | Steps: {t+1} | {status}")

env.close()


agent.save_checkpoint("ddpg_quad_full_state.pth")
agent.export_to_npz("nn_weights_quad_2_effort_300.npz")
# Quadrotor
# Since the actor/critic now take the full 12-dim observation directly
# (no slicing), plot_quad_trajectory / plot_quad_Qfunction / animate_quad
# can pass env observations straight into agent.actor / agent.critic
# without any extract_policy_state adaptation.
plot_quad_learning_curve(rewards_history)
plot_quad_trajectory(env, agent)
plot_quad_Qfunction(agent)
animate_quad(env, agent)

print("Training and plotting complete!")