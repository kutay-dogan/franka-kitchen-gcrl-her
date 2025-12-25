import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.actor_critic import Actor, Critic


class DDPGAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        goal_dim,
        device,
        actor_lr=2e-4,
        critic_lr=2e-4,
        gamma=0.90,
        tau=0.005,
    ):
        self.device = device

        self.actor = Actor(state_dim, action_dim, goal_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, goal_dim).to(device)

        # initializing target and main network identically
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, goal_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, goal_dim).to(device)

        # initializing target and main network identically
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau

    def select_action(self, state, goal, noise=0.1):
        # Helper to get action from state+goal
        # Expects state and goal to be numpy arrays
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        goal = torch.FloatTensor(goal).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state, goal).cpu().data.numpy().flatten()

        action = action + np.random.normal(0, noise, size=action.shape)

        return np.clip(action, -1, 1)

    def train(self, replay_buffer, batch_size=256):
        # 1. Sample Batch
        state, action, goal, next_state, reward, not_done = replay_buffer.sample(
            batch_size
        )

        # 2. Update Critic
        with torch.no_grad():
            next_action = self.actor_target(next_state, goal)
            target_q = self.critic_target(next_state, next_action, goal)
            target_q = reward + (not_done * self.gamma * target_q)

        current_q = self.critic(state, action, goal)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 3. Update Actor
        actor_loss = -self.critic(state, self.actor(state, goal), goal).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 4. Updating Target Network with POlyack Average
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
