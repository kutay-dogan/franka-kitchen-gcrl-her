import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.actor_critic import Actor, Critic
import copy


class DDPGAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        ohe_dim,
        goal_dim,
        device,
        actor_lr=1e-4,
        critic_lr=2e-4,
        gamma=0.99,
        tau=0.001,
    ):
        self.device = device

        # discount factor
        self.gamma = gamma

        # hyperparameter for polyak average
        self.tau = tau

        # creating actor, critic with their target networks
        self.actor = Actor(state_dim, action_dim, ohe_dim, goal_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, ohe_dim, goal_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # creating an actor network for adding noise (optional, we did not use this network with final results!)
        self.perturbed_actor = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, ohe_dim, goal_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, ohe_dim, goal_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    # function for adding noise to the parameters.
    def perturb_policy(self, noise_scale=0.1):
        self.perturbed_actor.load_state_dict(self.actor.state_dict())
        with torch.no_grad():
            for param in self.perturbed_actor.parameters():
                # create and add scaled noise from standart normal distribution
                noise = torch.randn_like(param) * noise_scale
                param.add_(noise)

    def select_action(
        self,
        state,
        ohe,
        goal,
        noise=0.1,
        use_parameter_noise=False,
        min_action=-1,
        max_action=1,
    ):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        ohe = torch.FloatTensor(ohe).unsqueeze(0).to(self.device)
        goal = torch.FloatTensor(goal).unsqueeze(0).to(self.device)

        self.actor.eval()
        self.perturbed_actor.eval()

        with torch.no_grad():
            # sample action from the actor with the parameter noise (optional).
            if use_parameter_noise:
                action = (
                    self.perturbed_actor(state, ohe, goal).cpu().data.numpy().flatten()
                )
            else:
                action = self.actor(state, ohe, goal).cpu().data.numpy().flatten()

        self.actor.train()

        #
        if not use_parameter_noise and noise > 0:
            action = action + np.random.normal(0, noise, size=action.shape)

        # clipping actions. for franka kitchen [-1,1]
        return np.clip(action, min_action, max_action)

    def train(self, replay_buffer, batch_size=256):
        # take a sample batch from the replay buffer
        state, action, ohe, goal, next_state, reward, done = replay_buffer.sample(
            batch_size
        )

        with torch.no_grad():
            # get the next action and target q values from target networks
            next_action = self.actor_target(next_state, ohe, goal)
            target_q = self.critic_target(next_state, ohe, next_action, goal)

            # update q values
            # If done=1, term becomes 0. If done=0, term becomes (gamma * old target_q).
            target_q = reward + ((1 - done) * self.gamma * target_q)

        # getting current estimate of the q values.
        current_q = self.critic(state, ohe, action, goal)

        # calculate loss and update networks
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        predicted_action = self.actor(state, ohe, goal)
        actor_loss = -self.critic(state, ohe, predicted_action, goal).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # using polyak average for updating to mitigate moving target problem
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

        return critic_loss.item(), actor_loss.item(), current_q.mean().item()
