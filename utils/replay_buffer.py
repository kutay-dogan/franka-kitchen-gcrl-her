import numpy as np
import torch
# REPLAY BUFFER CLASS


class ReplayBuffer:
    def __init__(
        self,
        state_dim,
        action_dim,
        ohe_dim,
        goal_dim,
        device,
        max_size=100_000,
        seed=42,
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.ohe = np.zeros((max_size, ohe_dim))
        self.goal = np.zeros((max_size, goal_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((self.max_size, 1))

    def add(self, state, action, ohe, goal, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.ohe[self.ptr] = ohe
        self.goal[self.ptr] = goal
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.choice(self.size, size=batch_size, replace=False)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.ohe[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device),
        )
