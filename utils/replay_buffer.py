import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, goal_dim, device, max_size=100_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.desired_goal = np.zeros((max_size, goal_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, desired_goal, next_state, reward, done):        
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.desired_goal[self.ptr] = desired_goal
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        
        ind = np.random.choice(self.size, size=batch_size, replace=False)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.desired_goal[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )