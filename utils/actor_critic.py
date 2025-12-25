import torch
import torch.nn as nn
# ACTOR AND CRITIC CLASSES

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim+ goal_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),   
            nn.LeakyReLU(),
            nn.Linear(512, action_dim),
            nn.Tanh()
        )

    def forward(self, state, goal):
        return self.net(torch.cat([state, goal], 1))
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + goal_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action, goal):
        return self.net(torch.cat([state, action, goal], dim=1))