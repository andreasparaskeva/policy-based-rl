import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from models.actor_critic import SavedProb

class ActorCriticContinuousModel(nn.Module):
    """
    Implementation of Actor and Critic in one model
    """
    def __init__(self, input_size, n_actions, hidden_sizes=[64, 64]):
        super(ActorCriticContinuousModel, self).__init__()
        self.input_size = input_size
        self.n_actions = n_actions
        self.hidden_sizes = hidden_sizes

        self.hidden_layers = nn.Sequential(
            nn.Linear(self.input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )

        self.means_head = nn.Sequential(
            nn.Linear(hidden_sizes[1], self.n_actions),
            # nn.Hardtanh(min_val=-3, max_val=3)
        )
        self.sigmas_head = nn.Sequential(
            nn.Linear(hidden_sizes[1], self.n_actions),
            # nn.Hardtanh(min_val=, max_val=0.8)
        )
  
        self.value_head = nn.Sequential(
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, x):
        x = self.hidden_layers(x.float())
        v_val = F.relu(self.value_head(x))
        means = 2*self.means_head(x)
        sigmas = self.sigmas_head(x)
        sigmas = torch.clamp(sigmas, -20, 2)
        sigmas = sigmas.exp()
        action_distribution = torch.distributions.Normal(means, sigmas)
        action = action_distribution.sample()
        # print(action)
        # print(means.item(), sigmas.item(), action.item()) if np.random.uniform() > 0.999 else None
        # print(action) if np.random.uniform() > 0.9999 else None
        
        return action.clone().cpu().detach().numpy(), \
               SavedProb(
                   action_distribution.log_prob(action),
                   v_val,
                   action_distribution.entropy()
               )

    def select_action(self, means, sigmas, v_val):
        # action_distribution = torch.distributions.Normal(0, 1)
        # print(sigmas)
        # print(means, sigmas) if np.random.uniform() > 0.999 else None
        action_distribution = torch.distributions.Normal(means, sigmas)
        action = action_distribution.sample()
        # print(action) if np.random.uniform() > 0.9999 else None
        # tt = torch.nn.Hardtanh(-1, 1)
        # action = tt(means + sigmas*z)
        # action_distribution = torch.distributions.Normal(means, sigmas)
        # action = action_distribution.sample()
        
        return action.clone().cpu().detach().numpy(), \
               SavedProb(
                   torch.sum(action_distribution.log_prob(action)),
                   v_val,
                   action_distribution.entropy()
               )

    def select_greedy_action(self, x):
        x = self.hidden_layers(x.float())
        means = 2*self.means_head(x)
        return means.clone().cpu().detach().numpy()

    def name(self):
        return f'actor-critic-{self.hidden_sizes}'

    def reset(self):
        return ActorCriticContinuousModel(self.input_size, self.n_actions, self.hidden_sizes)
