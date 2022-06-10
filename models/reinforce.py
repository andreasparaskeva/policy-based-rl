from platform import release
import numpy as np
from torch import nn, relu
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class ReinforceModel(nn.Module):
    def __init__(self, input_size, n_actions, hidden_size=[512, 128]):
        super(ReinforceModel, self).__init__()
        self.input_size = input_size
        self.n_actions = n_actions
        self.hidden_sizes = hidden_size
        self.h1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.hidden = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.selu = nn.SELU()
        self.h2 = nn.Linear(self.hidden_sizes[1], n_actions)

    def forward(self, x):

        x = self.h1(x.float())
        x = F.relu(x)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.h2(x)
        action_prob = F.softmax(x, dim=-1)

        return self.select_action(action_prob)
       

    def select_greedy_action(self, x):
        x = self.h1(x.float())
        x = F.relu(x)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.h2(x)
        action_prob = F.softmax(x, dim=-1)
        return torch.argmax(action_prob).item()

    def select_action(self, probs):
        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        return action.item(), m.log_prob(action)
         
        
    def name(self):
        return f'reinforce-{self.hidden_sizes}'

    def reset(self):
        return ReinforceModel(self.input_size, self.n_actions, self.hidden_sizes)
