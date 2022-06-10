import numpy as np
from torch import nn
import torch
from collections import namedtuple
import torch.nn.functional as F
from torch.distributions import Categorical
import random

SavedProb = namedtuple('SavedProb', field_names=['log_prob', 'v_val', 'entropy'])


class ActorCriticModel(nn.Module):
    """
    Implementation of Actor and Critic in one model
    """
    def __init__(self, input_size, n_actions, hidden_sizes=[128, 128]):
        super(ActorCriticModel, self).__init__()
        self.input_size = input_size
        self.n_actions = n_actions
        self.hidden_sizes = hidden_sizes

        self.affine = nn.Linear(self.input_size, self.hidden_sizes[0])

        self.hidden_layers = nn.Sequential(
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[0]),
            nn.ReLU(),
            # nn.Linear(self.hidden_sizes[1], self.hidden_sizes[1]),
            # nn.ReLU(),

        )

        # actor's layer
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_sizes[0], self.n_actions),
            # nn.Hardtanh(min_val=0, max_val=1),
        )
  
        # critic's layer
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_sizes[0], 1),
        )

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine(x.float()))

        # the actor will select an action from state s_t and 
        # returns probability of each possible action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        v_val = F.relu(self.value_head(x))
        # if random.uniform(0, 1) < 0.0001:
        #     print(f'Q_value: {q_val}\t|Action Prob: {action_prob}')

        # return action and prob
        return self.select_action(action_prob, v_val)
        # return action_prob, state_values

    def select_action(self, probs, v_val):
        # create a categorical distribution over the list of probabilities of actions
        # print(probs)
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        entropy1 = m.entropy()
        # print([ p * torch.log(p) for p in probs][0])
        entropy = torch.sum([ p * torch.log(p) for p in probs][0])
        # print(entropy1, entropy)
        # save to action buffer
        return action.item(), SavedProb(m.log_prob(action), v_val, entropy)

    def select_greedy_action(self, x):
        x = F.relu(self.affine(x.float()))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        return torch.argmax(action_prob).item()
        
    def name(self):
        return f'actor-critic-{self.hidden_sizes}'

    def reset(self):
        return ActorCriticModel(self.input_size, self.n_actions, self.hidden_sizes)
