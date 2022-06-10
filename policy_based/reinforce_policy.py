import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from policy_based.template_policy import TemplatePolicy
from environments.environment import Environment
import torch.nn.functional as F


class ReinforcePolicy(TemplatePolicy):
    def __init__(
        self,
        env: Environment,
        model: torch.nn.Module,
        lr,
        gamma,
        name,
        step_size:int,
        baseline:bool,
        entropy:tuple,
        M:int
    ):
        super().__init__(env, model, lr, gamma, name)
        self.entropy_eta, self.eta_decrease_factor = entropy

    def update_target_weights(self, epoch):
        pass
    
    def optimize_per_step(self, t, episode_return, selected_action_probs):
      pass

    def optimize_per_epoch(self, epoch_experiences, selected_action_probs):
      epoch_experiences = epoch_experiences[0]
      selected_action_probs = selected_action_probs[0]
      ep_returns = []
      
      for t in range(len(epoch_experiences)):
        episode_return = 0
        for i in range(t, len(epoch_experiences)):
            episode_return += self.gamma**(i-t) * epoch_experiences[i].reward
        ep_returns.append(episode_return)
        
      eps = np.finfo(np.float32).eps.item()

      ep_returns = (ep_returns - np.mean(ep_returns)) / (np.std(ep_returns) + eps)

      policy_loss = []
      
      for log_prob, ep_return in zip(selected_action_probs, ep_returns):
        policy_loss.append(-log_prob*ep_return)
      
      
      self.optimizer.zero_grad()
      loss = torch.stack(policy_loss).sum()

      loss.backward()
      self.optimizer.step()