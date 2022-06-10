import gym
import torch
import numpy as np
from environments.environment import Environment, Experience

class CartPoleEnv(Environment):

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make('CartPole-v1')
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.env.reset()
        self.max_reward = 500
        
    def state(self):
        return self.env.state
        # return torch.tensor(np.array(self.env.state), requires_grad=True).float().to(device=self.device)

    def reset(self):
        return self.env.reset()
        
    def step(self, action, evaluate_mode=False):
        state = self.state()
        next_state, reward, done, _ = self.env.step(action)
        # next_state = torch.tensor(np.array(next_state), requires_grad=True).float().to(device=self.device)
        if done and not evaluate_mode:
            reward = -reward

        return Experience(
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state
        )

    def name(self):
        return 'cart-pole'
