import gym
import numpy as np
import torch
from environments.environment import Environment, Experience

class Pendulum(Environment):

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make('Pendulum-v1', g=9.81)
        self.n_states = self.env.observation_space.shape[0]
        print(self.n_states)
        self.n_actions = 1
        self.env.reset()
        
    def state(self):
        theta, thetadot = self.env.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        # print(self.env.state)
        # return self.env.state

    def reset(self):
        return self.env.reset()
        
    def step(self, action, evaluate_mode=False):
        state = self.state()
        next_state, reward, done, _ = self.env.step(action)

        # if done and not evaluate_mode:
        #     reward = -reward

        return Experience(
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state
        )

    def name(self):
        return 'pendulum'
