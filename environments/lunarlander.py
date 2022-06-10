import gym
import torch
from environments.environment import Environment, Experience
import matplotlib.pyplot as plt


class LunarLanderEnv(Environment):

    def __init__(self, continuous=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.continuous = continuous
        self.env = gym.make("LunarLander-v2", continuous=self.continuous)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n if not continuous else 2
        self.max_reward = 200
        self.env.reset()
        self.curr_state = self.env.reset()

    def render(self):
        imgplot = plt.imshow(self.current_screen)
        plt.show()
        
    def state(self):
        return self.curr_state

    def reset(self):
        return self.env.reset()
        
    def step(self, action, evaluate_mode=False):
        state = self.state()
        next_state, reward, done, _ = self.env.step(action)

        self.curr_state = next_state

        return Experience(
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state
        )

    def name(self):
        return 'lunar-lander'
