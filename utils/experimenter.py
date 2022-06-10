import os
from dataclasses import dataclass
import numpy as np
from policy_based.template_policy import TemplatePolicy, TrainResults
from utils.Helper import LearningCurvePlot, smooth


@dataclass
class ExperimentResults:
  rewards_per_rep: np.ndarray
  eval_rewards_per_rep: np.ndarray


class Experimenter:

  DIR = 'figures'

  def __init__(self, repetitions, eval_repetitions, title, filename, max_epochs, smoothing_window=51):
    self.repetitions = repetitions
    self.eval_repetitions = eval_repetitions
    self.title = title
    self.filename = filename
    self.max_epochs = max_epochs
    self.smoothing_window = smoothing_window
    self.eval_smoothing_window = int(eval_repetitions / 5) + 1
    self.data_per_run = {}
    self.rewards_plot = LearningCurvePlot(title=title)
    self.eval_plot = LearningCurvePlot(title=title)
    self.rewards_plot_var = LearningCurvePlot(title=title)
    self.eval_plot_var = LearningCurvePlot(title=title)

    self.train_dir = os.path.join(self.DIR, 'default', 'train')
    self.eval_dir = os.path.join(self.DIR, 'default', 'eval')
    if not os.path.isdir(self.DIR):
      os.makedirs(self.DIR)

  def make_dirs(self, env_name):
    self.train_dir = os.path.join(self.DIR, env_name, 'train')
    self.eval_dir = os.path.join(self.DIR, env_name, 'eval')

    if not os.path.isdir(self.train_dir):
      os.makedirs(self.train_dir)
    if not os.path.isdir(self.eval_dir):
      os.makedirs(self.eval_dir)

  def run_all(self, policies):
    for policy in policies:
      self.run(policy)


  def run(self, policy:TemplatePolicy, demo_mode=False):
    print(policy.device)
    self.make_dirs(policy.env.name())

    rewards_per_rep = np.empty([self.repetitions, self.max_epochs])
    eval_rewards_per_rep = np.empty([self.repetitions, self.eval_repetitions])

    for rep in range(self.repetitions):
      policy.reset()
      results:TrainResults = policy.train(self.max_epochs)
      rewards_per_rep[rep] = results.rewards_per_epoch

      eval_rewards = []
      for _ in range(self.eval_repetitions):
        rewards = policy.evaluate(demo_mode)
        eval_rewards.append(rewards)

      eval_rewards_per_rep[rep] = np.array(eval_rewards)

    self.data_per_run[policy.name] = ExperimentResults(
      rewards_per_rep=rewards_per_rep,
      eval_rewards_per_rep=eval_rewards_per_rep
    )


  def gather_results(self):
    for label, results in self.data_per_run.items():
      rewards_data, rewards_std, rewards_min, rewards_max = self.get_plot_data_for(results.rewards_per_rep, window=self.smoothing_window)
      eval_rewards_data, eval_rewards_std, eval_rewards_min, eval_rewards_max = self.get_plot_data_for(results.eval_rewards_per_rep, window=self.eval_smoothing_window)
      
      self.rewards_plot.add_curve(rewards_data, rewards_std, label=label)
      self.eval_plot.add_curve(eval_rewards_data, eval_rewards_std, label=label)
      self.rewards_plot_var.add_curve(rewards_data, [rewards_data - rewards_min, rewards_max - rewards_data], label=label)
      # self.eval_plot_var.add_curve(eval_rewards_data, eval_rewards_var, label=label)

    self.rewards_plot.save(os.path.join(self.train_dir, self.filename), 'upper left')
    self.eval_plot.save(os.path.join(self.eval_dir, self.filename), 'upper left')
    self.rewards_plot_var.save(os.path.join(self.train_dir, self.filename + '_minmax'), 'upper left')
    # self.eval_plot_var.save(os.path.join(self.eval_dir, self.filename + '_var'), 'upper left')
    
  def demo(self, repetitions, policy:TemplatePolicy):
    for _ in repetitions:
        policy.evaluate(demo_mode=True)

  def get_plot_data_for(self, data_per_rep, window, needs_smoothing=True):
    data = np.mean(data_per_rep, axis=0)
    data_std = np.std(data_per_rep, axis=0)
    data_min = np.min(data_per_rep, axis=0)
    data_max = np.max(data_per_rep, axis=0)
    if needs_smoothing:
      data = smooth(data, window)
      data_std = smooth(data_std, window)
      data_min = smooth(data_min, window)
      data_max = smooth(data_max, window)
    
    # print(data_std)
    return data, data_std, data_min, data_max



