from run_cma_es import SMOOTHING_WINDOW, RunArgs, run_experiment, REPETITIONS, EVAL_REPETITIONS, MAX_EPOCHS_BY_ENV
from utils.experimenter import Experimenter

ENVS = ['cartpole']# 'cartpole', 'lunarlander'
ENV_NAME = {
  'cartpole': 'CartPole',
  'lunarlander': 'LunarLander'
}

if __name__ == '__main__':
    for ENV in ENVS:
      experimenter = Experimenter(
          REPETITIONS,
          EVAL_REPETITIONS,
          f'CMA-ES (alpha, beta) tuning on {ENV_NAME[ENV]}',
          f'cma-es-alpha-beta-{ENV}',
          MAX_EPOCHS_BY_ENV[ENV],
          SMOOTHING_WINDOW
      )
      for alpha, beta in [(0.0, 1.0), (0.01, 1.3), (0.1, 1.7)]:
          run_experiment(
              args=RunArgs(
                  env=ENV,
                  eval_reps=experimenter.eval_repetitions,
                  reps=experimenter.repetitions,
                  name=f'({alpha},{beta})'
              ),
              title=experimenter.title,
              filename=experimenter.filename,
              experimenter=experimenter
          )
      experimenter.gather_results()