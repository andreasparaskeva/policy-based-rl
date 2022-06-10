import argparse
from dataclasses import dataclass
from policy_based.cma_es_policy import CMA_ES
from models.cma_es import CMA_ES_Model
from run import MAX_EPOCHS_BY_ENV
from utils.experimenter import Experimenter
from environments.cartpole_state import CartPoleEnv
from environments.lunarlander import LunarLanderEnv

MAX_EPOCHS = 2_500
SMOOTHING_WINDOW = 69
REPETITIONS = 1
EVAL_REPETITIONS = 100
ENVIRONMENTS = ['cartpole', 'lunarlander']

ALPHA_BY_ENV = {
  'cartpole': 0.0,
  'lunarlander': 0.01,
}

BETA_BY_ENV = {
  'cartpole': 0.9,
  'lunarlander': 1.7,
}

MAX_EPOCHS_BY_ENV = {
    'lunarlander': 2_000,
    'cartpole': 2_000
}

ENV_BY_NAME = {
    'cartpole': CartPoleEnv(),
    'lunarlander': LunarLanderEnv(continuous=False),
    'lunarlander-continuous': LunarLanderEnv(continuous=True),
}

@dataclass
class RunArgs:
    env: str
    reps: int
    eval_reps: int
    name: str

def parsed_args():
    parser = argparse.ArgumentParser(description='Policy based TD')
    parser.add_argument('--env', help='Environemnt', choices=ENVIRONMENTS, default='cartpole')
    parser.add_argument('--reps', help='Repetitions of each experimen', type=int, default=REPETITIONS)
    parser.add_argument('--evalReps', help='Evaluation repetitions after training', type=int, default=EVAL_REPETITIONS)
    parser.add_argument('--figureTitle', help='Name for the title of the figure', type=str, default='')
    parser.add_argument('--filename', help='Name for the file name of the figure', type=str, default='')
    return parser.parse_args()

      
def run_experiment(
    args: RunArgs,
    title,
    filename,
    experimenter=None,
):

    if experimenter is None:
        experimenter = Experimenter(
            repetitions=args.reps, 
            eval_repetitions=args.eval_reps,
            title=title,
            filename=filename,
            max_epochs=MAX_EPOCHS,
            smoothing_window=SMOOTHING_WINDOW,
        )
    
    env = ENV_BY_NAME[args.env]
    n_states = env.env.observation_space.shape[0]
    n_actions = env.env.action_space.n
    model = CMA_ES_Model(n_states=n_states, n_actions=n_actions)
    alpha = ALPHA_BY_ENV[args.env]
    beta = BETA_BY_ENV[args.env]
    policy = CMA_ES(env, model, alpha, beta, name=args.name)
    
    experimenter.run(policy)
    return experimenter
    
    


if __name__ == '__main__':
  args = parsed_args()
  run_args = RunArgs(
      env=args.env,
      reps=args.reps,
      eval_reps=args.evalReps,
      name='cma-es'

  )
  
  experimenter = run_experiment(run_args, args.figureTitle, args.filename)
  experimenter.gather_results()
