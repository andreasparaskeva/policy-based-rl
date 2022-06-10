import argparse
import sys
from dataclasses import dataclass
from environments.pendulum import Pendulum
from policy_based.actor_critic_policy import ActorCritic
from functools import partial
from policy_based.reinforce_policy import ReinforcePolicy
from utils.experimenter import Experimenter
from environments.cartpole_state import CartPoleEnv
from environments.lunarlander import LunarLanderEnv
from models.actor_critic import ActorCriticModel
from models.actor_critic_continuous import ActorCriticContinuousModel
from models.reinforce import ReinforceModel

MAX_EPOCHS = 2_000
SMOOTHING_WINDOW = 501
REPETITIONS = 5
EVAL_REPETITIONS = 100
GAMMA = 0.99
M = 1
POLICIES = ['reinforce', 'actor-critic', 'cma-es']
ENVIRONMENTS = ['cartpole', 'lunarlander', 'lunarlander-continuous', 'pendulum']

ENV_BY_NAME = {
    'cartpole': CartPoleEnv(),
    'lunarlander': LunarLanderEnv(continuous=False),
    'lunarlander-continuous': LunarLanderEnv(continuous=True),
    'pendulum': Pendulum(),
}

MAX_EPOCHS_BY_ENV = {
    'cartpole': 1_500,
    'lunarlander': 2_000,
    'lunarlander-continuous': 2_000,
    'pendulum': 10_000
}

POLICY_BY_NAME = {
    'reinforce': partial(ReinforcePolicy),
    'actor-critic': partial(ActorCritic),
}

MODEL_BY_ENV = {
    'cartpole': partial(ActorCriticModel, hidden_sizes=[128]),
    'lunarlander': partial(ActorCriticModel, hidden_sizes=[128, 128]),
    'lunarlander-continuous': partial(ActorCriticContinuousModel, hidden_sizes=[128, 128]),
    'pendulum': partial(ActorCriticContinuousModel, hidden_sizes=[128, 64]),
}

LEARNING_RATE_BY_ENV = {
    'cartpole': 0.01,
    'lunarlander': 1e-3,
    'lunarlander-continuous': 5e-3,
    'pendulum': 1e-4
}

STEP_SIZE_BY_ENV = {
    'cartpole': 300,
    'lunarlander': 1000,
    'lunarlander-continuous': 400,
    'pendulum': 500
}

ENTROPY_BY_ENV = {
    'cartpole': (0.2, 0.99),
}

def parsed_args():
    parser = argparse.ArgumentParser(description='Deep Q Network')
    parser.add_argument('--policy', help='Policy-based method', choices=POLICIES, default='actor-critic')
    parser.add_argument('--env', help='Environemnt', choices=ENVIRONMENTS, default='cartpole')
    parser.add_argument('-bootstrap', action='store_true')
    parser.add_argument('-baseline', action='store_true')
    parser.add_argument('-entropy', action='store_true')    
    parser.add_argument('--reps', help='Repetitions of each experimen', type=int, default=REPETITIONS)
    parser.add_argument('--evalReps', help='Evaluation repetitions after training', type=int, default=EVAL_REPETITIONS)
    parser.add_argument('--figureTitle', help='Name for the title of the figure', type=str, default='')
    parser.add_argument('--filename', help='Name for the file name of the figure', type=str, default='')
    return parser.parse_args()


@dataclass
class RunArgs:
    policy: str
    env: str
    lr: float
    M:int
    bootstrap_steps: int
    baseline: bool
    entropy: tuple
    reps: int
    eval_reps: int
    name: str
    
def run_experiment(
    args: RunArgs,
    title,
    filename,
    experimenter=None,
):
    
    env = ENV_BY_NAME[args.env]
    if args.policy == 'reinforce':
        model = ReinforceModel(input_size=env.n_states, n_actions=env.n_actions, hidden_size=[64, 8])
        lr = 5e-3
        max_epochs = 1_500
    else:
        max_epochs=MAX_EPOCHS_BY_ENV[args.env]
        lr = LEARNING_RATE_BY_ENV[args.env]
        model = MODEL_BY_ENV[args.env](input_size=env.n_states, n_actions=env.n_actions)
        
    if experimenter is None:
        experimenter = Experimenter(
            repetitions=args.reps, 
            eval_repetitions=args.eval_reps,
            title=title,
            filename=filename,
            max_epochs=max_epochs,
            smoothing_window=SMOOTHING_WINDOW,
        )
    
    policy = POLICY_BY_NAME[args.policy](
        env=env,
        model=model,
        lr=args.lr,
        gamma=GAMMA,
        step_size=args.bootstrap_steps,
        baseline=args.baseline,
        entropy=args.entropy,
        name=args.name,
        M=args.M
    )
    
    experimenter.run(policy)
    return experimenter
    
    
if __name__ == '__main__':
    args = parsed_args()
    
    step_size = STEP_SIZE_BY_ENV[args.env] if args.bootstrap else sys.maxsize
    entropy = ENTROPY_BY_ENV[args.env] if args.entropy else (0.0, 0.0)
    lr = LEARNING_RATE_BY_ENV[args.env]
    
    run_args = RunArgs(
        policy=args.policy,
        lr=lr,
        M=M,
        env=args.env,
        bootstrap_steps=step_size,
        baseline=args.baseline,
        entropy=entropy,
        reps=args.reps,
        eval_reps=args.evalReps,
        name=args.policy
    )
    
    experimenter = run_experiment(run_args, args.figureTitle, args.filename)
    experimenter.gather_results()
    