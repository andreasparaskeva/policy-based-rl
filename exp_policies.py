from run import LEARNING_RATE_BY_ENV, SMOOTHING_WINDOW, RunArgs, run_experiment, REPETITIONS, EVAL_REPETITIONS, MAX_EPOCHS_BY_ENV, M, STEP_SIZE_BY_ENV
from utils.experimenter import Experimenter

ENV = 'cartpole'

if __name__ == '__main__':
    experimenter = Experimenter(
        REPETITIONS,
        EVAL_REPETITIONS,
        'Cartpole: Policies comparison',
        'policies',
        MAX_EPOCHS_BY_ENV[ENV],
        SMOOTHING_WINDOW
    )
    
    run_experiment(
        args=RunArgs(
            policy='reinforce',
            env=ENV,
            lr=5e-3,
            bootstrap_steps=None,
            baseline=None,
            entropy=(0, 0),
            M=M,
            eval_reps=experimenter.eval_repetitions,
            reps=experimenter.repetitions,
            name='reinforce'
        ),
        title=experimenter.title,
        filename=experimenter.filename,
        experimenter=experimenter
    )
    
    run_experiment(
        args=RunArgs(
            policy='actor-critic',
            env=ENV,
            lr=LEARNING_RATE_BY_ENV[ENV],
            bootstrap_steps=1,
            baseline=False,
            entropy=(0, 0),
            M=M,
            eval_reps=experimenter.eval_repetitions,
            reps=experimenter.repetitions,
            name='actor-critic'
        ),
        title=experimenter.title,
        filename=experimenter.filename,
        experimenter=experimenter
    )
    
    run_experiment(
        args=RunArgs(
            policy='actor-critic',
            env=ENV,
            lr=LEARNING_RATE_BY_ENV[ENV],
            bootstrap_steps=STEP_SIZE_BY_ENV[ENV],
            baseline=False,
            entropy=(0, 0),
            M=M,
            eval_reps=experimenter.eval_repetitions,
            reps=experimenter.repetitions,
            name='actor-critic + bootstrap'
        ),
        title=experimenter.title,
        filename=experimenter.filename,
        experimenter=experimenter
    )
    
    run_experiment(
        args=RunArgs(
            policy='actor-critic',
            env=ENV,
            lr=LEARNING_RATE_BY_ENV[ENV],
            bootstrap_steps=1,
            baseline=True,
            entropy=(0, 0),
            M=M,
            eval_reps=experimenter.eval_repetitions,
            reps=experimenter.repetitions,
            name='actor-critic + baseline subtraction'
        ),
        title=experimenter.title,
        filename=experimenter.filename,
        experimenter=experimenter
    )
    
    run_experiment(
        args=RunArgs(
            policy='actor-critic',
            env=ENV,
            lr=LEARNING_RATE_BY_ENV[ENV],
            bootstrap_steps=STEP_SIZE_BY_ENV[ENV],
            baseline=True,
            entropy=(0, 0),
            M=M,
            eval_reps=experimenter.eval_repetitions,
            reps=experimenter.repetitions,
            name='actor-critic + bootstrap + baseline subtraction'
        ),
        title=experimenter.title,
        filename=experimenter.filename,
        experimenter=experimenter
    )
    experimenter.gather_results()