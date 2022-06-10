from run import LEARNING_RATE_BY_ENV, SMOOTHING_WINDOW, RunArgs, run_experiment, REPETITIONS, EVAL_REPETITIONS, MAX_EPOCHS_BY_ENV, M, STEP_SIZE_BY_ENV
from utils.experimenter import Experimenter

ENV = 'lunarlander'

if __name__ == '__main__':
    experimenter = Experimenter(
        2,
        EVAL_REPETITIONS,
        'Lunar Lander: Policies comparison',
        'policies',
        MAX_EPOCHS_BY_ENV[ENV],
        SMOOTHING_WINDOW
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
    
    experimenter.gather_results()