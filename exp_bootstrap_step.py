from run import LEARNING_RATE_BY_ENV, SMOOTHING_WINDOW, RunArgs, run_experiment, REPETITIONS, EVAL_REPETITIONS, MAX_EPOCHS_BY_ENV, M, STEP_SIZE_BY_ENV
from utils.experimenter import Experimenter

ENV = 'cartpole'

if __name__ == '__main__':
    experimenter = Experimenter(
        REPETITIONS,
        EVAL_REPETITIONS,
        'Cartpole Actor Critic: bootsrap step size tuning',
        'bootstrap_step',
        MAX_EPOCHS_BY_ENV[ENV],
        SMOOTHING_WINDOW
    )
    
    for bootsrap_steps in [1, 5, 20, 100, 300, 500]:
        run_experiment(
            args=RunArgs(
                policy='actor-critic',
                env=ENV,
                lr=LEARNING_RATE_BY_ENV[ENV],
                bootstrap_steps=bootsrap_steps,
                baseline=True,
                entropy=(0, 0),
                M=M,
                eval_reps=experimenter.eval_repetitions,
                reps=experimenter.repetitions,
                name=bootsrap_steps
            ),
            title=experimenter.title,
            filename=experimenter.filename,
            experimenter=experimenter
        )
    experimenter.gather_results()