from run import SMOOTHING_WINDOW, RunArgs, run_experiment, REPETITIONS, EVAL_REPETITIONS, MAX_EPOCHS_BY_ENV, M, STEP_SIZE_BY_ENV
from utils.experimenter import Experimenter

ENV = 'cartpole'
if __name__ == '__main__':
    experimenter = Experimenter(
        REPETITIONS,
        EVAL_REPETITIONS,
        'Cartpole: learning rate tuning',
        'learning_rate',
        MAX_EPOCHS_BY_ENV[ENV],
        SMOOTHING_WINDOW
    )
    
    for lr in [1e-2, 5e-3, 1e-3, 1e-4]:
        run_experiment(
            args=RunArgs(
                policy='actor-critic',
                env=ENV,
                lr=lr,
                bootstrap_steps=STEP_SIZE_BY_ENV[ENV],
                baseline=True,
                entropy=(0, 0),
                M=M,
                eval_reps=experimenter.eval_repetitions,
                reps=experimenter.repetitions,
                name=lr
            ),
            title=experimenter.title,
            filename=experimenter.filename,
            experimenter=experimenter
        )
    experimenter.gather_results()