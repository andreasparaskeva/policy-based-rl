from run import LEARNING_RATE_BY_ENV, SMOOTHING_WINDOW, RunArgs, run_experiment, REPETITIONS, EVAL_REPETITIONS, MAX_EPOCHS_BY_ENV, M, STEP_SIZE_BY_ENV
from utils.experimenter import Experimenter

ENV = 'cartpole'

if __name__ == '__main__':
    experimenter = Experimenter(
        REPETITIONS,
        EVAL_REPETITIONS,
        'Actor Critic on cartpole: Number of traces tuning',
        'number_of_traces',
        MAX_EPOCHS_BY_ENV[ENV],
        SMOOTHING_WINDOW
    )
    
    for m in [1, 2, 5]:
        run_experiment(
            args=RunArgs(
                policy='actor-critic',
                env=ENV,
                lr=LEARNING_RATE_BY_ENV[ENV],
                bootstrap_steps=STEP_SIZE_BY_ENV[ENV],
                baseline=True,
                entropy=(0, 0),
                M=m,
                eval_reps=experimenter.eval_repetitions,
                reps=experimenter.repetitions,
                name=m
            ),
            title=experimenter.title,
            filename=experimenter.filename,
            experimenter=experimenter
        )
    experimenter.gather_results()