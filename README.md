## Policy-based RL

With the following command and parameters, it is possible to run the tuned models of policy-based RL:

~~~
python run.py --policy [reinforce, actor-critic] --env [cartpole, lunarlander, lunarlander-continuous] -bootstrap -baseline -entropy --reps <REPS> --evalReps <EVAL-REPS> --figureTitle <TITLE> --filename <FILENAME>
~~~

where `-bootstrap`, `-baseline` and `-entropy` are optional flags to activate or deactivate the respective features.


## CMA-ES
With the following command and parameters, it is possible to run the tuned model of CMA-ES method:

~~~
python run_cma_es.py --env [cartpole, lunarlander]--reps <REPS> --evalReps <EVAL-REPS> --figureTitle <TITLE> --filename <FILENAME>
~~~

## Experiments

To run the experiments presented in the report, use the following commands.

### Learning rate tuning
~~~
python exp_learning_rate.py
~~~

### M size tuning
~~~
python exp_m_size.py
~~~

### Bootstrap step tuning
~~~
python exp_bootstrap_step.py
~~~

### Ablation study (reinforce, actor-critic, bootstrap, baseline subtraction)
~~~
python exp_policies.py
~~~

### Entropy regularization
~~~
python exp_entropy.py
~~~

### Continuous space
~~~
python exp_continuous_policies.py
~~~

### CMA-ES
~~~
python exp_cma_alpha_beta.py
~~~