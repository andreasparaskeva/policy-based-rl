import numpy as np
from utils.normalizer import Normalizer

class CMA_ES_Model():
    """
    Implementation the cma_es 'model'
    """
    def __init__(self, n_states, n_actions):
        super(CMA_ES_Model, self).__init__()
        self._n_states = n_states
        self._n_actions = n_actions
        self._dims = (n_states, n_actions)
        self._normalizer = Normalizer(n_states)

 
    def select_action(self, state, theta):
      # Normalize observation
      self._normalizer.observe(state)
      state = self._normalizer.normalize(state)

      # Linear layer (without bias)
      x = np.dot(state, np.reshape(theta, self._dims))

      # Argmax with random tie-breaking to get action
      return np.random.choice(np.flatnonzero(x == x.max()))

    def select_greedy_action(self, state, theta):
      return self.select_action(state, theta)
        
    def name(self):
        return f'cma-es'

    def reset(self):
        return CMA_ES_Model(self._n_states, self._n_actions)
