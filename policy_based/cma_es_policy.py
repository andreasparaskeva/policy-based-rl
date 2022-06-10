import math
import numpy as np
from tqdm import tqdm
from typing import Tuple, List
from collections import deque
from models.cma_es import CMA_ES_Model
from policy_based.template_policy import TrainResults
from environments.environment import Environment, Experience


class CMA_ES():
  def __init__(
    self,
    env: Environment,
    model: CMA_ES_Model,
    alpha,
    beta,
    name
  ):
    self.env = env
    self.model = model
    self._alpha = alpha
    self._beta = beta
    self.name = name
    self.device = None
  # mu = 0.0
    self._mu = np.random.uniform(alpha, beta) 
    self._sigma =  0.3 * (beta-alpha)
    self.trend = deque([], 100)

    n_states = self.env.env.observation_space.shape[0]
    n_actions = self.env.env.action_space.n
    n_dim = n_actions*n_states
    pop_size = 4 + math.floor(3*math.log(n_dim))
    self._pop_size = pop_size
    self._n_dim = n_dim
    
    self._C = np.identity(n_dim)
    self._pc = 0
    self._psigma = 0
    self.g = 0 # generation counter
    self._mean = np.full((n_dim), self._mu)

    self._EPS = 1e-8
    self._SIGMA_THRESHOLD = 1e32
    self._eigen_decomposition()
    self._init_weights()
    self._B = None
    self._D = None


  def _init_weights(self):
    mu = math.floor(self._pop_size/2)

    # equation 49 from paper
    weights_prime = np.array(
        [
            math.log((self._pop_size + 1) / 2) - math.log(i)
            for i in range(1, self._pop_size + 1)
        ]
    )
    # weights_prime up to mu value 
    mu_eff = (np.sum(weights_prime[:mu])**2) / np.sum(weights_prime[:mu]**2)
    # weights prime from mu till end
    mu_eff_minus = (np.sum(weights_prime[mu:])**2) / np.sum(weights_prime[mu:]**2)

    alpha_cov = 2

    # Equration 56 from paper 
    cc = (4 + mu_eff/self._n_dim)/(self._n_dim + 4 + 2*mu_eff/self._n_dim)
    # Equation 57 from paper
    c1 = alpha_cov/((self._n_dim + 1.3) ** 2 + mu_eff)
    # Equation 58 from paper
    c_mu = min(1 - c1,
              alpha_cov * ((mu_eff - 2 + 1/mu_eff)/
                ((self._n_dim + 2)**2) + alpha_cov*mu_eff/2)
            )
    # Equation 50 from paper
    alpha_mu_minus =  1 + c1/c_mu  
    # Equation 51 from paper
    alpha_mu_eff_minus = 1 + (2*mu_eff_minus)/(mu_eff+2) 
    # Equation 52 from paper
    alpha_pos_def_minus = (1 -c1 - c_mu)/(self._n_dim*c_mu)  

    # Calculate minimum alpha (part of weights)
    min_alpha = min(alpha_mu_minus, alpha_mu_eff_minus, alpha_pos_def_minus)

    # Equation 53 from paper 
    positive_weights_sum = 1/(np.sum(weights_prime[weights_prime>0]))
    negative_weights_sum = min_alpha/(np.sum(weights_prime[weights_prime<0]))
    weights = np.where( weights_prime >= 0,
                        positive_weights_sum * weights_prime,
                        negative_weights_sum * weights_prime
                      )

    # Equation 54 from paper
    cm = 1

    # Equation 55 from paper
    c_sigma = (mu_eff + 2)/(self._n_dim + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff-1)/self._n_dim+1)-1) + c_sigma
    # Save to object
    self._mu = mu
    self._mu_eff = mu_eff
    self._cc = cc
    self._c1 = c1
    self._c_mu = c_mu
    self._c_sigma = c_sigma
    self._d_sigma = d_sigma
    self._cm = cm
    self._weights = weights
    # evolution paths
    self._p_sigma = np.zeros(self._n_dim)
    self._pc = np.zeros(self._n_dim)

  def _eigen_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
    self._C = (self._C + self._C.T) / 2
    D2, B = np.linalg.eigh(self._C)
    D = np.sqrt(np.where(D2 < 0, self._EPS, D2))
    self._C = np.dot(np.dot(B, np.diag(D ** 2)), B.T)
    self._B, self._D = B, D

  def _getBD(self):
    return self._B, self._D

  def sample_pop(self):
    # self._eigen_decomposition()
    # ~ N(m, σ^2 C)
    pop = [np.random.multivariate_normal(mean=self._mean, cov=(self._sigma**2 * self._C)) 
              for i in range(self._pop_size)]
    return pop

  def cma_procedure(self, solutions, fitnesses):
    self.g += 1
    solutions_sorted = [solutions[i[0]] for i in sorted(enumerate(fitnesses), key=lambda x:x[1])]
    # ___________ Sample new population ___________
    self._eigen_decomposition()
    B, D = self._getBD()

    x_k = np.array([s for s in solutions_sorted])  
    y_k = (x_k - self._mean) / self._sigma  

    # ___________ Selection and recombination ___________
    # Equtation 41 from paper
    yw = np.sum(y_k[: self._mu].T * self._weights[: self._mu], axis=1)  
    self._mean += self._cm * self._sigma * yw

    # ___________ Step size control ___________
    # C^(-1/2) = B D^(-1) B^T
    # Note: D can be inverted by inverting its diagonal elements
    eps = 1e-64

    C_1_2 = np.array(B.dot(np.diag(1/(D+eps))).dot(B.T))
    # Equation 43 from paper
    self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(
                      self._c_sigma * (2 - self._c_sigma) * self._mu_eff
                    ) * C_1_2.dot(yw)

    # E||N(0, I)|| From Paper this can be approximated as follows (p.28 of paper)
    e = math.sqrt(self._n_dim) * (
        1.0 - (1.0 / (4.0 * self._n_dim)) + 1.0 / (21.0 * (self._n_dim ** 2))
    )

    # Equation 44 from paper
    self._sigma *= np.exp(
        (self._c_sigma / self._d_sigma) * (np.linalg.norm(self._p_sigma) / e - 1)
    )
    # TODO: check later if needed
    self._sigma = min(self._sigma, self._SIGMA_THRESHOLD)


    # ___________ Covariance Matrix Adaptation ___________
    # Look at page 28 of paper for how to calculate h_sigma 
    left_condition = np.linalg.norm(self._p_sigma) /\
                              math.sqrt(
                                1 - (1 - self._c_sigma)**(2*(self.g+1))
                              )
    right_condition = (1.4 + 2/(self._n_dim+1)) * e
    h_sigma = 1.0 if left_condition < right_condition else 0.0
    # Equation 45 from paper
    self._pc = (1-self._cc)*self._pc + h_sigma * math.sqrt(self._cc*(2-self._cc)*self._mu_eff)*yw
    # Equation 46 from paper
    w_i_zero = self._weights * np.where(
            self._weights >= 0,
            1,
            self._n_dim / (np.linalg.norm(C_1_2.dot(y_k.T), axis=0) ** 2 + self._EPS),
        )
    # Equation 47 from paper
    # calculate δ(hσ) from page 28
    delta_h_sigma = (1 - h_sigma) * self._cc * (2 - self._cc) 
    rank_mu = np.sum(
            np.array([w * np.outer(y, y).transpose() for w, y in zip(w_i_zero, y_k)]), axis=0
        )
    self._C = (1 + self._c1*delta_h_sigma - self._c1 - self._c_mu*np.sum(self._weights))*self._C \
      +  self._c1 * np.outer(self._pc,self._pc).transpose() + self._c_mu * rank_mu

  def reset(self):
    self.model = self.model.reset()

  def evaluate(self, demo_mode=False):
    if demo_mode:
      self.env.enable_demo_mode()

    done = False
    rewards = 0
    self.env.reset()
    while not done:
      action = self.model.select_greedy_action(state=self.env.state(), theta=self._mean)
      experience = self.env.step(action, evaluate_mode=True)
      done = experience.done
      rewards += experience.reward
    # print(rewards)
    return rewards

  def run_episode(self, theta):
    self.env.reset()
    rewards = 0
    done = False
    while not done:
        action = self.model.select_action(state=self.env.state(), theta=theta)
        experience = self.env.step(action, evaluate_mode=True)
        done = experience.done
        rewards += experience.reward
        if done:
            break

    return rewards

  def simulate_remaining(self, index, max_epochs, gen_rewards: List[float]):
    print('Training Finished')
    for _ in tqdm(range(index, max_epochs)):
      gen_reward = 0
      for _ in range(self._pop_size):
        r = self.evaluate()
        gen_reward += r
      gen_rewards.append(gen_reward/self._pop_size)


  def train(self, max_epochs):
    ep = 1
    gen_rewards = []
    for i in tqdm(range(max_epochs)):
      population = self.sample_pop()
      fitnesses = []
      gen_reward = 0
      for individual in population:
          ep +=1  
          r = self.run_episode(individual)
          # Negate r because CMA-ES minimizes
          fitnesses.append(-r)
          self.trend.append(r)
          gen_reward += r
      
      if np.mean(self.trend) > (self.env.max_reward-1):
        self.simulate_remaining(i, max_epochs, gen_rewards)
        break
      if self.g % 10 == 0:
        print(f'Current Generation: {self.g}')
        print(ep, np.mean(self.trend))
      gen_rewards.append(gen_reward/self._pop_size)

      self.cma_procedure(population, fitnesses)

    return TrainResults(
      rewards_per_epoch=gen_rewards,
    )