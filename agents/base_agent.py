import numpy as np
import torch as T
from replay_memory import ReplayBuffer


class BaseAgent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 memory_size, batch_size, eps_min=0.01, decay_rate=5e-7,
                 replace=1000, algo=None, env_name=None, checkpoint_dir="tmp/dqn") -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.decay_rate = decay_rate
        self.replace_target_count = replace
        self.algo = algo
        self.env_name = env_name
        self.checkpoint_dir = checkpoint_dir
        self.action_space = list(range(self.n_actions))
        self.learn_step_counter = 0 # C steps

        self.memory = ReplayBuffer(self.memory_size, self.input_dims)
        
    def choose_action(self, observation):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.decay_rate if self.epsilon > self.eps_min else self.eps_min
    
    def store_transition(self, state, action, reward, done, next_state):
        self.memory.store_transition(state, action, reward, done, next_state)

    def sample_memory(self):
        state, action, reward, done, next_state = self.memory.sample_buffer(self.batch_size)
    
        states = T.tensor(state).to(self.Q_eval.device)
        actions = T.tensor(action).to(self.Q_eval.device)
        rewards = T.tensor(reward).to(self.Q_eval.device)
        dones = T.tensor(done).to(self.Q_eval.device)
        next_states = T.tensor(next_state).to(self.Q_eval.device)

        return states, actions, rewards, dones, next_states
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_count == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
    
    def save_models(self):
        self.Q_eval.save_checkpoint()
        self.Q_next.save_checkpoint()
    
    def load_models(self):
        self.Q_eval.load_checkpoint()
        self.Q_next.load_checkpoint()
    
    def learn(self):
        raise NotImplementedError("Subclass must implement abstract method")