import numpy as np
from typing import Tuple, Optional, Union

class ReplayBuffer:
    def __init__(self, max_size: int, input_shape: Union[int, Tuple[int]]) -> None:
        """Initialize Replay Buffer

        Args:
            max_size (int): Maximum Size of the Buffer
            input_shape (Union[int, Tuple[int]]): shape of the Buffer
            n_actions (Optional[int], optional): N actions. Defaults to None.
        """        
        self.memory_size = max_size
        self.memory_counter = 0

        self.state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int64) #change to 64 bit if it doesnt work
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, done, next_state):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.new_state_memory[index] = next_state
        self.memory_counter+=1

    def sample_buffer(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, dones, next_states