from agents.DQNAgent import DQNAgent
import numpy as np
import torch as T

class DDQNAgent(DQNAgent):
    def learn(self):
        if self.memory.memory_counter >= self.batch_size:
            self.Q_eval.optimizer.zero_grad()
            self.replace_target_network()
            states, actions, rewards, dones, next_states = self.sample_memory()
            
            indices = np.arange(self.batch_size)
            q_pred = self.Q_eval.forward(states)[indices, actions]
            q_next = self.Q_next.forward(next_states)
            q_eval = self.Q_eval.forward(next_states)

            max_actions = T.argmax(q_eval, dim=1)

            q_next[dones] = 0.0
            q_target = rewards + self.gamma * q_next[indices, max_actions]
            
            loss = self.Q_eval.criterion(q_target, q_pred).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()
            self.learn_step_counter+=1

            self.decrement_epsilon()