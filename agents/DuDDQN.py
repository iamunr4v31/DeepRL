import numpy as np
import torch as T
from agents.DQNAgent import DQNAgent

class DuDDQNAgent(DQNAgent):
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation]), dtype=T.float).to(self.Q_eval.device) # batch_size * input_dims
            advantage = self.Q_eval.forward(state)[1]
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action

    def learn(self):
        if self.memory.memory_counter >= self.batch_size:
            self.Q_eval.optimizer.zero_grad()
            self.replace_target_network()
            states, actions, rewards, dones, next_states = self.sample_memory()
            
            indices = np.arange(self.batch_size)

            value_s, advantage_s = self.Q_eval.forward(states)
            value_s_next, advantage_s_next = self.Q_next.forward(next_states)

            value_s_eval, advantage_s_eval = self.Q_eval.forward(next_states)

            q_pred = T.add(value_s, 
                          (advantage_s - advantage_s.mean(dim=1, keepdim=True)))[indices, actions]
            q_next = T.add(value_s_next, 
                          (advantage_s_next - advantage_s_next.mean(dim=1, keepdim=True)))
            
            q_eval = T.add(value_s_eval,
                          (advantage_s_eval - advantage_s_eval.mean(dim=1, keepdim=True)))

            max_actions = T.argmax(q_eval, dim=1)
            
            q_next[dones] = 0.0
            q_target = rewards + self.gamma * q_next[indices, max_actions]

            loss = self.Q_eval.criterion(q_target, q_pred).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()
            self.learn_step_counter+=1

            self.decrement_epsilon()