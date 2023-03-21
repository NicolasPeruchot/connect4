import numpy as np

class QAgent:
    def __init__(self, state_size, action_size, learning_rate=0.8, discount_factor=0.95, exploration_rate=0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            # Explore action space
            return np.random.choice(self.action_size)
        else:
            # Exploit learned values
            return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state, :])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state, action] = new_q


