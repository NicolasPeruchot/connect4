import numpy as np

from tqdm import tqdm

from base_model import BaseQLearningModel


class Qlearning(BaseQLearningModel):
    def __init__(
        self,
        initial_exploration_factor=0.5,
        final_exploration_factor=0.1,
        discount_factor=0.7,
        learning_rate=0.1,
    ) -> None:
        super().__init__(
            initial_exploration_factor,
            final_exploration_factor,
            discount_factor,
            learning_rate,
        )
        self.q_table = {}

    def __get_q_table_key(self, state):
        key = "".join([str(x) for x in state.flatten()])
        if key not in self.q_table.keys():
            self.q_table[key] = [0 for _ in range(7)]
        return key

    def training(self, n_training_game=1000):
        self.initialize_game(render=None)
        for game in tqdm(range(n_training_game)):
            self.env.reset()
            self.reset_agents()
            end = False

            i = 0
            while end is False:
                end = self.play_game(i, game, n_training_game)

                self.update_policy(end, i)

                i += 1

            self.env.close()

    def update_policy(self, end, i):
        if end:
            if self.agents[i % 2]["reward"] == 1:
                self.agents[(i + 1) % 2]["reward"] = -1

            for j in [0, 1]:
                self.update_q_table(j)
            # Update stats
            self.update_stats(winner=self.agents[i % 2]["name"], nb_moves_to_win=i)

        elif self.agents[(i + 1) % 2]["last_state"] is not None:
            self.update_q_table((i + 1) % 2)

    def update_q_table(self, i):
        action = self.agents[i]["last_action"]
        last_state = self.agents[i]["last_state"]
        current_state = self.agents[i]["current_state"]

        old_value = self.q_table[self.__get_q_table_key(last_state)][action]
        next_max = np.max(self.q_table[self.__get_q_table_key(current_state)])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
            self.agents[i]["reward"] + self.discount_factor * next_max
        )
        self.q_table[self.__get_q_table_key(self.agents[i]["last_state"])][action] = new_value

    def get_action(self, state):
        key = self.__get_q_table_key(state["observation"])
        possible = [
            self.q_table[key][i] if state["action_mask"][i] != 0 else -np.inf for i in range(7)
        ]
        action = np.argmax(possible)
        return action
