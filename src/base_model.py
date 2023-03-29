import random
import time

import matplotlib.pyplot as plt
import numpy as np

from pettingzoo.classic import connect_four_v3


class BaseQLearningModel:
    def __init__(
        self,
        initial_exploration_factor=0.5,
        final_exploration_factor=0.1,
        discount_factor=0.7,
        learning_rate=0.1,
    ) -> None:
        self.initial_exploration_factor = initial_exploration_factor
        self.final_exploration_factor = final_exploration_factor
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.q_table = {}

    def get_action(self):
        raise NotImplementedError("The method not implemented")

    def update_policy(self):
        raise NotImplementedError("The method not implemented")

    def get_exploration_factor(self, step, n_training_game):
        tau = -(n_training_game) / (
            np.log(self.final_exploration_factor / self.initial_exploration_factor)
        )

        return self.initial_exploration_factor * np.exp(-step / tau)

    def play_game(self, i, game_number, n_total):
        current_agent = self.agents[i % 2]["name"]
        self.env.agent_selection = current_agent
        state = self.env.observe(current_agent)
        self.agents[i % 2]["last_state"] = state["observation"]

        if random.uniform(0, 1) < self.get_exploration_factor(
            step=game_number, n_training_game=n_total
        ):
            action = self.env.action_space(current_agent).sample(state["action_mask"])
        else:
            action = self.get_action(state)

        self.agents[i % 2]["last_action"] = action
        self.env.step(action)
        state, reward, termination, truncation, info = self.env.last()

        self.agents[i % 2]["reward"] = reward
        self.agents[i % 2]["current_state"] = state["observation"]

        end = termination or truncation
        return end

    def reset_agents(self):
        self.agents = {
            0: {
                "name": "player_0",
                "last_state": None,
                "current_state": None,
                "reward": 0,
                "last_action": None,
            },
            1: {
                "name": "player_1",
                "last_state": None,
                "current_state": None,
                "reward": 0,
                "last_action": None,
            },
        }
        return None

    def initialize_game(self, render=None):
        self.env = connect_four_v3.env(render_mode=render)
        self.stats = {"winner": [], "nb_moves_to_win": [], "exploration factor": []}
        self.reset_agents()
        self.env.reset()
        return None

    def update_stats(self, winner, nb_moves_to_win, game, n_training_game):
        if nb_moves_to_win != 6 * 7:
            self.stats["winner"].append(winner)
        else:
            self.stats["winner"].append(None)
        self.stats["nb_moves_to_win"].append(nb_moves_to_win)
        self.stats["exploration factor"].append(self.get_exploration_factor(game, n_training_game))

    def play(self):
        self.initialize_game(render="human")
        self.reset_agents()
        end = False
        i = 0
        while end is False:
            current_agent = self.agents[i % 2]["name"]
            self.env.agent_selection = current_agent
            state = self.env.observe(current_agent)
            action = self.get_action(state)
            self.env.step(action)
            state, reward, termination, truncation, info = self.env.last()
            end = termination or truncation
            i += 1
            time.sleep(0.3)
        self.env.close()

    def plot_training_stats(self):
        # initial stats
        winner = np.array(self.stats["winner"])
        nb_moves_to_win = np.array(self.stats["nb_moves_to_win"])
        explo_factor = np.array(self.stats["exploration factor"])
        mobile_mean_nb_moves_to_win = np.convolve(nb_moves_to_win, np.ones(10), "valid") / 10
        N = len(winner)

        # compute advanced stats
        player_0_is_winner = winner == "player_0"
        percentage_win_player_0 = np.zeros(N)
        for idx in range(N):
            percentage_win_player_0[idx] = np.sum(player_0_is_winner[: idx + 1]) / (idx + 1)

        # plot stats
        fig, axs = plt.subplots(4, 1, figsize=(10, 20))

        axs[0].set_title("Winning percentage for player 0 (red)")
        axs[0].plot(range(N), percentage_win_player_0, color="red")
        axs[0].axhline(y=np.mean(percentage_win_player_0), color="red", linestyle="--")
        axs[0].axhline(y=0.5, color="gray", linestyle="--")
        axs[0].set_xlabel("Epoch")

        axs[1].set_title("Number of moves needed to win")
        axs[1].plot(
            range(len(mobile_mean_nb_moves_to_win)), mobile_mean_nb_moves_to_win, color="grey"
        )
        axs[1].set_xlabel("Epoch")

        axs[2].set_title("Histogram of number of moves needed to win")
        axs[2].hist(nb_moves_to_win, color="grey")
        axs[2].set_xlabel("Epoch")

        axs[3].set_title("Exploration factor")
        axs[3].plot(
            range(len(explo_factor)), explo_factor, color="grey"
        )
        axs[3].set_xlabel("Epoch")
