import sys

from src.deepqlearning import DeepQlearning, NeuralNetwork
from src.agent import Player
import matplotlib.pyplot as plt


if __name__ == "__main__":
    Q = DeepQlearning(model=NeuralNetwork(), initial_exploration_factor=0.8)
    #Q.training(n_training_game=1000, batch_size=64)
    Q.play_user("player_1")