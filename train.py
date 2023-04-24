import torch
import numpy as np
from src.deepqlearning import DeepQlearning
from src.deepqlearning import NeuralNetwork
from pettingzoo.classic import connect_four_v3
from tqdm import tqdm
import os


def main():
    # Define hyperparameters
    initial_exploration_factor = 0.5
    final_exploration_factor = 0.1
    discount_factor = 0.7
    learning_rate = 0.1
    n_training_game = 100000
    batch_size = 64
    model_save_path = 'models/deepqlearning/model.pt'
    stats_save_path = 'models/deepqlearning/stats.npy'

    # Create game environment
    #render = None if training else "human"
    connect_four_v3.env(render_mode=None)

    # Create neural network model
    model = NeuralNetwork()
    dql = DeepQlearning(
        initial_exploration_factor=initial_exploration_factor,
        final_exploration_factor=final_exploration_factor,
        discount_factor=discount_factor,
        learning_rate=learning_rate,
        model=model,
    )

    # Train the model
    dql.training(n_training_game=n_training_game, batch_size=batch_size)

    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(dql.model.state_dict(), model_save_path)
    np.save(stats_save_path, dql.stats)



if __name__ == "__main__":
    main()
