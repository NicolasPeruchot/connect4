import torch
import numpy as np
from src.deepqlearning import DeepQlearning, NeuralNetwork
from src.qlearning import Qlearning
from pettingzoo.classic import connect_four_v3
from tqdm import tqdm
import os
import click

@click.command()
@click.option(
    '--save-dir',
    required=True,
    type=click.Path()
)
@click.option(
    '--model-type',
    required=True,
    type=click.Path()
)
def main(save_dir, model_type):
    # Define hyperparameters
    initial_exploration_factor = 0.5
    final_exploration_factor = 0.1
    discount_factor = 0.7
    learning_rate = 0.1
    n_training_game = 100000
    batch_size = 64
    model_save_path = os.path.join(save_dir, model_type, 'model.pt')

    # Create game environment
    #render = None if training else "human"
    connect_four_v3.env(render_mode=None)

    if model_type == 'deepqlearning':
        # Create neural network model
        model = NeuralNetwork()
        model = DeepQlearning(
            initial_exploration_factor=initial_exploration_factor,
            final_exploration_factor=final_exploration_factor,
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            model=model,
        )
        # Train the model
        model.training(n_training_game=n_training_game, batch_size=batch_size)

    if model_type == 'qlearning':
        # Create neural network model
        model = Qlearning(
            initial_exploration_factor=initial_exploration_factor,
            final_exploration_factor=final_exploration_factor,
            discount_factor=discount_factor,
            learning_rate=learning_rate
        )
        # Train the model
        model.training(n_training_game=n_training_game)

    # Save the model and training stats
    torch.save(model, model_save_path)
    print('Model and stats have been saved')


if __name__ == "__main__":
    main()
