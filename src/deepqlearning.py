import numpy as np
import torch

from torch import nn
from tqdm import tqdm

from base_model import BaseQLearningModel


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=6 * 7 * 2, output_size=7):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.stack = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.input_size**2),
            nn.PReLU(),
            nn.Linear(self.input_size**2, self.output_size),
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits


class DeepQlearning(BaseQLearningModel):
    def __init__(
        self,
        initial_exploration_factor=0.5,
        final_exploration_factor=0.1,
        discount_factor=0.7,
        learning_rate=0.1,
        model=None,
    ) -> None:
        super().__init__(
            initial_exploration_factor, final_exploration_factor, discount_factor, learning_rate
        )
        self.model = model
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def get_action(self, state):
        y = self.model(torch.Tensor(state["observation"].flatten()))
        possible = [y[i].item() if state["action_mask"][i] != 0 else -np.inf for i in range(7)]
        return np.argmax(possible)

    def training(self, n_training_game=1000, batch_size=100):
        self.initialize_game()
        game = 0
        for epoch in tqdm(range(n_training_game // batch_size)):
            inputs = []
            outputs = []
            for _ in range(batch_size):
                self.initialize_game(render=None)
                end = False
                i = 0
                while end is False:
                    end = self.play_game(i, game, n_training_game)
                    inputs, outputs = self.update_policy(end, i, inputs, outputs, game, n_training_game)

                    i += 1
                game += 1
                self.env.close()

            inputs = torch.Tensor(np.array(inputs))
            outputs = torch.Tensor(torch.cat(outputs, dim=0)).reshape(
                inputs.shape[0], 1, self.model.output_size
            )

            pred = self.model(inputs)
            loss = self.loss(pred.unsqueeze(1), outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_policy(self, end, i, inputs, outputs, game, n_training_game):
        y = self.model(torch.Tensor(self.agents[i % 2]["last_state"].flatten())).detach().numpy()

        if end:
            if self.agents[i % 2]["reward"] == 1:
                self.agents[(i + 1) % 2]["reward"] = -1

            for j in [0, 1]:
                try:
                    Q_sa = torch.max(
                        self.model(torch.Tensor(self.agents[j]["current_state"].flatten()))
                    ).item()
                    y = self.model(torch.Tensor(self.agents[j]["last_state"].flatten()))
                    y[self.agents[j]["last_action"]] = (
                        self.agents[j]["reward"] + self.discount_factor * Q_sa
                    )

                    inputs.append(self.agents[j]["last_state"].flatten())
                    outputs.append(y)

                except:
                    pass

            # Update stats
            self.update_stats(winner=self.agents[i % 2]["name"], nb_moves_to_win=i+1, game=game, n_training_game=n_training_game)

        elif self.agents[(i + 1) % 2]["last_state"] is not None:
            Q_sa = torch.max(
                self.model(torch.Tensor(self.agents[(i + 1) % 2]["current_state"].flatten()))
            ).item()
            y = self.model(torch.Tensor(self.agents[(i + 1) % 2]["last_state"].flatten()))
            y[self.agents[(i + 1) % 2]["last_action"]] = (
                self.agents[(i + 1) % 2]["reward"] + self.discount_factor * Q_sa
            )

            inputs.append(self.agents[(i + 1) % 2]["last_state"].flatten())
            outputs.append(y)

        return inputs, outputs
