import numpy as np
from connect4 import ConnectFourGame 
from qlearning import QAgent

# Define state and action spaces
state_size = 6 * 7  # size of the board
action_size = 7  # number of columns

# Initialize Q-learning agent
agent = QAgent(state_size, action_size)

# Train the agent
episodes = 10000  # number of games to play
for episode in range(episodes):
    # Start new game
    game = ConnectFourGame()
    done = False
    while not done:
        # Get current state
        state = game.board.flatten()

        # Choose action
        action = agent.choose_action(state)

        # Take action and get reward and next state
        reward, next_state, done = game.play_move(action)
        next_state = next_state.flatten()

        # Update Q-table
        agent.learn(state, action, reward, next_state)

# Evaluate the agent
game = ConnectFourGame()
done = False
while not done:
    # Get current state
    state = game.board.flatten()

    # Choose action
    action = agent.choose_action(state)

    # Take action and get reward and next state
    reward, next_state, done = game.play_move(action)
    next_state = next_state.flatten()

    # Print board
    print(game.board)

    # Switch players
    game.switch_players()
