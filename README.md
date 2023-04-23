# Connect4 - Reinforcement Learning

## Introduction :wave:
The well-known game of Connect4 is an interesting example to implement in Reinforcement Learning. Indeed, the rules are mastered by everyone, the game environment is a simple grid to model, and the game involves two players who act successively.

In the context of our project, we will use the PettingZoo library and its API to visualize the game of Connect4 and the tokens placed by the players.

Our group, consisting of Tanguy Blervacque, Lucie Clémot, Nicolas Péruchot and Thomas Vicaire, has chosen two more or less sophisticated learning models (Q-Learning and Deep-Q-Learning) and worked on the quality of our learning by adding strategic aspects and quantifying the evolution of victories obtained by our agent. Moreover, we studied these elements during a game between two agents, between an agent and a random player, and between an agent and a human player.

## Setup
To setup your environment in order to use this project, run the following line in a terminal :
`make install`.

## Repository structure

### Repository tree
Our git repository has the following structure :
```
connect4
├─ Makefile
├─ README.md
├─ main_play.py
├─ notebooks
│  ├─ deepqlearning.ipynb
│  ├─ qlearning.ipynb
│  └─ test_agents.ipynb
├─ pyproject.toml
├─ setup.py
├─ src
│  ├─ agent.py
│  ├─ base_model.py
│  ├─ check_functions.
│  ├─ deepqlearning.py
│  └─ qlearning.py
└─ tools
   ├─ images
   │  └─ tree.png
   ├─ train.py
   ├─ unit_test.py
   └─ win_checks.py

```

### Files and Folders
The main folders and files to understand are :
- The [src folder](src/) holds the main elements needed to run our project. In the files [base_model.py](src/base_model.py), [qlearning.py](src/qlearning.py) and [deepqlearning.py](src/deepqlearning.py), we create the Base model, the Q-Learning and Deep-Q-Learning model classes respectively, along with the needed methods to visualize the training evolution and to visualize the game. Moreover, this folder holds the [agent.py](src/agent.py) file which creates the agent class.
- The [tools folder](tools/) holds the main tools needed for our project. Mainly, the [win_checks.py](tools/win_checks.py) file has the functions needed to analyze if a win is possible in different configurations (a column, row or diagonal win) aswell as to identify situations where a defense play is possible, and when this play is taken. Secondly, the [unit_test.py](tools/unit_test.py) file (which can be run using `pytest -s` from the [tools folder](tools/)) tests every one of these functions to ensure we are getting the wanted behavior.
- The [notebooks folder](notebooks/) holds three main notebooks : one per model, [qlearning.ipynb](notebooks/qlearning.ipynb) and [deepqlearning.ipynb](notebooks/deepqlearning.ipynb), which trains and saves the models, aswell as [test_agents.ipynb](notebooks/test_agents.ipynb) which enables the user to study the training and learning evolution.
- The [main_play](main_play.py) file allows the user to play against our agent, using a terminal to indicate the requested plays, and using the PettingZoo graphic interfact to visualize the game.

## Key takeaways
