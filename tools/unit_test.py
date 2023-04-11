from pettingzoo.classic import connect_four_v3
import pygame

from win_checks import win_row, win_column, win_diagonal

env = connect_four_v3.env()

### Test the row function ###
def test_row_gravity():
    print("\nRow test 1 : testing gravity")
    env.reset()
    current_agent = "player_0"
    env.agent_selection = current_agent

    for x in [0,2,1,4,0,5,1,6,2,4]:
        env.step(x)

    state = env.observe(agent="player_0")["observation"]
    grille = state[:, :, 0] + state[:, :, 1] * 2
    print(grille)
    print("In this situation, the agent shouldn't be able to win.")
    assert(win_row(state) == False)

def test_row_2_and_2():
    print("\nRow test 2 : testing 2 and 2 configuration")
    env.reset()
    current_agent = "player_0"
    env.agent_selection = current_agent

    for x in [0,5,1,0,3,1,4,4]:
        env.step(x)

    state = env.observe(agent="player_0")["observation"]
    grille = state[:, :, 0] + state[:, :, 1] * 2
    print(grille)
    print("In this situation, the agent should be able to win.")
    assert(win_row(state) == True)

### Test the column function ###
def test_column():
    print("\nColumn test 1 : basic test")
    env.reset()
    current_agent = "player_0"
    env.agent_selection = current_agent

    for x in [1,1,1,2,1,4,1]:
        env.step(x)

    state = env.observe(agent="player_0")["observation"]
    grille = state[:, :, 0] + state[:, :, 1] * 2
    print(grille)
    print("In this situation, the agent should be able to win.")
    assert(win_column(state) == True)

def test_full_column():
    print("\nColumn test 2 : full column configuration")
    env.reset()
    current_agent = "player_0"
    env.agent_selection = current_agent

    for x in [1,1,2,1,1,4,1,2,2,3,1]:
        env.step(x)

    state = env.observe(agent="player_0")["observation"]
    grille = state[:, :, 0] + state[:, :, 1] * 2
    print(grille)
    print("In this situation, the agent shouldn't be able to win.")
    assert(win_column(state) == False)

### Test the dialog function ###
def test_right_diagonal():
    print("\nDiagonal test 1 : Classic right diagonal")
    env.reset()
    current_agent = "player_0"
    env.agent_selection = current_agent

    for x in [0,1,1,2,3,2,4,3,2,3]:
        env.step(x)

    state = env.observe(agent="player_0")["observation"]
    grille = state[:, :, 0] + state[:, :, 1] * 2
    print(grille)
    print("In this situation, the agent should be able to win.")

    assert(win_diagonal(state) == True)

def test_left_diagonal():
    print("\nDiagonal test 2 : Classic left diagonal")
    env.reset()
    current_agent = "player_0"
    env.agent_selection = current_agent

    for x in [1,0,4,1,1,2,3,0,2,0]:
        env.step(x)

    state = env.observe(agent="player_0")["observation"]
    grille = state[:, :, 0] + state[:, :, 1] * 2
    print(grille)
    print("In this situation, the agent should be able to win.")

    assert(win_diagonal(state) == True)

def test_diagonal_gravity():
    print("\nDiagonal test 3 : Gravity diagonal")
    env.reset()
    current_agent = "player_0"
    env.agent_selection = current_agent

    for x in [0,1,1,2,4,2,2]:
        env.step(x)

    state = env.observe(agent="player_0")["observation"]
    grille = state[:, :, 0] + state[:, :, 1] * 2
    print(grille)
    print("In this situation, the agent shouldn't be able to win.")

    assert(win_diagonal(state) == False)