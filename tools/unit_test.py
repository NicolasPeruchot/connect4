from pettingzoo.classic import connect_four_v3
import pygame

# from tools.win_checks import win_row, win_column, win_diagonal


def win_row(grille):
    winning_position = [0,1,1,1]
    below = [True for _ in range(grille.shape[1])]
    for i in range(grille.shape[0] - 1, -1, -1):
        for j in range (0,4):
            current_check=grille[i][j:j+4].tolist()
            current_check_ordered = current_check.copy()
            current_check_ordered.sort()
            if current_check_ordered == winning_position:
                if below[current_check.index(0)+j]:
                    return True
        below = [x!=0 for x in grille[i]]
    return False


def win_column(grille):
    for j in range(grille.shape[1]):
        if grille[0, j] == 0 and [x for x in grille[:, j] if x != 0][:3] == [1, 1, 1]:
            return True
    return False


def win_diagonal(grille):
    winning_position = [0, 1, 1, 1]
    rows, cols = grille.shape
    for i in range(rows - 3):
        for j in range(cols - 3):
            # Vérifie la diagonale montante (de gauche à droite)
            if grille[i+3][j] == 0 and grille[i+2][j+1] == 1 and grille[i+1][j+2] == 1 and grille[i][j+3] == 1:
                if i == rows - 4 or grille[i+4][j] != 0:
                    return True
            # Vérifie la diagonale descendante (de gauche à droite)
            if grille[i][j] == 1 and grille[i+1][j+1] == 1 and grille[i+2][j+2] == 1 and grille[i+3][j+3] == 0:
                if i == 0 or grille[i-1][j+3] != 0:
                    return True
            # Vérifie la diagonale montante (de droite à gauche)
            if grille[i+3][j+3] == 0 and grille[i+2][j+2] == 1 and grille[i+1][j+1] == 1 and grille[i][j] == 1:
                if i == rows - 4 or grille[i+4][j+3] != 0:
                    return True
            # Vérifie la diagonale descendante (de droite à gauche)
            if grille[i][j+3] == 0 and grille[i+1][j+2] == 1 and grille[i+2][j+1] == 1 and grille[i+3][j] == 1:
                if i == 0 or grille[i-1][j] != 0:
                    return True
    return False

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
    assert(win_row(grille) == False)

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
    assert(win_row(grille) == True)

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
    assert(win_column(grille) == True)

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
    assert(win_column(grille) == False)

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

    assert(win_diagonal(grille) == False)

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

    assert(win_diagonal(grille) == False)

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

    assert(win_diagonal(grille) == False)