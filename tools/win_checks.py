from pettingzoo.classic import connect_four_v3
import numpy as np


def is_valid_play(grille, position):
    (i, j) = position
    rows, cols = grille.shape
    # Check nothing above
    if np.all(grille[:i, j] == 0):
        # Check something underneath
        if i == rows - 1 or grille[i + 1, j] != 0:
            return True
    return False


def win_row(state):
    grille = state[:, :, 0] + state[:, :, 1] * 2
    winning_position = [0, 1, 1, 1]
    below = [True for _ in range(grille.shape[1])]
    for i in range(grille.shape[0] - 1, -1, -1):
        for j in range(0, 4):
            current_check = grille[i, j : j + 4].tolist()
            current_check_ordered = current_check.copy()
            current_check_ordered.sort()
            if current_check_ordered == winning_position:
                if below[current_check.index(0) + j]:
                    return True
        below = [x != 0 for x in grille[i]]
    return False


def win_column(state):
    grille = state[:, :, 0] + state[:, :, 1] * 2
    for j in range(grille.shape[1]):
        # S'il est possible de jouer dans dans une colonne, on vérifie si il y a victoire direct si on y joue
        if grille[0, j] == 0 and [x for x in grille[:, j] if (x != 0)][:3] == [1, 1, 1]:
            return True
    return False


def win_diagonal(state):
    grille = state[:, :, 0] + state[:, :, 1] * 2
    rows, cols = grille.shape
    # Vérifie les diagonales montantes
    for i in range(3, rows):
        for j in range(cols - 3):
            n_zeros = 0
            n_ones = 0
            for idx in range(4):
                if grille[i - idx, j + idx] == 1:
                    n_ones += 1
                if grille[i - idx, j + idx] == 0:
                    n_zeros += 1
                    zero_position = (i - idx, j + idx)
            if n_ones == 3 and n_zeros == 1 and is_valid_play(grille, zero_position):
                return True
    # Vérifie les diagonale descendante
    for i in range(rows - 3):
        for j in range(cols - 3):
            n_zeros = 0
            n_ones = 0
            for idx in range(4):
                if grille[i + idx, j + idx] == 1:
                    n_ones += 1
                if grille[i + idx, j + idx] == 0:
                    n_zeros += 1
                    zero_position = (i + idx, j + idx)
            if n_ones == 3 and n_zeros == 1 and is_valid_play(grille, zero_position):
                return True
    return False


def is_direct_win(state):
    return win_column(state) or win_row(state) or win_diagonal(state)


def is_direct_defense(state):
    sub_state = state.copy()
    grille = state[:, :, 0] + state[:, :, 1] * 2

    # invert coins because we want to know what would happen in each column IF PLAYER_1 WAS TO PLAY
    sub_state[grille == 1] = np.array([0, 1])
    sub_state[grille == 2] = np.array([1, 0])

    return win_column(state) or win_row(state) or win_diagonal(state)

def was_succesfull_direct_defense(state, last_action):
    sub_state = state.copy()
    grille = state[:, :, 0] + state[:, :, 1] * 2

    # invert coins (because the state is for player_0, but it was actually player_1's turn before)
    sub_state[grille == 1] = np.array([0, 1])
    sub_state[grille == 2] = np.array([1, 0])
    grille[grille == 1] = 2
    grille[grille == 2] = 1

    # find state in which other player would have been without defense
    for i in range(state.shape[0]):
        if grille[i, last_action] != 0:
            sub_state[i, last_action] = np.array([0, 0])

    # If sub_state was a direct win situation, then this was a direct defense
    return win_column(sub_state) or win_row(sub_state) or win_diagonal(sub_state)
