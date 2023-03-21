import numpy as np

class ConnectFourGame:
    def __init__(self):
        self.board = np.zeros((6, 7))
        self.current_player = 1
        
    def play_move(self, column):
        if self.is_valid_move(column):
            row = self.get_next_open_row(column)
            self.board[row][column] = self.current_player
            
            # Check for a win
            if self.is_winner(row, column):
                reward = 1
                done = True
            else:
                reward = 0
                done = False
                
            # Switch players
            self.switch_players()
            
            return reward, self.board, done
        else:
            # Invalid move, penalize the agent
            return -1, self.board, False
        
    def is_valid_move(self, column):
        return self.board[0][column] == 0
    
    def get_next_open_row(self, column):
        for r in range(5, -1, -1):
            if self.board[r][column] == 0:
                return r
        return -1
    
    def is_winner(self, row, column):
        # Check horizontal
        for c in range(4):
            if self.board[row][c] == self.current_player and self.board[row][c+1] == self.current_player and self.board[row][c+2] == self.current_player and self.board[row][c+3] == self.current_player:
                return True
        
        # Check vertical
        for r in range(3):
            if self.board[r][column] == self.current_player and self.board[r+1][column] == self.current_player and self.board[r+2][column] == self.current_player and self.board[r+3][column] == self.current_player:
                return True
        
        # Check diagonal (down-right)
        for r in range(3):
            for c in range(4):
                if self.board[r][c] == self.current_player and self.board[r+1][c+1] == self.current_player and self.board[r+2][c+2] == self.current_player and self.board[r+3][c+3] == self.current_player:
                    return True
        
        # Check diagonal (up-right)
        for r in range(3, 6):
            for c in range(4):
                if self.board[r][c] == self.current_player and self.board[r-1][c+1] == self.current_player and self.board[r-2][c+2] == self.current_player and self.board[r-3][c+3] == self.current_player:
                    return True
        
        return False
    
    def switch_players(self):
        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1