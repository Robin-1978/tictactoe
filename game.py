import numpy as np

class Game:
    def __init__(self, board_size=3, win_length=3):
        self.board_size = board_size
        self.win_length = win_length
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1

    def get_legal_moves(self):
        return np.argwhere(self.board == 0).tolist()

    def make_move(self, move):
        row, col = move
        if self.board[row][col] != 0:
            return False
        self.board[row][col] = self.current_player
        if self.check_win(row, col):
            return True
        self.current_player = 2 if self.current_player == 1 else 1
        return False

    def check_win(self, row, col):
        player = self.board[row][col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 水平、垂直、正斜线、反斜线

        for dr, dc in directions:
            count = 1  # 当前位置已经有一个棋子
            # 向一个方向检查
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc
            # 向相反方向检查
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            # 检查是否达到获胜条件
            if count >= self.win_length:
                return True
        return False

    def is_full(self):
        return np.all(self.board != 0)

    def get_state(self):
        """
        AlphaZero标准状态表示：
        第0层: 玩家1(X)的棋子
        第1层: 玩家2(O)的棋子  
        第2层: 当前玩家标识 (1表示玩家1的回合，0表示玩家2的回合)
        """
        state = np.zeros((3, self.board_size, self.board_size), dtype=int)
        state[0] = (self.board == 1).astype(int)  # 玩家1(X)
        state[1] = (self.board == 2).astype(int)  # 玩家2(O)
        state[2] = int(self.current_player == 1)  # 当前玩家标识
        return state

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1

    def render(self):
        print("  " + " ".join(str(i) for i in range(self.board_size)))
        for i in range(self.board_size):
            print(f"{i} " + " ".join(
                "X" if self.board[i, j] == 1 else 
                "O" if self.board[i, j] == 2 else 
                "." for j in range(self.board_size)
            ))
        print()
