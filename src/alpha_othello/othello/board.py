from typing import List, Optional, Tuple

import numpy as np


def get_valid_moves(board: np.ndarray, player: bool) -> List[Tuple[int, int]]:
    size = get_size(board)
    valid_moves = []

    for row in range(size):
        for col in range(size):
            if is_valid_move(board, player, row, col):
                valid_moves.append((row, col))
    return valid_moves


def is_valid_move(board: np.ndarray, player: bool, row: int, col: int) -> bool:
    if board[row, col].any():
        return False
    flips = get_flips(board, player, row, col)
    return len(flips) > 0


def get_flips(
    board: np.ndarray, player: bool, row: int, col: int
) -> List[Tuple[int, int]]:
    size = get_size(board)
    flips = []
    directions = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        temp_flips = []
        while 0 <= r < size and 0 <= c < size:
            if board[r, c, int(player)]:
                flips.extend(temp_flips)
                break
            elif board[r, c, int(not player)]:
                temp_flips.append((r, c))
                r += dr
                c += dc
            else:
                break
    return flips


def get_size(board: np.ndarray) -> int:
    return board.shape[0]


class OthelloState:
    def __init__(self, size: int = 8):
        self.player: bool = False

        self.board: np.ndarray = np.zeros((size, size, 2), dtype=bool)
        mid = size // 2
        self.board[mid - 1, mid - 1, int(self.player)] = True
        self.board[mid, mid, int(self.player)] = True
        self.board[mid - 1, mid, int(not self.player)] = True
        self.board[mid, mid - 1, int(not self.player)] = True

    def get_score(self) -> Tuple[int, int]:
        black_count = np.sum(self.board[:, :, 0]).item()
        white_count = np.sum(self.board[:, :, 1]).item()
        return black_count, white_count

    def make_move(self, move: Optional[Tuple[int, int]]):
        if move is None:
            if get_valid_moves(self.board, self.player):
                raise ValueError("Cannot pass when there are valid moves.")
            self.player = not self.player
            return

        row, col = move
        flips = get_flips(self.board, self.player, row, col)

        if not flips:
            raise ValueError("Invalid move.")

        for r, c in flips:
            self.board[r, c, int(self.player)] = True
            self.board[r, c, int(not self.player)] = False
        self.board[row, col, int(self.player)] = True
        self.board[row, col, int(not self.player)] = False
        self.player = not self.player

    @property
    def size(self) -> int:
        return get_size(self.board)

    def __eq__(self, other) -> bool:
        if not isinstance(other, OthelloState):
            return False
        return self.player == other.player and np.array_equal(self.board, other.board)

    def __str__(self) -> str:
        board_str = ""
        legal_moves = get_valid_moves(self.board, self.player)
        for row in range(self.size):
            board_str += f"{row + 1} "
            for col in range(self.size):
                if self.board[row, col, 0]:
                    board_str += "b "
                elif self.board[row, col, 1]:
                    board_str += "w "
                elif (row, col) in legal_moves:
                    board_str += "* "
                else:
                    board_str += ". "
            board_str += "\n"
        board_str += (
            "  " + " ".join(chr(col + ord("A")) for col in range(self.size)) + "\n"
        )
        return board_str
