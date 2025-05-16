# Skeleton code for generalised tree search to be filled in by LLMs

from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt

T_BOARD = Annotated[npt.NDArray[np.bool_], Literal["size", "size", 2]]
T_MOVE = tuple[int, int]


def get_size(board: T_BOARD) -> int:
    return board.shape[0]


def get_valid_moves(board: T_BOARD, player: bool) -> list[T_MOVE]:
    empty = np.logical_not(board.any(axis=2))
    valid_moves = [
        move for move in np.argwhere(empty) if is_valid_move(board, player, move)
    ]
    return valid_moves


def is_valid_move(board: T_BOARD, player: bool, move: T_MOVE) -> bool:
    if board[move].any():
        return False
    flips = get_flips(board, player, move)
    return len(flips) > 0


def get_flips(board: T_BOARD, player: bool, move: T_MOVE) -> list[T_MOVE]:
    size = get_size(board)
    row, col = move
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


def ai_tree_search(
    board: T_BOARD,
    player: bool,
    time_remaining: tuple[int, int],
) -> T_MOVE:
    return (0, 0)
