from typing import Annotated, Literal, TypeVar

import numpy as np
import numpy.typing as npt

from alpha_othello.othello.board import get_flips, get_size, get_valid_moves

DType = TypeVar("DType", bound=np.generic)
T_BOARD = Annotated[npt.NDArray[DType], Literal["size", "size", 2]]
T_MOVE = tuple[int, int]


def ai_random(board: T_BOARD, player: bool, time_remaining: tuple[int, int]) -> T_MOVE:
    valid_moves = get_valid_moves(board, player)
    return valid_moves[np.random.randint(len(valid_moves))]


def ai_greedy(board: T_BOARD, player: bool, time_remaining: tuple[int, int]) -> T_MOVE:
    valid_moves = get_valid_moves(board, player)
    return max(valid_moves, key=lambda move: len(get_flips(board, player, *move)))


def ai_minimax(board: T_BOARD, player: bool, time_remaining: tuple[int, int]) -> T_MOVE:
    def minimax(board, player, depth, maximizing):
        moves = get_valid_moves(board, player)
        if depth == 0 or not moves:
            return np.sum(board == player), None
        best_move = None
        if maximizing:
            max_eval = -float("inf")
            for move in moves:
                flips = get_flips(board, player, *move)
                new_board = board.copy()
                new_board[move] = player
                for fx, fy in flips:
                    new_board[fx, fy] = player
                eval_score, _ = minimax(new_board, not player, depth - 1, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
            return max_eval, best_move
        else:
            min_eval = float("inf")
            for move in moves:
                flips = get_flips(board, player, *move)
                new_board = board.copy()
                new_board[move] = player
                for fx, fy in flips:
                    new_board[fx, fy] = player
                eval_score, _ = minimax(new_board, not player, depth - 1, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
            return min_eval, best_move

    _, move = minimax(board, player, depth=2, maximizing=True)
    if move is None:
        raise ValueError("No valid moves available")
    return move


def ai_heuristic(
    board: T_BOARD, player: bool, time_remaining: tuple[int, int]
) -> T_MOVE:
    size = get_size(board)

    def score_move(move) -> float:
        # Copy the board and apply the move
        new_board = board.copy()
        flips = get_flips(board, player, *move)
        new_board[move] = player
        for fx, fy in flips:
            new_board[fx, fy] = player

        # Heuristic: corners are best, edges are good, avoid giving up corners
        x, y = move
        corner_bonus = (
            25
            if (x, y) in [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
            else 0
        )
        edge_bonus = 5 if (x == 0 or x == size - 1 or y == 0 or y == size - 1) else 0

        # Disc difference
        disc_diff = np.sum(new_board == player) - np.sum(new_board == (not player))

        # Mobility (number of moves next turn)
        opp_moves = len(get_valid_moves(new_board, not player))
        mobility = -opp_moves

        # Weighted sum
        return (
            10 * len(flips)  # prioritize flipping more discs
            + corner_bonus
            + edge_bonus
            + 2 * disc_diff
            + 3 * mobility
        )

    valid_moves = get_valid_moves(board, player)
    return max(valid_moves, key=score_move)


def ai_mobility(
    board: T_BOARD, player: bool, time_remaining: tuple[int, int]
) -> T_MOVE:
    """
    AI that chooses the move maximizing its own mobility (number of valid moves next turn).
    If multiple moves yield the same mobility, pick one at random.
    """
    valid_moves = get_valid_moves(board, player)

    best_moves = []
    max_mobility = -float("inf")
    for move in valid_moves:
        # Apply move
        new_board = board.copy()
        flips = get_flips(board, player, *move)
        new_board[move] = player
        for fx, fy in flips:
            new_board[fx, fy] = player
        # Count mobility for next turn
        mobility = len(get_valid_moves(new_board, player))
        if mobility > max_mobility:
            max_mobility = mobility
            best_moves = [move]
        elif mobility == max_mobility:
            best_moves.append(move)
    return best_moves[np.random.randint(len(best_moves))]


def ai_parity(board: T_BOARD, player: bool, time_remaining: tuple[int, int]) -> T_MOVE:
    """
    AI that prefers moves leaving an even number of empty squares (parity heuristic).
    In Othello, parity is often advantageous in endgames.
    """
    valid_moves = get_valid_moves(board, player)

    def parity_score(move):
        new_board = board.copy()
        flips = get_flips(board, player, *move)
        new_board[move] = player
        for fx, fy in flips:
            new_board[fx, fy] = player
        empty_count = np.sum(new_board == -1)  # assuming -1 is empty
        # Prefer even parity
        return -(empty_count % 2)

    return max(valid_moves, key=parity_score)
