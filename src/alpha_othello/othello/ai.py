from typing import Optional

import numpy as np

from alpha_othello.othello.board import get_flips, get_valid_moves


def ai_pass(board: np.ndarray, player: bool, size: int) -> Optional[tuple[int, int]]:
    return None


def ai_random(board: np.ndarray, player: bool, size: int) -> Optional[tuple[int, int]]:
    valid_moves = get_valid_moves(board, player, size)
    if valid_moves:
        return valid_moves[np.random.randint(len(valid_moves))]
    return None


def ai_greedy(board: np.ndarray, player: bool, size: int) -> Optional[tuple[int, int]]:
    valid_moves = get_valid_moves(board, player, size)
    best_move = None
    best_score = 0
    for move in valid_moves:
        score = len(get_flips(board, player, size, *move))
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def ai_minimax(board: np.ndarray, player: bool, size: int) -> Optional[tuple[int, int]]:
    valid_moves = get_valid_moves(board, player, size)
    if not valid_moves:
        return None

    def minimax(board, player, size, depth, maximizing):
        moves = get_valid_moves(board, player, size)
        if depth == 0 or not moves:
            return np.sum(board == player), None
        best_move = None
        if maximizing:
            max_eval = -float("inf")
            for move in moves:
                flips = get_flips(board, player, size, *move)
                new_board = board.copy()
                new_board[move] = player
                for fx, fy in flips:
                    new_board[fx, fy] = player
                eval_score, _ = minimax(new_board, not player, size, depth - 1, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
            return max_eval, best_move
        else:
            min_eval = float("inf")
            for move in moves:
                flips = get_flips(board, player, size, *move)
                new_board = board.copy()
                new_board[move] = player
                for fx, fy in flips:
                    new_board[fx, fy] = player
                eval_score, _ = minimax(new_board, not player, size, depth - 1, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
            return min_eval, best_move

    _, move = minimax(board, player, size, depth=2, maximizing=True)
    return move


def ai_heuristic(
    board: np.ndarray, player: bool, size: int
) -> Optional[tuple[int, int]]:
    valid_moves = get_valid_moves(board, player, size)
    if not valid_moves:
        return None

    def score_move(move) -> float:
        # Copy the board and apply the move
        new_board = board.copy()
        flips = get_flips(board, player, size, *move)
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
        opp_moves = len(get_valid_moves(new_board, not player, size))
        mobility = -opp_moves

        # Weighted sum
        return (
            10 * len(flips)  # prioritize flipping more discs
            + corner_bonus
            + edge_bonus
            + 2 * disc_diff
            + 3 * mobility
        )

    return max(valid_moves, key=score_move)
