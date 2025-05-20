import threading
import time
from collections import defaultdict
from enum import Enum
from itertools import product
from typing import Callable, Optional

import numpy as np
from tqdm.auto import tqdm

from alpha_othello.othello.board import OthelloState, get_valid_moves
from alpha_othello.othello.util import move_to_string, moves_to_string

AI_FN = Callable[[np.ndarray, bool, tuple[int, int]], tuple[int, int]]


class GameResult(Enum):
    BLACK_WINS = "black"
    WHITE_WINS = "white"
    DRAW = "draw"


class GameOverReason(Enum):
    NONE_EMPTY = "none_empty"
    TWO_PASSES = "two_passes"
    ILLEGAL_LIMIT = "illegal_limit"
    TIME_LIMIT = "time_limit"


class OthelloGame:
    def __init__(
        self,
        ai_fns: tuple[AI_FN, AI_FN],
        size: int = 8,
        time_control_millis: int = 100,
        illegal_limit: int = 3,
    ) -> None:
        self.ai_fns = ai_fns
        self.size = size
        self.state = OthelloState(self.size)
        self.moves = []
        self.times = []
        self.illegal_remaining = [illegal_limit, illegal_limit]
        self.remaining_time = (time_control_millis, time_control_millis)
        self.result = None
        self.reason = None

    def get_ai_move(self) -> tuple[tuple[int, int], int]:
        _move: list = [None]
        exception: list = [None]

        def ai_call():
            try:
                board = self.state.board.copy()
                _move[0] = self.ai_fns[self.state.player](
                    board, self.state.player, self.remaining_time
                )
            except Exception as e:
                exception[0] = e

        start_time = time.time()
        thread = threading.Thread(target=ai_call)
        thread.start()
        thread.join(self.remaining_time[self.state.player] / 1000)
        end_time = time.time()

        return _move[0], int((end_time - start_time) * 1000)

    def play(self) -> None:
        while True:
            self.done, self.reason = self.is_game_over()
            if self.done:
                self.result = self.get_result()
                break

            if get_valid_moves(self.state.board, self.state.player):
                move, elapsed_time = self.get_ai_move()
            else:
                # If there are no valid moves, just pass without querying the AI_FN
                move = None
                elapsed_time = 0

            if self.state.player == 0:
                remaining_time = (self.remaining_time[0] - elapsed_time, self.remaining_time[1])
            else:
                remaining_time = (self.remaining_time[0], self.remaining_time[1] - elapsed_time)
            self.remaining_time = remaining_time
            self.times.append(remaining_time[self.state.player])
            if any(x <= 0 for x in self.remaining_time):
                continue

            try:
                self.state.make_move(move)
                self.moves.append(move)
            except ValueError:
                self.illegal_remaining[self.state.player] -= 1
                continue

    def is_game_over(self) -> tuple[bool, Optional[GameOverReason]]:
        if self.state.board.sum(axis=-1).all():
            return True, GameOverReason.NONE_EMPTY
        if len(self.moves) > 1 and all(move is None for move in self.moves[-2:]):
            return True, GameOverReason.TWO_PASSES
        if any(x <= 0 for x in self.illegal_remaining):
            return True, GameOverReason.ILLEGAL_LIMIT
        if any(x <= 0 for x in self.remaining_time):
            return True, GameOverReason.TIME_LIMIT
        return False, None

    def get_result(self) -> Optional[GameResult]:
        if self.reason is None:
            return None
        if self.reason == GameOverReason.TIME_LIMIT:
            if self.remaining_time[0] <= 0:
                return GameResult.WHITE_WINS
            else:
                return GameResult.BLACK_WINS
        if self.reason == GameOverReason.ILLEGAL_LIMIT:
            if self.illegal_remaining[0] <= 0:
                return GameResult.WHITE_WINS
            else:
                return GameResult.BLACK_WINS
        black_count, white_count = self.state.get_score()
        if black_count > white_count:
            return GameResult.BLACK_WINS
        elif black_count < white_count:
            return GameResult.WHITE_WINS
        else:
            return GameResult.DRAW

    def __str__(self) -> str:
        state = OthelloState(self.size)
        game_str = str(state) + "\n"
        for i, move in enumerate(self.moves):
            state.make_move(move)
            game_str += f"{i}. {move_to_string(move)}"
            if self.times:
                game_str += f" ({self.times[i]} ms)"
            game_str += "\n" + str(state) + "\n"
        game_str += moves_to_string(self.moves) + "\n"

        if self.reason is not None and self.result is not None:
            game_str += f"Game Over! Reason: {self.reason.value}\n"
            winner_str = self.result.value
            score_str = "-".join(map(str, state.get_score()))
            game_str += f"Winner: {winner_str} ({score_str})\n"
        return game_str


def run_tournament(
    ais: list[AI_FN],
    size: int = 6,
    n_games_per_pair: int = 100,
    time_control_millis: int = 100,
):
    pairs = [(ai1, ai2) for ai1 in ais for ai2 in ais]
    n_pairs = len(pairs)
    n_games = n_pairs * n_games_per_pair
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for i, _ in tqdm(product(range(n_pairs), range(n_games_per_pair)), total=n_games):
        ai1, ai2 = pairs[i]
        game = OthelloGame(
            (ai1, ai2), size=size, time_control_millis=time_control_millis
        )
        game.play()
        result = game.result
        reason = game.reason
        if result is None or reason is None:
            raise ValueError("Game did not finish properly.")
        results[(ai1.__name__, ai2.__name__)][result.value][reason.value] += 1

    # results: {(ai1_name, ai2_name): {result: {reason: count}}}
    return results
