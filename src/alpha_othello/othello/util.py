from typing import List, Optional, Tuple


def move_to_string(moves: Optional[Tuple[int, int]]) -> str:
    if moves is None:
        return "pass"
    row, col = moves
    return f"{chr(col + ord('A'))}{row + 1}"


def moves_to_string(moves: List[Optional[Tuple[int, int]]]) -> str:
    return " ".join(f"{i}. {move_to_string(move)}" for i, move in enumerate(moves))
