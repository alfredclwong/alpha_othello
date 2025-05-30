# Task:
# Choose circle (c, r) packing within a unit square that maximises the sum
# of the radii of the circles, while ensuring that no two circles overlap

import argparse
import signal
from enum import Enum, auto
from pathlib import Path


class Reason(Enum):
    VALID = auto()
    INVALID_TYPE = auto()
    INVALID_LENGTH = auto()
    INVALID_CIRCLE = auto()
    OUT_OF_BOUNDS = auto()
    OVERLAP = auto()
    TIMEOUT = auto()


def is_valid(
    packing: list[tuple[float, float, float]], tolerance: float = 1e-6
) -> tuple[bool, Reason]:
    # type checks
    if not isinstance(packing, list):
        print("Packing must be a list.")
        return False, Reason.INVALID_TYPE
    if len(packing) != 26:
        print("Packing must contain exactly 26 circles.")
        return False, Reason.INVALID_LENGTH
    if not all(isinstance(circle, tuple) and len(circle) == 3 for circle in packing):
        print("Each circle must be a tuple of (x, y, r).")
        return False, Reason.INVALID_CIRCLE
    if not all(
        isinstance(x, (int, float)) and x > 0 for circle in packing for x in circle
    ):
        print("Circle coordinates and radius must be positive numbers.")
        return False, Reason.INVALID_CIRCLE
    # check if any circle is outside the unit square
    for x, y, r in packing:
        if not (
            0 <= x - r <= 1 and 0 <= y - r <= 1 and 0 <= x + r <= 1 and 0 <= y + r <= 1
        ):
            return False, Reason.OUT_OF_BOUNDS
    # check if any two circles overlap
    for i in range(len(packing)):
        for j in range(i + 1, len(packing)):
            x1, y1, r1 = packing[i]
            x2, y2, r2 = packing[j]
            distance_squared = (x1 - x2) ** 2 + (y1 - y2) ** 2
            radius_sum = r1 + r2
            if distance_squared < (radius_sum - tolerance) ** 2:
                return False, Reason.OVERLAP
    return True, Reason.VALID


def score_packing(packing: list[tuple[float, float, float]]) -> float:
    return sum(r for _, _, r in packing)


def init_parser():
    parser = argparse.ArgumentParser(description="Evaluate circle packing solutions.")
    parser.add_argument(
        "-c", "--completion", type=str, required=True, help="Completion code path"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    return parser


def print_score(score: float, reason: Reason):
    score_dict = {
        f"{r.name}_CHECK": 0.0 if r == reason else 1.0
        for r in Reason if r != Reason.VALID
    }
    score_dict["SCORE"] = score
    score_str = ", ".join(
        f"{key}: {value}" for key, value in score_dict.items()
    )
    print(f"<SCORE>{score_str}</SCORE>")


def run_with_timeout(func, timeout):
    def timeout_handler(signum, frame):
        raise TimeoutError("Execution time exceeded the 5-minute limit.")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        return func()
    except TimeoutError:
        raise
    finally:
        signal.alarm(0)  # Disable the alarm after execution


def main():
    parser = init_parser()
    args = parser.parse_args()
    completion_path = Path(args.completion)
    verbose = args.verbose

    with open(completion_path, "r") as f:
        completion_str = f.read()

    local_vars = {}
    exec(f"def pack_26():\n{completion_str}", globals(), local_vars)

    score = 0.0
    try:
        packing = run_with_timeout(local_vars["pack_26"], 300)
        if verbose:
            print(f"<PACKING>{packing}</PACKING>")
        valid, reason = is_valid(packing)
        if valid:
            score = score_packing(packing)
    except TimeoutError:
        reason = Reason.TIMEOUT

    print_score(score, reason)

if __name__ == "__main__":
    main()
