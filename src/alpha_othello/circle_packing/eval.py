# Task:
# Choose circle (c, r) packing within a unit square that maximises the sum
# of the radii of the circles, while ensuring that no two circles overlap

import argparse
from pathlib import Path


def is_valid(
    packing: list[tuple[float, float, float]], tolerance: float = 1e-6
) -> bool:
    # type checks
    if not isinstance(packing, list):
        print("Packing must be a list.")
        return False
    if len(packing) != 26:
        print("Packing must contain exactly 26 circles.")
        return False
    if not all(isinstance(circle, tuple) and len(circle) == 3 for circle in packing):
        print("Each circle must be a tuple of (x, y, r).")
        return False
    if not all(
        isinstance(x, (int, float)) and x > 0 for circle in packing for x in circle
    ):
        print("Circle coordinates and radius must be positive numbers.")
        return False

    # check if any circle is outside the unit square
    for x, y, r in packing:
        if not (
            0 <= x - r <= 1 and 0 <= y - r <= 1 and 0 <= x + r <= 1 and 0 <= y + r <= 1
        ):
            return False

    # check if any two circles overlap
    for i in range(len(packing)):
        for j in range(i + 1, len(packing)):
            x1, y1, r1 = packing[i]
            x2, y2, r2 = packing[j]
            distance_squared = (x1 - x2) ** 2 + (y1 - y2) ** 2
            radius_sum = r1 + r2
            if distance_squared < (radius_sum - tolerance) ** 2:
                return False

    return True


def score_packing(packing: list[tuple[float, float, float]]) -> float:
    """Calculate the score of a packing as the sum of the radii of the circles."""
    return sum(r for _, _, r in packing)


def init_parser():
    parser = argparse.ArgumentParser(description="Evaluate circle packing solutions.")
    parser.add_argument(
        "-c", "--completion", type=str, required=True, help="Completion code path"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    return parser


def print_score(score: float):
    print(f"<SCORE>{score}</SCORE>")


def main():
    parser = init_parser()
    args = parser.parse_args()
    completion_path = Path(args.completion)
    verbose = args.verbose

    with open(completion_path, "r") as f:
        completion_str = f.read()

    local_vars = {}
    exec(f"def pack_26():\n{completion_str}", globals(), local_vars)
    packing = local_vars["pack_26"]()

    if verbose:
        print(f"<PACKING>{packing}</PACKING>")

    if not is_valid(packing):
        print_score(0.0)
    else:
        score = score_packing(packing)
        print_score(score)


if __name__ == "__main__":
    main()
