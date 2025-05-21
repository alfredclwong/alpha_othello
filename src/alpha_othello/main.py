# Components
#  1. Prompt generator
#  2. AI generator
#  3. AI evaluator
#  4. Results database
# Routing
#  1. 4 -> 1: Use results (and skeleton) to generate an evolutionary prompt
#  2. 1 -> 2: Use prompt to generate an AI (and reasoning)
#  3. 2 -> 3: Score the AI
#  4. 3 -> 4: Store AI scores (and reasoning)
# For now, don't generate reasoning

import inspect
from pathlib import Path

from othello.types import Player

from alpha_othello.database import SQLiteDatabase
from alpha_othello.othello.ai import ai_greedy, ai_heuristic, ai_random
from alpha_othello.othello.docker import play_in_docker
from alpha_othello.workers import (
    extract_completion,
    generate_prompt,
    get_llm_output,
)


def main():
    size = 6
    max_tokens = 2000
    n_generations_per_llm = 3

    secret_path = Path("secret.txt")
    with open(secret_path, "r") as f:
        api_key = f.read().strip()

    skeleton_path = Path("src/alpha_othello/othello/skeleton.txt")
    with open(skeleton_path, "r") as f:
        skeleton = f.read()

    task = f"Implement the AI function to play the best possible move in {size}x{size} Othello."

    db = SQLiteDatabase("sqlite:///othello.db")

    llm_ids = [
        db.store_llm("meta-llama/llama-3.3-8b-instruct:free"),
    ]

    ais = [
        ai_random,
        ai_greedy,
        ai_heuristic,
    ]

    for ai in ais:
        code = inspect.getsource(ai)
        completion = "\n".join(code.splitlines()[1:])
        completion_id = db.store_completion(completion, 1, 1)
        db.store_score(0, completion_id)

    k = 2
    top_completion_ids = db.get_topk_completions(k)
    inspirations = [
        db.get_completion(completion_id) for completion_id in top_completion_ids
    ]
    scores = [db.get_score(completion_id) for completion_id in top_completion_ids]
    inspirations, scores = zip(
        *[
            (insp, score)
            for insp, score in zip(inspirations, scores)
            if insp is not None and score is not None
        ]
    )
    inspirations = list(inspirations)
    scores = list(scores)
    prompt = generate_prompt(skeleton, inspirations, scores, task)
    prompt_id = db.store_prompt(prompt, top_completion_ids)

    print(prompt)

    for llm_id in llm_ids:
        llm = db.get_llm(llm_id)
        if llm is None:
            continue

        for _ in range(n_generations_per_llm):
            # llm_output = get_llm_output(prompt, llm, api_key, max_tokens)
            # completion = extract_completion(llm_output)
            completion = db.get_completion(3)
            if completion is None:
                continue
            completion_id = db.store_completion(completion, llm_id, prompt_id)

            opponent_completion = db.get_completion(1)
            if opponent_completion is None:
                continue

            results_as_black = play_in_docker(
                completion,
                opponent_completion,
                n_games=100,
                size=size,
                time_control_millis=20,
            )

            results_as_white = play_in_docker(
                opponent_completion,
                completion,
                n_games=100,
                size=size,
                time_control_millis=20,
            )

            score = 0
            for (result, reason), count in results_as_black.items():
                if result == str(Player.BLACK):
                    score += count
                elif result == str(Player.WHITE):
                    score -= count
                elif result == "None":
                    score += 0

            for (result, reason), count in results_as_white.items():
                if result == str(Player.BLACK):
                    score -= count
                elif result == str(Player.WHITE):
                    score += count
                elif result == "None":
                    score += 0

            score_id = db.store_score(score, completion_id)


if __name__ == "__main__":
    main()
