import random
from pathlib import Path
from typing import Optional

import pygments
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer

from alpha_othello.circle_packing.initial import pack_26
from alpha_othello.database.database import Database
from alpha_othello.evaluate import DockerEvaluator, Evaluator, OthelloDockerEvaluator
from alpha_othello.llm import (
    PROMPT_VARIATIONS,
    extract_tagged_text,
    generate_prompt,
    get_llm_output,
)
from alpha_othello.othello.ai import (
    ai_egaroucid_easy,
    ai_egaroucid_hard,
    ai_egaroucid_med,
    ai_egaroucid_very_hard,
    ai_greedy,
    ai_random,
    get_function_source,
)


def get_api_key() -> str:
    secret_path = Path("secret.txt")
    with open(secret_path, "r") as f:
        api_key = f.read().strip()
    return api_key


def evolve(
    db: Database,
    llm: str,
    api_key: str,
    max_tokens: Optional[int],
    temperature: float,
    skeleton: str,
    task: str,
    topk: int,
    evaluator: Evaluator,
    prev: int = 0,
):
    """Apply an evolutionary step to the database.
        1. Get topk inspirations from previous completions
        2. Generate a prompt from the inspirations
        3. Pass the prompt to an LLM to generate a new reasoning/completion pair
        5. Score the new completion within a docker container
        6. Only store the completion if it is better than the previous ones

    Possible improvements:
        1. Use a bigger LLM to generate reasoning, then use a smaller LLM to generate the completion
        2. Use a bigger LLM to seed new explorative completions every n generations
        3. Use a bigger LLM to improve the skeleton/task
        4. Use explorative completion to separate islands of completions that evolve in parallel
        5. Decouple completion and scoring
        6. Change to diff models to generate completions
    """
    topk_completion_ids = db.get_topk_completion_ids(topk)
    if prev > 0:
        lastp_completion_ids = db.get_lastp_completion_ids(prev)
    else:
        lastp_completion_ids = []
    inspiration_ids = list(set(topk_completion_ids + lastp_completion_ids))
    inspirations = [db.get_completion(cid) for cid in inspiration_ids]
    score_dicts = [db.get_scores(cid) for cid in inspiration_ids]

    # Generate prompt
    n_completions = db.get_completion_count()
    metadata = {
        "n_completions": n_completions,
        "inspiration_ids": inspiration_ids,
    }
    variations_keys = list(PROMPT_VARIATIONS.keys())
    if n_completions < 20 or not variations_keys:
        variation = None
    else:
        variation = random.choice(variations_keys)
    prompt = generate_prompt(
        skeleton,
        inspirations,
        score_dicts,
        task,
        variation=variation,
        metadata=metadata,
    )
    print(f"Prompt:\n{prompt}")
    try:
        llm_output = get_llm_output(prompt, llm, api_key, max_tokens, temperature)
    except Exception as e:
        print(f"Error generating LLM output: {e}")
        llm_output = ""
    reasoning = extract_tagged_text(llm_output, "REASONING")
    completion = extract_tagged_text(llm_output, "COMPLETION")
    if not completion:
        print("No completion found")
        print("LLM output:")
        print(llm_output)
        return

    print(f"Reasoning:\n{reasoning}")
    highlighted = pygments.highlight(completion, PythonLexer(), TerminalFormatter())
    print(f"Completion:\n{highlighted}")

    score_dict = evaluator.evaluate(completion)
    completion_id = db.store_completion(completion, reasoning, inspiration_ids)
    db.store_scores(score_dict, completion_id)
    db.store_prompt(prompt, variation, completion_id)
    print(f"Scores: {score_dict}")


def main_othello():
    temperature = 0.7
    size = 8
    time_limit_ms = 999

    max_tokens = 2000
    topk_completions = 3
    api_key = get_api_key()

    skeleton_path = Path("src/alpha_othello/othello/skeleton.txt")
    with open(skeleton_path, "r") as f:
        skeleton = f.read()
    task = (
        f"Complete the ai() function body to return the best move on a {size}x{size} Othello board. "
        f"The clock represents the time left for each player to complete all of their remaining moves. "
        f"Each player starts with {time_limit_ms} milliseconds. If they run out of time, they lose."
    )

    db = Database(f"sqlite:///othello_{size}.db")

    if not db.get_topk_completion_ids(1):
        completion_id = db.store_completion(
            get_function_source(ai_random), "Randomly selects a move", []
        )
        db.store_score(0, "score", completion_id)

    # llm = "meta-llama/llama-3.3-8b-instruct:free"
    # llm = "deepseek/deepseek-chat-v3-0324:free"
    # llm = "google/gemini-2.0-flash-exp:free"
    # llm = "meta-llama/llama-4-maverick:free"
    # llm = "qwen/qwen3-235b-a22b:free"  # super slow
    llm = "google/gemma-3-27b-it:free"

    evaluator = OthelloDockerEvaluator(
        name="othello",
        docker_image="python-othello:latest",
        memory_limit="1g",
        cpu_limit="1",
        ais=[
            ai_random,
            ai_greedy,
            ai_egaroucid_easy,
            ai_egaroucid_med,
            ai_egaroucid_hard,
            ai_egaroucid_very_hard,
        ],
        eval_script_path=Path("src/alpha_othello/othello/eval.py"),
        n_games=50,
        size=size,
        time_limit_ms=time_limit_ms,
    )

    for i in range(1000):
        print(f"Generation {i}")
        evolve(
            db,
            llm,
            api_key,
            max_tokens,
            temperature,
            skeleton,
            task,
            topk_completions,
            evaluator,
        )


def main_circles():
    temperature = 0.7
    max_tokens = 2000
    topk = 2
    prev = 2
    api_key = get_api_key()

    skeleton_path = Path("src/alpha_othello/circle_packing/skeleton.txt")
    with open(skeleton_path, "r") as f:
        skeleton = f.read()
    task = """\
You are an expert mathematician specializing in circle packing problems and computational \
geometry. Your task is to improve a constructor function that directly produces a specific \
arrangement of 26 circles in a unit square, such that none of them overlap. The function \
should return a list of tuples, (x, y, r), where (x, y) is the center of a circle and r is \
its radius. The score will be the sum of the radii of all circles, which you should maximise. \
Invalid packings, where circles overlap or extend beyond the unit square, will score 0. \
Function completions which take longer than 5 minutes to run will also score 0. \
The current best score found by other researchers is 2.635. You can beat this. \
The Python environment has the following additional libraries available: numpy, scipy. \
"""

    evaluator = DockerEvaluator(
        name="circles",
        docker_image="circle-packing:latest",
        eval_script_path=Path("src/alpha_othello/circle_packing/eval.py"),
        memory_limit=None,
        cpu_limit=None,
    )

    db = Database("sqlite:///circles.db")

    if not db.get_topk_completion_ids(1):
        completion_str = get_function_source(pack_26)
        completion_id = db.store_completion(
            completion_str,
            "Place 26 circles in a very simple structured pattern with no overlaps.",
            [],
        )
        score_dict = evaluator.evaluate(completion_str)
        db.store_scores(score_dict, completion_id)

    llm = "google/gemma-3-27b-it:free"

    for i in range(200):
        print(f"Generation {i}")
        evolve(
            db,
            llm,
            api_key,
            max_tokens,
            temperature,
            skeleton,
            task,
            topk,
            evaluator,
            prev,
        )


if __name__ == "__main__":
    # main_othello()
    main_circles()
