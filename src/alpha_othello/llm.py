import json
from typing import Optional, Any

import requests


PROMPT_VARIATIONS = {
    "": "",
    "tweak": (
        "Try to tweak the parameters used in the previous completions to improve the score. "
        "This might include changing the weights in a weighted sum, adjusting learning rates, "
        "or modifying other hyperparameters. The goal is to find a better configuration that "
        "leads to a higher score without changing the overall approach or algorithm significantly."
    ),
    "explore": (
        "Try to explore new ideas or approaches that might lead to a higher score. This could "
        "include changing the algorithm, introducing new techniques, or applying different "
        "strategies. The goal is to find a new direction that could lead to a better score, "
        "and to provide variety in the completions. This might involve taking risks or "
        "experimenting with unconventional methods."
    ),
    "simplify": (
        "Try to simplify the previous completions by removing unnecessary complexity or "
        "redundancy. This might include eliminating unused variables, reducing the number of "
        "functions, or streamlining the logic. The goal is to make the code more straightforward "
        "and easier to understand, which could lead to a higher score. Simplification can also "
        "improve performance by reducing the amount of code that needs to be executed."
    ),
    "refactor": (
        "Try to refactor the previous completions by improving the structure or organization of the code. "
        "This might include breaking down large functions into smaller ones, improving variable names, "
        "or reorganizing the code for better readability. The goal is to make the code cleaner and more maintainable, "
        "which could lead to a higher score."
    ),
    "fix": (
        "Try to fix any issues or bugs in the previous completions that might be causing low scores. "
        "This might include correcting logic errors, checking for edge cases, or addressing performance issues. "
        "The goal is to ensure that the code works correctly and efficiently, which could lead to a "
        "higher score."
    ),
    "optimize": (
        "Try to optimize the previous completions by improving performance or reducing resource usage. "
        "This might include optimizing algorithms, reducing memory usage, or improving execution speed. "
        "The goal is to make the code more efficient and effective, which could lead to a higher score."
    ),
    "combine": (
        "Try to combine the best parts of the previous completions to create a new, improved completion. "
        "This might include merging different approaches, integrating successful techniques, or synthesizing "
        "ideas from multiple completions. The goal is to leverage the strengths of previous attempts to achieve "
        "a higher score."
    ),
}


def generate_prompt(
    skeleton: str,
    inspirations: list[tuple[str, str]],
    score_dicts: dict[str, float],
    task: str,
    variation: Optional[str] = None,
    metadata: dict[str, Any] = {},
) -> str:
    # TODO add IDs such that LLM knows how many attempts it has made
    # TODO probabilistic prompt templates (e.g. tweak vs explore)
    """Return a generated prompt using the skeleton and the code IDs.

    @param skeleton: The skeleton is the code that should be completed.
    @param inspirations: The inspirations are previously generated AI code that should be used to inspire the new AI.
    @param scores: The scores are the scores of the inspirations.
    @param task: The task is the task that the AI should complete.
    @return: The generated prompt. This will be passed to an LLM to generate new AI code.
    @rtype: str
    """
    preamble_str = """\
Act as an expert Python developer. Your job is to provide the best possible completion of \
a code skeleton. The completion will be appended to the skeleton and the completed code \
will be scored on its ability to complete a task. Higher scores are better.\
"""
    task_str = f"<TASK>\n{task}\n</TASK>"
    skeleton_str = f"<SKELETON>\n{skeleton}\n</SKELETON>"

    # Sort inspirations in ascending order of scores
    # I noticed that the LLM reasons as if the inspirations are iterative improvements
    if not inspirations or not score_dicts:
        inspiration_str = ""
    else:
        inspiration_str = """\
To aid you with your task, we will provide you with some previous completions as inspirations. \
You should aim to achieve a higher score by making improvements and/or trying new ideas.\n\
"""
        if "inspiration_ids" in metadata:
            iids = metadata["inspiration_ids"]
            inspiration_data = zip(iids, inspirations, score_dicts)
        else:
            inspiration_data = zip(
                range(len(inspirations)), inspirations, score_dicts
            )
        inspiration_data = sorted(inspiration_data, key=lambda x: x[-1]["SCORE"])
        inspiration_str += "\n".join(
            f"<COMPLETION_{iid}>\n{completion}\n</COMPLETION_{iid}>\n"
            f"<REASONING_{iid}>\n{reasoning}\n</REASONING_{iid}>\n"
            f"<SCORE_{iid}>{score_dict}</SCORE_{iid}>"
            for (iid, (completion, reasoning), score_dict) in inspiration_data
        )
        if "n_completions" in metadata:
            n_completions = metadata["n_completions"]
            inspiration_str += f"\nYou are now generating completion #{n_completions+1}."

    variation_str = PROMPT_VARIATIONS.get(variation, "") if variation else ""
    epilogue_str = """\
Your output should consist of two parts: your reasoning for the completion and the completion itself. \
The reasoning should explain how this completion will improve upon previous iterations. \
The completion will be appended to the skeleton into a single function, so it should not \
repeat the function signature and it should start with one level of indentation. If you \
import libraries or define helper functions, make sure to do so within the scope of the \
function and before they are used. Make sure that the completion is valid Python code.

Your output should follow this format:
<REASONING>
{{reasoning}}
</REASONING>
<COMPLETION>
{{completion}}
</COMPLETION>
"""
    parts = [
        preamble_str,
        task_str,
        skeleton_str,
        inspiration_str,
        variation_str,
        epilogue_str,
    ]
    parts = [s for s in parts if s.strip()]
    prompt = "\n\n".join(parts)
    return prompt


def extract_tagged_text(llm_output: str, tag: str) -> Optional[str]:
    start = f"<{tag}>"
    end = f"</{tag}>"
    if start not in llm_output or end not in llm_output:
        return None
    start_index = llm_output.index(start) + len(start)
    end_index = llm_output.index(end)
    text = llm_output[start_index:end_index]
    return text


def get_llm_output(
    prompt: str,
    model_name: str,
    api_key: str,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
    except requests.RequestException as e:
        raise Exception(f"Error while calling LLM API: {e}")
    if response.status_code == 200:
        response_data = response.json()
        try:
            llm_output = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise Exception("Error: Unexpected response format from LLM", response_data)
        return llm_output
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")
