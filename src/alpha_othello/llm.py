import json
from typing import Optional

import requests


def generate_prompt(
    skeleton: str, inspirations: list[str], scores: list[int], task: str
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
    skeleton_str = f"<SKELETON>:\n{skeleton}\n</SKELETON>"
    task_str = f"<TASK>:\n{task}\n</TASK>"
    # Sort inspirations in ascending order of scores
    # I noticed that the LLM reasons as if the inspirations are iterative improvements
    inspiration_str = """\
Previously we found that the following completions scored highly. You should aim to achieve \
a higher score by making improvements and/or trying new ideas.\n\
"""
    inspiration_str += "\n".join(
        f"<COMPLETION_{i}>\n{completion}\n</COMPLETION_{i}>\n<SCORE_{i}>{score}</SCORE_{i}>"
        for i, (completion, score) in enumerate(
            sorted(zip(inspirations, scores), key=lambda x: x[1])
        )
    )
    epilogue_str = """\
Output your reasoning in between <REASONING> and </REASONING> tags, followed by the code \
completion in between <COMPLETION> and </COMPLETION> tags. Do not output any other text. \
The reasoning should explain how this completion will improve upon previous iterations. \
The completion will be appended to the skeleton into a single function, so it should not \
repeat the function signature and it should start with one level of indentation. If you \
import libraries or define helper functions, make sure to do so within the scope of the \
function. Make sure that the completion is valid Python code.\
"""
    parts = [preamble_str, task_str, skeleton_str, inspiration_str, epilogue_str]
    if not inspirations:
        parts.pop(2)
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
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        try:
            llm_output = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise Exception("Error: Unexpected response format from LLM", response_data)
        return llm_output
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")
