# AlphaEvolve-style workers which can be instatiated by the main process.

import json

import requests


def generate_prompt(
    skeleton: str, inspirations: list[str], scores: list[int], task: str
) -> str:
    """Return a generated prompt using the skeleton and the code IDs.

    @param skeleton: The skeleton is the code that should be completed.
    @param inspirations: The inspirations are previously generated AI code that should be used to inspire the new AI.
    @param scores: The scores are the scores of the inspirations.
    @param task: The task is the task that the AI should complete.
    @return: The generated prompt. This will be passed to an LLM to generate new AI code.
    @rtype: str
    """
    preamble_str = """\
Act as an expert Python developer. Your job is to provide a completion for a code skeleton \
such that the code achieves the highest possible score on a task. The completion text will \
be appended to the end of the skeleton and the completed code will be scored.\
"""
    skeleton_str = f"<SKELETON>:\n{skeleton}\n</SKELETON>"
    inspiration_str = """\
Previously we found that the following completions performed well. You should aim to do \
better than these.\n\
"""
    inspiration_str += "\n\n".join(
        f"<COMPLETION_{i}>\n{completion}\n</COMPLETION_{i}>\n<SCORE_{i}>{score}</SCORE_{i}>"
        for i, (completion, score) in enumerate(zip(inspirations, scores))
    )
    task_str = f"<TASK>:\n{task}\n</TASK>"
    epilogue_str = """\
You should output your reasoning in between <REASONING> and </REASONING> tags, followed \
by the code completion in between <COMPLETION> and </COMPLETION> tags. You should not \
output any other text. The reasoning should be a step-by-step explanation of how the code \
will achieve the task to get the highest possible score. The completion should be valid \
Python code that will be appended to the skeleton. The completion will be inserted into a \
function body, so it should not repeat the function signature or docstring and it should \
start with an indented line. The completion should not contain any print statements. \
"""
    parts = [preamble_str, skeleton_str, inspiration_str, task_str, epilogue_str]
    if not inspirations:
        parts.pop(2)
    prompt = "\n\n".join(parts)
    return prompt


def extract_completion(llm_output: str) -> str:
    start = "<COMPLETION>"
    end = "</COMPLETION>"
    if start not in llm_output or end not in llm_output:
        return ""
    start_index = llm_output.index(start) + len(start)
    end_index = llm_output.index(end)
    code = llm_output[start_index:end_index].strip()
    return code


def get_llm_output(
    prompt: str, model_name: str, api_key: str, max_tokens: int
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
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        llm_output = response_data["choices"][0]["message"]["content"]
        return llm_output
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")
