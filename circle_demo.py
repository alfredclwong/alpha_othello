# %%
from alpha_othello.database.database import Database
from alpha_othello.evaluate import DockerEvaluator
from alpha_othello.llm import extract_tagged_text
from pathlib import Path
from alpha_othello.circle_packing.vis import plot_circle_packing
import matplotlib.pyplot as plt

# %%
packing_strs_path = Path("packings_latest.txt")

if packing_strs_path.exists():
    with open(packing_strs_path, "r") as f:
        packing_strs = [line.strip() for line in f.readlines()]
else:
    db = Database("sqlite:///circles.db")
    # completion_ids = db.get_all_completion_ids()
    completion_ids = db.get_topk_completion_ids(1, "SCORE")
    completions = [db.get_completion(cid) for cid in completion_ids]
    db.close()

    evaluator = DockerEvaluator(
        name="circle_test",
        docker_image="circle-packing",
        eval_script_path=Path("src/alpha_othello/circle_packing/eval.py"),
    )
    packing_strs = []
    score_strs = []

    try:
        for completion in completions:
            print(completion)
            completion_path = Path("/app/completion")
            evaluator._write(completion_path, completion)
            docker_stdout = evaluator._eval(["-c", str(completion_path), "-v"])
            packing = extract_tagged_text(docker_stdout, "PACKING")
            score_str = extract_tagged_text(docker_stdout, "SCORE")
            score_dict = {
                k: float(v) for k, v in (s.split(": ") for s in score_str.split(", "))
            }
            score = score_dict["SCORE"]
            if packing and score > 0:
                packing_strs.append(packing)
                score_strs.append(float(score))
                print(f"{score=}\n{packing=}")
    finally:
        del evaluator

# %%
packings = [
    [
        tuple(
            float(x[len("np.float64(") : -1])
            if x.startswith("np.float64(")
            else float(x)
            for x in circle_str[1:-1].split(", ")
        )
        for circle_str in packing_str[1:-1].replace("), (", "); (").split("; ")
    ]
    for packing_str in packing_strs
]
# Write packings to file
with open(packing_strs_path, "w") as f:
    for packing in packings:
        f.write(str(packing) + "\n")

# %%
for packing in packings:
    ax = plot_circle_packing(packing)
    # ax.set_title(f"Score: {sum(r for _, _, r in packing):.2f}")
    # ax.figure.savefig("circle_packing.png", dpi=300, bbox_inches="tight")
    plt.show()

# %%
