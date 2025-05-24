# %%
import colorsys
from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from multiprocess import Pool
from othello.game import Game
from othello.types import T_PLAYER_FN, Player
from scipy.optimize import minimize
from tqdm.auto import tqdm, trange

from alpha_othello.database.database import Database
from alpha_othello.othello.ai import (
    ai_egaroucid_easy,
    ai_egaroucid_med,
    ai_egaroucid_hard,
    _ai_egaroucid,
    ai_greedy,
    ai_heuristic,
    ai_minimax,
    ai_mobility,
    ai_parity,
    ai_random,
)

# %%
# # Get topk completions from the database
# db = Database("sqlite:///othello_6.db")
# topk = 10
# topk_ids = db.get_topk_completion_ids(topk)
# topk_completions = [db.get_completion(completion_id) for completion_id in topk_ids]

# # Write the completions to a file
# with open("topk_ais.py", "w") as f:
#     f.write("""\
# import numpy as np
# from othello.state import (
#     get_flips,
#     get_legal_squares,
#     get_size,
#     is_empty,
#     is_legal_square,
# )
# from othello.types import T_BOARD, T_CLOCK, T_SQUARE, Player


# """)
#     for id, completion in zip(topk_ids, topk_completions):
#         code = f"def ai_{id}(board: T_BOARD, player: Player, clock: T_CLOCK) -> T_SQUARE:\n"
#         code += completion
#         code += "\n\n"
#         f.write(code)


# from topk_ais import *

# %%
from alpha_othello.evaluate import OthelloDockerEvaluator
from pathlib import Path
from alpha_othello.othello.ai import get_function_source

evaluator = OthelloDockerEvaluator(
    name="test",
    docker_image="python-othello",
    memory_limit="1g",
    cpu_limit="1",
    ais=[ai_random, ai_greedy, ai_minimax, ai_heuristic, ai_egaroucid_easy, ai_egaroucid_med, ai_egaroucid_hard],
    eval_script_path=Path("src/alpha_othello/othello/eval.py"),
    n_games=50,
    size=8,
    time_limit_ms=999,
)

evaluator.evaluate(get_function_source(ai_egaroucid_hard))

# %%
del evaluator

# %%
def game_worker(ai1, ai2, size, time_limit_ms, batch_size=100):
    results = Counter()
    for _ in range(batch_size):
        game = Game((ai1, ai2), size=size, time_limit_ms=time_limit_ms)
        game.play()
        winner = "Draw" if game.winner is None else game.winner.name
        reason = game.reason.name
        results[(winner, reason)] += 1
    return results


def run_tournament(
    ais: dict[str, T_PLAYER_FN],
    size: int = 6,
    n_games_per_pair: int = 100,
    time_limit_ms: int = 10,
):
    n_games_total = len(ais) * len(ais) * n_games_per_pair
    pbar = tqdm(total=n_games_total, desc="Running tournament")

    pairs = [(i, j) for i in ais.keys() for j in ais.keys()]
    results = {}

    def update(result, pair):
        results[pair] = result
        pbar.update(n_games_per_pair)

    with Pool() as pool:
        async_results = []
        for pair in pairs:
            args = (ais[pair[0]], ais[pair[1]], size, time_limit_ms, n_games_per_pair)
            async_result = pool.apply_async(
                game_worker, args, callback=partial(update, pair=pair)
            )
            async_results.append(async_result)
        # Wait for all tasks to finish
        for async_result in async_results:
            async_result.wait()
    return results


ais = {}
ais |= {
    f"({depth}, {final_depth})": partial(_ai_egaroucid, depth=depth, final_depth=final_depth)
    for depth in [2, 4, 8, 16]
    for final_depth in [2, 4, 8, 16]
    if depth <= final_depth
}
ais |= {
#     "random": ai_random,
    "greedy": ai_greedy,
#     "minimax": ai_minimax,
#     # "mobility": ai_mobility,
#     # "parity": ai_parity,
    "heuristic": ai_heuristic,
#     "egaroucid": ai_egaroucid,
}
# ais |= {f"ai_{id}": globals()[f"ai_{id}"] for id in topk_ids}
print(ais)

results = run_tournament(ais, size=8, n_games_per_pair=10, time_limit_ms=9999)

# %%
df = pd.DataFrame(
    [
        {
            "AI 1": ai1,
            "AI 2": ai2,
            "Winner": winner,
            "Reason": reason,
            "Count": count,
        }
        for (ai1, ai2), result in results.items()
        for (winner, reason), count in result.items()
    ]
)
df

# %%
# Calculate a df where each row is a (ai1, ai2) pair with counts for each result
result_counts = df.groupby(["AI 1", "AI 2", "Winner"])["Count"].sum().reset_index()
result_counts = (
    result_counts.pivot(index=["AI 1", "AI 2"], columns="Winner", values="Count")
    .fillna(0)
    .reset_index()
)


# Fit ratings to the results using maximum likelihood estimation
def sigmoid(x, k=1):
    return 1 / (1 + np.exp(-x / k))


def log_likelihood(ratings, result_counts, k=400 / np.log(10)):
    # Calculate the log likelihood of the ratings given the result counts
    log_likelihood = 0
    for _, row in result_counts.iterrows():
        ai1, ai2 = row["AI 1"], row["AI 2"]
        black_count, white_count, draw_count = (
            row.get("BLACK", 0),
            row.get("WHITE", 0),
            row.get("Draw", 0),
        )
        total_count = black_count + white_count + draw_count
        if total_count == 0:
            continue
        p_black = sigmoid(ratings[ai1] - ratings[ai2], k=k)
        p_white = sigmoid(ratings[ai2] - ratings[ai1], k=k)
        # p_draw = 1 - p_black - p_white
        log_likelihood += (
            black_count * np.log(p_black + 1e-9) + white_count * np.log(p_white + 1e-9)
            # + draw_count * np.log(p_draw + 1e-9)
        ) / total_count
    return log_likelihood


ais = list(set(result_counts["AI 1"].unique()) | set(result_counts["AI 2"].unique()))
ratings = {ai: 1500.0 for ai in result_counts["AI 1"].unique()}

# Use scipy optimize to find the best ratings
# Track the log likelihood and ratings history
rating_history = {ai: [] for ai in ais}
log_likelihood_history = []

# Initial ratings vector (one per AI)
init_ratings = np.full(len(ais), 1500.0)
ai_idx = {ai: idx for idx, ai in enumerate(ais)}


def ratings_dict_from_vec(vec):
    return {ai: vec[ai_idx[ai]] for ai in ais}


def neg_log_likelihood_vec(vec):
    ratings = ratings_dict_from_vec(vec)
    return -log_likelihood(ratings, result_counts)


def callback(vec):
    ratings = ratings_dict_from_vec(vec)
    for ai in ais:
        rating_history[ai].append(ratings[ai])
    ll = log_likelihood(ratings, result_counts)
    log_likelihood_history.append(ll)
    pbar.set_postfix({"log_likelihood": ll})
    pbar.update(1)


n_iter = 100
pbar = trange(n_iter, desc="Optimizing ratings")
res = minimize(
    neg_log_likelihood_vec,
    init_ratings,
    callback=callback,
    jac="2-point",
    options={"maxiter": n_iter, "disp": False, "gtol": 1e-6, "eps": 1e-3},
)
pbar.close()

ratings = ratings_dict_from_vec(res.x)

# Plot log likelihood history
fig_ll = go.Figure()
fig_ll.add_trace(
    go.Scatter(y=log_likelihood_history, mode="lines+markers", name="Log Likelihood")
)
fig_ll.update_layout(
    title="Log Likelihood History",
    xaxis_title="Iteration",
    yaxis_title="Log Likelihood",
)
fig_ll.show()

# Plot rating history for each AI
fig_ratings = go.Figure()
for ai in ais:
    fig_ratings.add_trace(
        go.Scatter(y=rating_history[ai], mode="lines+markers", name=ai)
    )
fig_ratings.update_layout(
    title="Rating History", xaxis_title="Iteration", yaxis_title="Rating"
)
fig_ratings.show()
ratings

# %%
# Save plot
fig_ratings.write_html("ratings_history.html")

# Save ratings
ratings_df = pd.DataFrame(ratings.items(), columns=["AI", "Rating"])
ratings_df.to_csv("ratings.csv", index=False)


# %%
def hue_to_hex(hue, lightness=0.6, saturation=0.7):
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    return "#%02x%02x%02x" % tuple(int(255 * x) for x in rgb)


# Assign a base hue for each result
unique_results = sorted(df["Winner"].unique())
result_hues = {
    Player.BLACK.name: 0.0,
    "Draw": 1 / 3,
    Player.WHITE.name: 2 / 3,
}
black_color = hue_to_hex(result_hues[Player.BLACK.name])
white_color = hue_to_hex(result_hues[Player.WHITE.name])

pairs = sorted(results.keys())
subplot_titles = [
    f"<span style='color:{black_color}'>{ai1}</span> vs <span style='color:{white_color}'>{ai2}</span>"
    for ai1, ai2 in pairs
]

unique_reasons = df["Reason"].unique()
n_reasons = len(unique_reasons)
reason_lightnesses = {
    reason: 0.5 + i / (2 * n_reasons) for i, reason in enumerate(unique_reasons)
}

# Create subplots for each AI pair
n_cols = 3
n_rows = (len(pairs) + n_cols - 1) // n_cols
fig = sp.make_subplots(
    rows=n_rows,
    cols=n_cols,
    subplot_titles=subplot_titles,
    specs=[[{"type": "pie"}] * n_cols for _ in range(n_rows)],
)

for i, (ai1, ai2) in enumerate(pairs):
    pair_df = df[(df["AI 1"] == ai1) & (df["AI 2"] == ai2)]
    result_counts = pair_df.groupby(["Winner", "Reason"])["Count"].sum().reset_index()
    result_counts = result_counts.sort_values(by=["Winner", "Reason"])
    labels = result_counts.apply(
        lambda x: f"{x['Winner']} ({x['Reason']})",
        axis=1,
    )
    colors = result_counts.apply(
        lambda x: hue_to_hex(
            result_hues[x["Winner"]],
            lightness=reason_lightnesses[x["Reason"]],
        ),
        axis=1,
    )
    fig.add_trace(
        go.Pie(
            labels=labels,
            marker={"colors": colors},
            values=result_counts["Count"],
            name=f"{ai1} vs {ai2}",
            hole=0.4,
            sort=False,
        ),
        row=i // n_cols + 1,
        col=i % n_cols + 1,
    )

fig.update_layout(
    title_text="Tournament Results",
    showlegend=True,
    height=300 * n_rows,
)
fig.show()

# %%
