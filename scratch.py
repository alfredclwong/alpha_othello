# %%
import colorsys
from collections import Counter
from functools import partial

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from multiprocess import Pool
from othello.game import Game
from othello.types import T_PLAYER_FN, Player
from scipy.optimize import minimize
from tqdm.auto import tqdm, trange
from bokeh.models import Circle, HoverTool
from bokeh.plotting import figure, show, from_networkx, ColumnDataSource
from bokeh.io import output_notebook, output_file
from bokeh.models import CustomJS, Div, LabelSet
from bokeh.layouts import row, column

from alpha_othello.database.database import Database
from alpha_othello.othello.ai import (
    _ai_egaroucid,
    ai_egaroucid_easy,
    ai_egaroucid_hard,
    ai_egaroucid_med,
    ai_egaroucid_very_hard,
    ai_greedy,
    ai_heuristic,
    ai_human,
    ai_minimax,
    ai_mobility,
    ai_parity,
    ai_random,
)

# %%
# # Get topk completions from the database
# db = Database("sqlite:///othello_8.db")
# topk = 1
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
#         code += f"    # Score: {db.get_score(id)}\n"
#         code += completion
#         code += "\n\n"
#         f.write(code)
from topk_ais import *

# %%
db_path = "sqlite:///othello_8.db"
db = Database(db_path)
completion_ids = db.get_all_completion_ids()
db_df = pd.DataFrame(
    {
        "Completion ID": completion_ids,
        "Completion": [db.get_completion(id) for id in completion_ids],
        "Reasoning": [db.get_reasoning(id) for id in completion_ids],
        "Inspirations": [db.get_inspirations(id) for id in completion_ids],
        "Score": [db.get_score(id) for id in completion_ids],
    }
)
# Filter out completions which were below the topk threshold at any point in time
k = 3
topk_scores = db_df["Score"].expanding().apply(lambda x: pd.Series(x).nlargest(k).min())
db_df = db_df[db_df["Score"] >= topk_scores]
db_df


# %%
# Each node is a completion, and its parents are its inspirations
def create_evolutionary_tree(df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in df.iterrows():
        completion_id = row["Completion ID"]
        G.add_node(
            completion_id,
            score=row["Score"],
            size=max(10, row["Score"] * 0.1) * 2,
            completion=row["Completion"],
        )
        for inspiration_id in row["Inspirations"]:
            if inspiration_id:
                G.add_edge(inspiration_id, completion_id)
    return G


G = create_evolutionary_tree(db_df)
print(G.nodes(data=True))

# %%
# Use graphviz_layout with 'dot' for hierarchical layout, then jitter overlapping nodes
pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")

# Jitter positions to avoid overlapping nodes
def jitter_positions(pos, min_dist=10) -> dict:
    # Convert to array for easier manipulation
    coords = np.array(list(pos.values()))
    keys = list(pos.keys())
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < min_dist:
                # Move them apart slightly
                direction = coords[i] - coords[j]
                if np.all(direction == 0):
                    direction = np.random.randn(2)
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                shift = direction * (min_dist - dist) / 2
                coords[i] += shift
                coords[j] -= shift
    return {k: tuple(coords[i]) for i, k in enumerate(keys)}

pos = jitter_positions(pos, min_dist=100)

# Use Bokeh to visualise the graph
# The nodes are sized by their score, with larger nodes having higher scores
# A text box on the side of the graph shows the full completion text when hovering over a node
output_notebook()
height = 1500
plot = figure(
    width=150,
    height=height,
    title="Evolutionary Tree",
)
plot.xgrid.grid_line_color = None
plot.ygrid.grid_line_color = None
plot.axis.visible = False
plot.background_fill_color = "#f0f0f0"
plot.toolbar_location = None

# Text box for completion text
completion_div = Div(
    text="Hover over a node to see the completion.",
    width=550,
    height=height,
    styles={
        "overflow-x": "auto",
        "overflow-y": "auto",
        "margin-top": "0px",
        "background": "#f9f9f9",
        "padding": "2px",
        "font-size": "10px",
    },
)

layout = row(plot, completion_div)

# Data for nodes
hover_color = "#ffcc00"  # Color for hovered node
default_color = "lightblue"  # Default color for nodes
source = ColumnDataSource(
    data={
        "index": list(G.nodes),
        "score": [G.nodes[n]["score"] for n in G.nodes],
        "size": [G.nodes[n]["size"] for n in G.nodes],
        "completion": [G.nodes[n]["completion"] for n in G.nodes],
        "fill_color": [default_color] * len(G.nodes),
    }
)

nodes = from_networkx(G, pos, scale=1, center=(0, 0))
nodes.node_renderer.data_source = source
nodes.node_renderer.glyph = Circle(
    radius="size", fill_color="fill_color", line_color="black",
)

# Add a callback to update the Div with the completion text when hovering
def display_completion() -> CustomJS:
    return CustomJS(
        args=dict(source=source, div=completion_div),
        code=f"""
            const {{indices}} = cb_data.index;
            if (indices.length > 0) {{
                const idx = indices[0];

                // Change color of hovered node
                const node_renderer = cb_obj.renderers[0];
                const node_data = node_renderer.data_source;
                // Reset all nodes to lightblue
                for (let i = 0; i < node_data.data['index'].length; i++) {{
                    node_data.data['fill_color'] = node_data.data['fill_color'] || [];
                    node_data.data['fill_color'][i] = "{default_color}";
                }}
                // Highlight hovered node
                node_data.data['fill_color'][idx] = "{hover_color}";
                node_data.change.emit();

                const id = source.data['index'][idx];
                const score = source.data['score'][idx];
                const completion = source.data['completion'][idx];
                div.text = "<b>Completion #" + id + ":</b><br>";
                div.text += "[Score: " + score + "]<br>";
                div.text += "<pre style='white-space:pre-wrap;'><code class='language-python'>";
                div.text += completion;
                div.text += "</code></pre>";

            }}
        """,
    )


hover_tool = HoverTool(
    tooltips=[
        ("Completion ID", "@index"),
        ("Score", "@score"),
    ],
    renderers=[nodes.node_renderer],
    callback=display_completion(),
)

# Manually trigger the callback for node 277 (if it exists)
top_completion_id = db.get_topk_completion_ids(1)[0]
if top_completion_id in source.data["index"]:
    idx = source.data["index"].index(top_completion_id)
    # Simulate hover event by updating the Div
    completion_id = source.data["index"][idx]
    score = source.data["score"][idx]
    completion = source.data["completion"][idx]
    completion_div.text = (
        f"<b>Completion #{completion_id}:</b><br>"
        f"[Score: {score}]<br>"
        "<pre style='white-space:pre-wrap;'><code class='language-python'>"
        f"{completion}"
        "</code></pre>"
    )
    source.data["fill_color"][idx] = hover_color

plot.add_tools(hover_tool)
plot.renderers.append(nodes)

# Layout
show(layout)
output_file("evolutionary_tree.html")

# %%
db_df["Score"].plot(backend="plotly")

# %%
from pathlib import Path

from alpha_othello.evaluate import OthelloDockerEvaluator

evaluator = OthelloDockerEvaluator(
    name="test",
    docker_image="python-othello:latest",
    memory_limit="1g",
    cpu_limit="1",
    ais=[
        ai_random,
        ai_greedy,
        ai_egaroucid_easy,
        ai_egaroucid_hard,
        ai_egaroucid_med,
        ai_egaroucid_very_hard,
    ],
    eval_script_path=Path("src/alpha_othello/othello/eval.py"),
    n_games=50,
    size=8,
    time_limit_ms=9999,
)
score = evaluator.evaluate(topk_completions[0])
print(f"{score=}")
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
    f"({depth}, {final_depth})": partial(
        _ai_egaroucid, depth=depth, final_depth=final_depth
    )
    # for depth in [2, 4, 8, 16]
    # for final_depth in [2, 4, 8, 16]
    # if depth <= final_depth
    for depth, final_depth in [
        # (2, 2),
        # # (2, 4),
        # (4, 8),
        # # (8, 8),
        # (6, 12),
        # (8, 16),
    ]
}
ais |= {
    # "random": ai_random,
    # "greedy": ai_greedy,
    # "egaroucid_easy": ai_egaroucid_easy,
    "egaroucid_med": ai_egaroucid_med,
    "egaroucid_hard": ai_egaroucid_hard,
    "egaroucid_very_hard": ai_egaroucid_very_hard,
    #     "minimax": ai_minimax,
    #     # "mobility": ai_mobility,
    #     # "parity": ai_parity,
    # "heuristic": ai_heuristic,
    #     "egaroucid": ai_egaroucid,
}
ais |= {f"ai_{id}": globals()[f"ai_{id}"] for id in topk_ids}
print(ais)

results = run_tournament(ais, size=8, n_games_per_pair=50, time_limit_ms=9999)

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
ratings_df = pd.DataFrame(
    {
        "AI": list(ratings.keys()),
        "Rating": list(ratings.values()),
    }
).sort_values(by="Rating", ascending=False)

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

ratings_df


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
