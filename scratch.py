# %%
from alpha_othello.othello.ai import ai_random, ai_greedy, ai_pass, ai_minimax, ai_heuristic
from alpha_othello.othello.game import run_tournament
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import colorsys

# %%
ais = [ai_random, ai_greedy, ai_minimax, ai_heuristic]
results = run_tournament(ais, size=6, n_games_per_pair=100, time_control_millis=50)

# %%
# Prepare data for visualization
data = [
    {
        "AI 1": ai1,
        "AI 2": ai2,
        "Result": result,
        "Reason": reason,
        "Count": count
    }
    for (ai1, ai2), result_dict in results.items()
    for result, reason_dict in result_dict.items()
    for reason, count in reason_dict.items()
]
df = pd.DataFrame(data)

# Assign a base hue for each result
unique_results = sorted(df["Result"].unique())
result_hues = {result: i / len(unique_results) for i, result in enumerate(unique_results)}

# Create subplots for each AI pair
n_cols = 3
pairs = sorted(results.keys())
n_rows = (len(pairs) + n_cols - 1) // n_cols
fig = sp.make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=[f"{ai1} vs {ai2}" for ai1, ai2 in pairs],
    specs=[[{"type": "pie"}] * n_cols for _ in range(n_rows)],
)

for i, (ai1, ai2) in enumerate(pairs):
    pair_df = df[(df["AI 1"] == ai1) & (df["AI 2"] == ai2)]
    result_counts = pair_df.groupby(["Result", "Reason"])["Count"].sum().reset_index()
    result_counts = result_counts.sort_values(by=["Result", "Reason"])

    # Assign colors: same result = same hue, different reasons = different lightness
    reasons = result_counts["Reason"].unique()
    n_reasons = len(reasons)
    colors = []
    for _, row in result_counts.iterrows():
        base_hue = result_hues[row["Result"]]
        reason_idx = list(reasons).index(row["Reason"])
        lightness = 0.5 + 0.2 * (reason_idx / max(1, n_reasons - 1))
        rgb = colorsys.hls_to_rgb(base_hue, lightness, 0.7)
        hex_color = '#%02x%02x%02x' % tuple(int(255 * x) for x in rgb)
        colors.append(hex_color)

    fig.add_trace(
        go.Pie(
            labels=result_counts.apply(lambda x: f"{x['Result']} ({x['Reason']})", axis=1),
            values=result_counts["Count"],
            name=f"{ai1} vs {ai2}",
            hole=0.4,
            marker=dict(colors=colors)
        ),
        row=i // n_cols + 1, col=i % n_cols + 1
    )

fig.update_layout(
    title_text="Tournament Results",
    showlegend=True,
    height=300 * n_rows,
)
fig.show()

# %%
