# %%
import colorsys

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

from alpha_othello.othello.ai import (
    ai_greedy,
    ai_heuristic,
    ai_minimax,
    ai_pass,
    ai_random,
)
from alpha_othello.othello.game import GameOverReason, GameResult, run_tournament

# %%
ais = [ai_random, ai_greedy, ai_minimax, ai_heuristic]
results = run_tournament(ais, size=6, n_games_per_pair=1000, time_control_millis=20)

# %%
df = pd.DataFrame(
    [
        {"AI 1": ai1, "AI 2": ai2, "Result": result, "Reason": reason, "Count": count}
        for (ai1, ai2), result_dict in results.items()
        for result, reason_dict in result_dict.items()
        for reason, count in reason_dict.items()
    ]
)
pairs = sorted(results.keys())


def hue_to_hex(hue, lightness=0.6, saturation=0.7):
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    return "#%02x%02x%02x" % tuple(int(255 * x) for x in rgb)


# Assign a base hue for each result
unique_results = sorted(df["Result"].unique())
result_hues = {
    result: i / len(unique_results) for i, result in enumerate(unique_results)
}
black_color = hue_to_hex(result_hues[GameResult.BLACK_WINS.value])
white_color = hue_to_hex(result_hues[GameResult.WHITE_WINS.value])
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
    result_counts = pair_df.groupby(["Result", "Reason"])["Count"].sum().reset_index()
    result_counts = result_counts.sort_values(by=["Result", "Reason"])
    fig.add_trace(
        go.Pie(
            labels=result_counts.apply(
                lambda x: f"{x['Result']} ({x['Reason']})", axis=1
            ),
            marker=dict(
                colors=[
                    hue_to_hex(
                        result_hues[x["Result"]], reason_lightnesses[x["Reason"]]
                    )
                    for _, x in result_counts.iterrows()
                ],
            ),
            values=result_counts["Count"],
            name=f"{ai1} vs {ai2}",
            hole=0.4,
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
