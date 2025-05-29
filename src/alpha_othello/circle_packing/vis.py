import matplotlib.pyplot as plt


def plot_circle_packing(circles, ax=None):
    """
    Plots a circle packing given a list of circles.

    Parameters:
    - circles: List of tuples (x, y, radius) representing the circles.
    - ax: Matplotlib axis to plot on. If None, creates a new figure and axis.
    """
    if ax is None:
        fig, ax = plt.subplots()

    for x, y, r in circles:
        circle = plt.Circle((x, y), r, edgecolor="none", facecolor="blue", alpha=0.5)
        ax.text(
            x,
            y,
            str(circles.index((x, y, r))),
            color="red",
            ha="center",
            va="center",
            fontsize=8,
        )
        ax.add_artist(circle)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    score = sum(r for _, _, r in circles)
    plt.title(f"Score: {score:.2f}")
    plt.show()
    return ax
