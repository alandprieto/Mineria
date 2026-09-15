import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _radar_axis(ax, values, labels, color, label):
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    vals = list(values) + list(values[:1])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, vals, color=color, linewidth=1.5, label=label)
    ax.fill(angles, vals, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title(label, fontsize=11, pad=12)


def radar_profiles(profile_df: pd.DataFrame, title: str = "Perfiles por clúster"):
    """Gráfico radar: profile_df = filas variables, columnas clústeres."""
    labels = list(profile_df.index)
    cols = list(profile_df.columns)
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(
        1, len(cols), subplot_kw={"polar": True}, figsize=(5.2 * len(cols), 4.6)
    )
    axes = np.atleast_1d(axes)
    for i, (ax, col) in enumerate(zip(axes, cols)):
        _radar_axis(ax, profile_df[col].values, labels, cmap(i), col)
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def elbowa_plot(scores: pd.DataFrame, title: str = ""):
    """scores = DataFrame con columnas k, inercia, db_score."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(scores["k"], scores["inercia"], marker="o")
    axes[0].set_title("Curva del codo (inercia)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inercia intra-cluster")
    axes[1].plot(scores["k"], scores["db_score"], marker="o", color="orange")
    axes[1].set_title("Davies-Bouldin Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("DB")
    if title:
        fig.suptitle(title, y=1.03)
    plt.tight_layout()
    return fig