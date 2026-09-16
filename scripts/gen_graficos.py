import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import davies_bouldin_score

from mineria.clustering import PLAYER_FIELD_COLS, TEAM_TACTIC_COLS, standardize
from mineria.config import DB_PATH
from mineria.data_loader import load_table

sns.set_theme(style="whitegrid")
PAL = sns.color_palette("Set2")
OUT = "graficos"

club_names = {
    "Conservador": "Club Brugge KV",
    "Contraataque": "Sporting Charleroi",
    "Ofensivo": "Real Sociedad",
}

player_names = {
    "Delantero": "Stefan Nijland",
    "Mediocampista": "Daniele Dessena",
    "Defensor": "Ryan McGivern",
}

RADAR_FIELDS = [
    "finishing",
    "dribbling",
    "short_passing",
    "vision",
    "marking",
    "standing_tackle",
    "sprint_speed",
    "stamina",
]

RADAR_LABELS = [
    "Remate",
    "Regate",
    "Pase corto",
    "Visión",
    "Marca",
    "Entrada firme",
    "Vel. punta",
    "Resistencia",
]


def player_clusters():
    pa = load_table("Player_Attributes", DB_PATH)
    players = load_table("Player", DB_PATH).set_index("player_api_id")["player_name"]
    df = pa[pa["gk_diving"] < 30][["player_api_id"] + PLAYER_FIELD_COLS].dropna()
    df["player_name"] = df["player_api_id"].map(players)
    sample = df.sample(n=20000, random_state=42)
    X = standardize(sample, PLAYER_FIELD_COLS)
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
    sample = sample.copy()
    sample["cluster"] = km.labels_
    order = np.argsort(km.cluster_centers_[:, PLAYER_FIELD_COLS.index("finishing")])
    label = {c: name for c, name in zip(order, ["Defensor", "Mediocampista", "Delantero"])}
    return sample, km, label


def team_clusters():
    ta = load_table("Team_Attributes", DB_PATH)
    tf = ta[TEAM_TACTIC_COLS].dropna().reset_index(drop=True)
    X = standardize(tf, TEAM_TACTIC_COLS)
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
    tf = tf.copy()
    tf["cluster"] = km.labels_
    order = np.argsort(km.cluster_centers_[:, TEAM_TACTIC_COLS.index("buildUpPlaySpeed")])
    label = {c: name for c, name in zip(order, ["Conservador", "Contraataque", "Ofensivo"])}
    return tf, km, label


def fig_clusters_equipos():
    tf, km, label = team_clusters()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c in sorted(label):
        mask = tf["cluster"] == c
        ax.scatter(
            tf.loc[mask, "buildUpPlaySpeed"],
            tf.loc[mask, "defencePressure"],
            s=48,
            alpha=0.75,
            color=PAL[c],
            edgecolor="w",
            label=f"{label[c]} (n={mask.sum()})",
        )
    for c, name in label.items():
        cx = km.cluster_centers_[c, TEAM_TACTIC_COLS.index("buildUpPlaySpeed")]
        cy = km.cluster_centers_[c, TEAM_TACTIC_COLS.index("defencePressure")]
        ax.scatter(cx, cy, marker="X", s=220, color="black", zorder=5)
        ax.annotate(club_names[name], (cx, cy), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Velocidad de juego (buildUpPlaySpeed)")
    ax.set_ylabel("Presión defensiva (defencePressure)")
    ax.set_title("Estilos tácticos de equipos — K-Medias (K=3)", fontsize=13)
    ax.legend(title="Cluster", loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/clusters_equipos.png", dpi=150)
    plt.close(fig)


def fig_clusters_jugadores():
    sample, km, label = player_clusters()
    fig, ax = plt.subplots(figsize=(8, 5.5), subplot_kw=dict(polar=True))
    angles = [n / len(RADAR_FIELDS) * 2 * pi for n in range(len(RADAR_FIELDS))]
    angles += angles[:1]
    for c, name in label.items():
        means = sample.loc[sample["cluster"] == c, RADAR_FIELDS].mean().tolist()
        values = means + means[:1]
        ax.plot(angles, values, linewidth=2, color=PAL[c], label=f"{name} (n={ (sample['cluster']==c).sum() })")
        ax.fill(angles, values, color=PAL[c], alpha=0.22)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_LABELS, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_title("Perfiles de jugadores por cluster — K-Medias (K=3)", fontsize=13, pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.08), frameon=True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/clusters_jugadores.png", dpi=150)
    plt.close(fig)


def fig_clasificacion():
    recipes = ["Team", "Player", "Player+Team", "Cuotas B365", "Combinado"]
    train = [52.95, 60.86, 60.87, 52.59, 59.77]
    test = [50.54, 52.00, 51.92, 52.19, 52.42]
    breakeven = 100 / 1.85
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(recipes))
    w = 0.36
    ax.bar(x - w / 2, train, w, label="Entrenamiento", color=PAL[0])
    ax.bar(x + w / 2, test, w, label="Test", color=PAL[1])
    ax.axhline(breakeven, ls="--", color="crimson", lw=1.5)
    ax.text(len(recipes) - 0.35, breakeven + 0.4, f"Breakeven {breakeven:.2f}%", color="crimson", fontsize=10, ha="right")
    ax.set_xticks(x, recipes)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("XGBoost — accuracy por receta de features", fontsize=13)
    ax.set_ylim(45, 66)
    for i in range(len(recipes)):
        ax.text(i + w / 2, test[i] + 0.5, f"{test[i]:.2f}", ha="center", fontsize=9)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/clasificacion_modelos.png", dpi=150)
    plt.close(fig)


def fig_mitigacion():
    models = ["Original", "Regularizado", "Optuna", "TabNet"]
    train = [60.52, 56.89, 66.62, None]
    test = [52.48, 52.41, 51.54, 50.99]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(models))
    w = 0.36
    tw = [t for t in train if t is not None]
    ax.bar(x[: len(tw)] - w / 2, tw, w, label="Entrenamiento", color=PAL[0])
    ax.bar(x + w / 2, test, w, label="Test", color=PAL[1])
    for i in range(len(models)):
        ax.text(i, test[i] + 0.6, f"{test[i]:.2f}%", ha="center", fontsize=10)
    for i in range(len(tw)):
        ax.text(i - 0.2, tw[i] + 0.6, f"{tw[i]:.2f}%", ha="center", fontsize=9)
    ax.set_xticks(x, models)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Mitigación del sobreajuste — train vs test", fontsize=13)
    ax.set_ylim(45, 72)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/mitigacion_sobreajuste.png", dpi=150)
    plt.close(fig)


def fig_regresion():
    models = ["Baseline", "Ridge", "Random Forest", "XGBoost"]
    mae = [5.034, 4.558, 4.537, 4.417]
    rmse = [6.264, 5.656, 5.629, 5.454]
    r2 = [-0.004, 0.182, 0.189, 0.239]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(models))
    w = 0.36
    ax.bar(x - w / 2, mae, w, label="MAE", color=PAL[0])
    ax.bar(x + w / 2, rmse, w, label="RMSE", color=PAL[2])
    for i in range(len(models)):
        ax.text(i, rmse[i] + 0.12, r"$R^2$" + f" {r2[i]:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, models)
    ax.set_ylabel("Error (puntos sobre 100)")
    ax.set_title("Regresión de overall_rating — perfil físico + posición", fontsize=13)
    ax.set_ylim(0, 7.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/regresion_atributos.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_clusters_equipos()
    fig_clusters_jugadores()
    fig_clasificacion()
    fig_mitigacion()
    fig_regresion()
    print("Gráficos guardados en", OUT)