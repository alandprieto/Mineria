import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from .config import SEED

TEAM_TACTIC_COLS = [
    "buildUpPlaySpeed",
    "buildUpPlayDribbling",
    "buildUpPlayPassing",
    "chanceCreationPassing",
    "chanceCreationCrossing",
    "chanceCreationShooting",
    "defencePressure",
    "defenceAggression",
    "defenceTeamWidth",
]

PLAYER_FIELD_COLS = [
    "crossing",
    "finishing",
    "heading_accuracy",
    "short_passing",
    "dribbling",
    "long_passing",
    "ball_control",
    "acceleration",
    "sprint_speed",
    "agility",
    "reactions",
    "shot_power",
    "jumping",
    "stamina",
    "strength",
    "long_shots",
    "aggression",
    "interceptions",
    "positioning",
    "vision",
    "penalties",
    "marking",
    "standing_tackle",
    "sliding_tackle",
]


def team_cluster_data(team_attributes: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Snapshots tácticos sin nulos (489 según el informe) con team_api_id."""
    cols = TEAM_TACTIC_COLS
    df = team_attributes[["team_api_id"] + cols].dropna().reset_index(drop=True)
    return df, cols


def player_cluster_data(
    player_attributes: pd.DataFrame,
    sample_size: int = 20000,
    seed: int = SEED,
) -> tuple[pd.DataFrame, list[str]]:
    """Jugadores de campo (gk_diving < 30), sin nulos y muestra aleatoria de 20000."""
    cols = PLAYER_FIELD_COLS
    df = player_attributes[player_attributes["gk_diving"] < 30][
        ["player_api_id"] + cols
    ].dropna()
    df = df.reset_index(drop=True)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return df, cols


def standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        StandardScaler().fit_transform(df[cols]), index=df.index, columns=cols
    )


def kmeans_scores(
    X: pd.DataFrame, k_range: range, seed: int = SEED
) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        rows.append(
            {
                "k": k,
                "inercia": km.inertia_,
                "db_score": davies_bouldin_score(X, km.labels_),
            }
        )
    return pd.DataFrame(rows)


def fit_kmeans(X: pd.DataFrame, k: int, seed: int = SEED) -> KMeans:
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)


def closest_to_centroids(
    df: pd.DataFrame, X: pd.DataFrame, km: KMeans
) -> pd.DataFrame:
    """Entidad de cada clúster más próxima a su centroide (distancia euclidiana)."""
    dist = km.transform(X)
    idx = dist.argmin(axis=0)
    return df.iloc[idx]


def profile_means(df: pd.DataFrame, km: KMeans, cols: list[str]) -> pd.DataFrame:
    """Media de atributos por clúster, para la interpretación de perfiles."""
    out = df[cols].copy()
    out["cluster"] = km.labels_
    return out.groupby("cluster").mean().T