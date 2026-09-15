import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .clustering import PLAYER_FIELD_COLS, standardize
from .config import SEED

POSITIONS = ["Arquero", "Defensor", "Mediocampista", "Delantero"]


def fit_position_kmeans(
    player_attributes: pd.DataFrame, k: int = 3, seed: int = SEED
) -> KMeans:
    """K-Means sobre los 24 atributos de campo (sin porteros) para etiquetar posición."""
    cols = PLAYER_FIELD_COLS
    df = player_attributes[player_attributes["gk_diving"] < 30][cols].dropna()
    X = standardize(df, cols)
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)


def map_clusters_to_positions(
    player_attributes: pd.DataFrame, km: KMeans
) -> dict:
    """Asigna a cada cluster su perfil semántico (Delantero/Defensor/Mediocampista)
    según las medias de finishing y marking en su espacio estandarizado."""
    cols = PLAYER_FIELD_COLS
    df = player_attributes[player_attributes["gk_diving"] < 30][cols].dropna()
    df = df.copy()
    X = standardize(df, cols)
    df["cluster"] = km.predict(X)

    means = (
        df.groupby("cluster")[["finishing", "marking", "short_passing"]].mean().round(1)
    )
    delantero = means["finishing"].idxmax()
    rest = [c for c in means.index if c != delantero]
    defensor = means.loc[rest, "marking"].idxmax()
    medio = [c for c in rest if c != defensor][0]
    return {delantero: "Delantero", defensor: "Defensor", medio: "Mediocampista"}


def _all_positions(
    player_attributes: pd.DataFrame, km: KMeans, cluster_map: dict
) -> pd.Series:
    """Posición para todos los jugadores: arqueros (gk_diving>=30) y de campo (cluster)."""
    latest = player_attributes.sort_values("date").drop_duplicates(
        "player_api_id", keep="last"
    )
    cols = PLAYER_FIELD_COLS

    field = latest[latest["gk_diving"] < 30][["player_api_id"] + cols].dropna()
    X = standardize(field, cols)
    labels = km.predict(X)
    field_pos = pd.Series(labels, index=field["player_api_id"].values).map(cluster_map)

    gk_ids = latest.loc[latest["gk_diving"] >= 30, "player_api_id"]
    gk_pos = pd.Series("Arquero", index=gk_ids.values)
    return pd.concat([field_pos, gk_pos])


def player_regression_data(
    player_attributes: pd.DataFrame,
    players: pd.DataFrame,
    km: KMeans,
    cluster_map: dict,
    attr: str = "overall_rating",
) -> pd.DataFrame:
    """Dataset de regresión: perfil físico (altura, peso, edad) + posición -> atributo."""
    pa = player_attributes.sort_values("date").drop_duplicates(
        "player_api_id", keep="last"
    )
    pa = pa.set_index("player_api_id")
    pa["date"] = pd.to_datetime(pa["date"])
    pa = pa[["date", attr]].dropna(subset=[attr])

    pos = _all_positions(player_attributes, km, cluster_map)
    player_tab = players.set_index("player_api_id")[
        ["player_name", "height", "weight", "birthday"]
    ]

    df = pa.join(pos.rename("position")).join(player_tab)
    df = df.dropna(subset=["position", "height", "weight", "birthday"])
    df["age"] = (df["date"] - pd.to_datetime(df["birthday"])).dt.days / 365.25
    df = df.reset_index()
    return df[["player_api_id", "player_name", "height", "weight", "age", "position", attr]]


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(r2_score(y_true, y_pred), 3),
    }


def features_prepare(df: pd.DataFrame, attr: str = "overall_rating"):
    """Devuelve (X, y): altura, peso, edad y posición one-hot."""
    X = df[["height", "weight", "age", "position"]].copy()
    X = pd.get_dummies(X, columns=["position"], dtype=int)
    return X, df[attr]