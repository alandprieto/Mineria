import re

import pandas as pd

ID_COLS_PLAYER = ["id", "player_fifa_api_id", "player_api_id"]


def latest_player_attrs(player_attributes: pd.DataFrame) -> pd.DataFrame:
    return (
        player_attributes.sort_values("date")
        .drop_duplicates("player_api_id", keep="last")
        .select_dtypes(include="number")
    )


def stat_cols_player(player_attrs: pd.DataFrame) -> list[str]:
    return [c for c in player_attrs.columns if c not in ID_COLS_PLAYER]


def player_slot_cols(match: pd.DataFrame) -> list[str]:
    return [c for c in match.columns if re.fullmatch(r"(home|away)_player_\d+", c)]


def attach_player_stats(
    id_series: pd.Series,
    player_attrs: pd.DataFrame,
    stat_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    id_col = "player_api_id"
    tmp = id_series.reset_index(drop=True).to_frame(name=id_col)
    tmp = tmp.merge(player_attrs[[id_col] + stat_cols], on=id_col, how="left")
    tmp = tmp.drop(columns=id_col)
    tmp.columns = [f"{prefix}_{c}" for c in tmp.columns]
    tmp.index = id_series.index
    return tmp