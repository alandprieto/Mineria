import re

import pandas as pd

ID_COLS_TEAM = ["id", "team_fifa_api_id", "team_api_id"]


def latest_team_attrs(team_attributes: pd.DataFrame) -> pd.DataFrame:
    return (
        team_attributes.sort_values("date")
        .drop_duplicates("team_api_id", keep="last")
        .select_dtypes(include="number")
    )


def stat_cols_team(team_attrs: pd.DataFrame) -> list[str]:
    return [c for c in team_attrs.columns if c not in ID_COLS_TEAM]


def team_slot_cols(match: pd.DataFrame) -> list[str]:
    return [c for c in match.columns if re.fullmatch(r"(home|away)_team_api_id", c)]


def attach_team_stats(
    id_series: pd.Series,
    team_attrs: pd.DataFrame,
    stat_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    id_col = "team_api_id"
    tmp = id_series.reset_index(drop=True).to_frame(name=id_col)
    tmp = tmp.merge(team_attrs[[id_col] + stat_cols], on=id_col, how="left")
    tmp = tmp.drop(columns=id_col)
    tmp.columns = [f"{prefix}_{c}" for c in tmp.columns]
    tmp.index = id_series.index
    return tmp