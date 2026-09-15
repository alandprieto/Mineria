import numpy as np
import pandas as pd


def _points_per_team(match: pd.DataFrame) -> pd.DataFrame:
    home = match[["match_api_id", "date", "home_team_api_id", "FTR"]].rename(
        columns={"home_team_api_id": "team_api_id"}
    )
    home["points"] = np.select([home["FTR"] == 0, home["FTR"] == 1], [3, 1], default=0)

    away = match[["match_api_id", "date", "away_team_api_id", "FTR"]].rename(
        columns={"away_team_api_id": "team_api_id"}
    )
    away["points"] = np.select([away["FTR"] == 2, away["FTR"] == 1], [3, 1], default=0)

    return pd.concat([home, away], axis=0).sort_values(["team_api_id", "date"])


def add_streaks(match: pd.DataFrame, windows=(5, 10, 15)) -> pd.DataFrame:
    history = _points_per_team(match)
    for w in windows:
        history[f"streak_{w}"] = (
            history.groupby("team_api_id")["points"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean().shift(1))
        )

    streak_cols = [f"streak_{w}" for w in windows]
    result = match
    for side in ("home", "away"):
        result = result.merge(
            history[["match_api_id", "team_api_id"] + streak_cols],
            left_on=["match_api_id", f"{side}_team_api_id"],
            right_on=["match_api_id", "team_api_id"],
            how="left",
        ).drop(columns=["team_api_id"])
        result = result.rename(columns={f"streak_{w}": f"{side}_streak_{w}" for w in windows})
    return result