import numpy as np
import pandas as pd


def build_ftr(match):
    result = match.copy()
    result["FTR"] = np.select(
        [
            result["home_team_goal"] > result["away_team_goal"],
            result["home_team_goal"] == result["away_team_goal"],
        ],
        [0, 1],
        default=2,
    )
    return result


def drop_high_null_columns(df, threshold=0.30):
    null_ratio = df.isnull().mean()
    return df.loc[:, null_ratio <= threshold].copy()