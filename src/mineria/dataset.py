import re

import pandas as pd

from .config import DB_PATH, ODDS_B365
from .data_loader import load_table
from .features.match_features import build_ftr
from .features.player_features import (
    attach_player_stats,
    latest_player_attrs,
    player_slot_cols,
    stat_cols_player,
)
from .features.streaks import add_streaks
from .features.team_features import (
    attach_team_stats,
    latest_team_attrs,
    stat_cols_team,
    team_slot_cols,
)

RECIPES = {"ta", "pa", "pa_ta", "odds", "combined"}


def build_dataset(recipe="combined", db_path=DB_PATH, streak_windows=(5, 10, 15)):
    if recipe not in RECIPES:
        raise ValueError(f"Receta desconocida: {recipe}")

    match = load_table("Match", db_path)
    match["date"] = pd.to_datetime(match["date"])
    match = build_ftr(match)

    if recipe == "combined":
        match = add_streaks(match, windows=streak_windows)

    blocks = []

    if recipe in ("pa", "pa_ta", "combined"):
        player_attrs = latest_player_attrs(load_table("Player_Attributes", db_path))
        player_cols = stat_cols_player(player_attrs)
        blocks += [
            attach_player_stats(match[col], player_attrs, player_cols, col)
            for col in player_slot_cols(match)
        ]

    if recipe in ("ta", "pa_ta", "combined"):
        team_attrs = latest_team_attrs(load_table("Team_Attributes", db_path))
        team_cols = stat_cols_team(team_attrs)
        blocks += [
            attach_team_stats(match[col], team_attrs, team_cols, col)
            for col in team_slot_cols(match)
        ]

    if recipe in ("odds", "combined"):
        blocks.append(match[ODDS_B365])

    if recipe == "combined":
        streak_cols = [
            c for c in match.columns if re.fullmatch(r"(home|away)_streak_\d+", c)
        ]
        blocks.append(match[streak_cols])

    features = pd.concat([match[["match_api_id", "FTR"]]] + blocks, axis=1)
    feature_columns = [c for c in features.columns if c not in ("match_api_id", "FTR")]
    return features, feature_columns