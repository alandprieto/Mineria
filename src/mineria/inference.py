import numpy as np
import pandas as pd

from .config import DB_PATH, ODDS_B365, STREAK_WINDOWS
from .data_loader import load_table
from .features.player_features import (
    latest_player_attrs,
    stat_cols_player,
)
from .features.streaks import _points_per_team
from .features.team_features import (
    latest_team_attrs,
    stat_cols_team,
)


def _attach_entity(entity_id, attrs_df, id_col, stat_cols, prefix):
    row = attrs_df.loc[attrs_df[id_col] == entity_id, stat_cols]
    row = row.reset_index(drop=True)
    row.columns = [f"{prefix}_{c}" for c in row.columns]
    return row


def team_streak(team_id, match, windows=STREAK_WINDOWS):
    history = _points_per_team(match)
    history = history[history["team_api_id"] == team_id].sort_values("date")
    streaks = {}
    for w in windows:
        streaks[w] = history["points"].rolling(w, min_periods=1).mean().iloc[-1]
    return streaks


def build_fixture_features(
    home_player_ids,
    away_player_ids,
    home_team_id,
    away_team_id,
    odds=None,
    streaks=None,
    recipe="combined",
    db_path=DB_PATH,
):
    player_attrs = latest_player_attrs(load_table("Player_Attributes", db_path))
    team_attrs = latest_team_attrs(load_table("Team_Attributes", db_path))
    pcols = stat_cols_player(player_attrs)
    tcols = stat_cols_team(team_attrs)

    blocks = []

    if recipe in ("pa", "pa_ta", "combined"):
        for i, pid in enumerate(home_player_ids):
            blocks.append(_attach_entity(pid, player_attrs, "player_api_id", pcols, f"home_player_{i + 1}"))
        for i, pid in enumerate(away_player_ids):
            blocks.append(_attach_entity(pid, player_attrs, "player_api_id", pcols, f"away_player_{i + 1}"))

    if recipe in ("ta", "pa_ta", "combined"):
        blocks.append(_attach_entity(home_team_id, team_attrs, "team_api_id", tcols, "home_team_api_id"))
        blocks.append(_attach_entity(away_team_id, team_attrs, "team_api_id", tcols, "away_team_api_id"))

    if recipe in ("odds", "combined"):
        if odds is None:
            odds = {k: np.nan for k in ODDS_B365}
        blocks.append(pd.DataFrame([{k: odds.get(k, np.nan) for k in ODDS_B365}]))

    if recipe == "combined":
        if streaks is None:
            streak_values = {f"{side}_streak_{w}": np.nan for side in ("home", "away") for w in STREAK_WINDOWS}
        else:
            streak_values = {
                f"{side}_streak_{w}": streaks.get(f"{side}_streak_{w}", np.nan)
                for side in ("home", "away")
                for w in STREAK_WINDOWS
            }
        blocks.append(pd.DataFrame([streak_values]))

    return pd.concat(blocks, axis=1)


def predict_fixture(
    model,
    imputer,
    feature_columns,
    home_player_ids,
    away_player_ids,
    home_team_id,
    away_team_id,
    odds=None,
    streaks=None,
    recipe="combined",
    db_path=DB_PATH,
):
    row = build_fixture_features(
        home_player_ids,
        away_player_ids,
        home_team_id,
        away_team_id,
        odds=odds,
        streaks=streaks,
        recipe=recipe,
        db_path=db_path,
    )
    row = row.reindex(columns=feature_columns)
    row = row.fillna(imputer.medians_)
    return model.predict_proba(row)[0], model.predict(row)[0]