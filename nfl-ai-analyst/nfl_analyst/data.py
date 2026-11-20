"""Download nflverse data with a local cache (data/cache, refreshed every 6 hours)."""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl
import polars as pl
from nflreadpy.config import update_config
from nflreadpy.utils_date import get_current_season

from nfl_analyst.teams import canonical

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIRST_SEASON = 2010  # Elo warm-up starts here; earlier seasons add little.

update_config(cache_mode="filesystem", cache_dir=DATA_DIR / "cache", cache_duration=6 * 3600)


def current_season() -> int:
    return get_current_season()


def load_games(first_season: int = FIRST_SEASON) -> pl.DataFrame:
    """Every game since `first_season`, with team codes mapped to current franchises."""
    seasons = list(range(first_season, current_season() + 1))
    games = nfl.load_schedules(seasons)
    return games.with_columns(
        pl.col("home_team").map_elements(canonical, return_dtype=pl.Utf8),
        pl.col("away_team").map_elements(canonical, return_dtype=pl.Utf8),
    ).sort(["season", "week", "gameday", "gametime", "game_id"])


def load_injuries(season: int | None = None) -> pl.DataFrame:
    season = season or current_season()
    injuries = nfl.load_injuries([season])
    return injuries.with_columns(pl.col("team").map_elements(canonical, return_dtype=pl.Utf8))
