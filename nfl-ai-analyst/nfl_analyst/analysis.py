"""Glue: load data, run the model and assemble reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import polars as pl

from nfl_analyst import availability, model, news
from nfl_analyst.data import FIRST_SEASON, current_season, load_games
from nfl_analyst.teams import TEAM_NAMES, canonical


@dataclass
class GameReport:
    prediction: model.Prediction
    kickoff: str
    injuries: pl.DataFrame | None = None
    headlines: list[news.Headline] = field(default_factory=list)
    news_errors: list[str] = field(default_factory=list)
    summary: str | None = None


@dataclass
class State:
    games: pl.DataFrame
    ratings: model.Ratings
    history: list[tuple[dict, model.Prediction]]
    availability: dict[str, model.GameAvailability]


@lru_cache(maxsize=1)
def _state() -> State:
    games = load_games()
    seasons = list(range(FIRST_SEASON, current_season() + 1))
    report, roster = availability.load_absence_sources(seasons)
    avail = availability.availability_by_game(
        games, availability.load_player_games(seasons), report, roster
    )
    ratings, history = model.run_history(games, availability=avail)
    return State(games, ratings, history, avail)


def refresh() -> None:
    _state.cache_clear()


def upcoming_games(week: int | None = None) -> pl.DataFrame:
    games = _state().games
    season = current_season()
    pending = games.filter((pl.col("season") == season) & pl.col("home_score").is_null())
    if pending.is_empty():
        return pending
    week = week or int(pending["week"].min())
    return pending.filter(pl.col("week") == week)


def _kickoff(game: dict) -> str:
    return f"{game['gameday']} {game.get('gametime') or ''} ET".strip()


def _predict(game: dict) -> model.Prediction:
    state = _state()
    return state.ratings.predict(game, state.availability.get(game["game_id"]))


def predict_week(week: int | None = None) -> list[GameReport]:
    return [
        GameReport(prediction=_predict(g), kickoff=_kickoff(g))
        for g in upcoming_games(week).iter_rows(named=True)
    ]


def find_game(team_a: str, team_b: str) -> dict:
    """Next unplayed game between two teams this season (either home/away)."""
    a, b = canonical(team_a), canonical(team_b)
    games = _state().games
    match = games.filter(
        (pl.col("season") == current_season())
        & pl.col("home_score").is_null()
        & (
            ((pl.col("home_team") == a) & (pl.col("away_team") == b))
            | ((pl.col("home_team") == b) & (pl.col("away_team") == a))
        )
    )
    if match.is_empty():
        raise ValueError(
            f"No hay un partido pendiente {a} vs {b} esta temporada. "
            "Revisa los equipos o usa el comando 'week'."
        )
    return match.row(0, named=True)


def analyze_game(team_a: str, team_b: str, with_news: bool = True) -> GameReport:
    game = find_game(team_a, team_b)
    report = GameReport(prediction=_predict(game), kickoff=_kickoff(game))
    if with_news:
        teams = [game["home_team"], game["away_team"]]
        report.injuries = news.injury_report(teams)
        report.headlines, report.news_errors = news.headlines_for(teams)
        matchup = f"{TEAM_NAMES[game['away_team']]} @ {TEAM_NAMES[game['home_team']]}"
        report.summary = news.summarize(matchup, report.injuries, report.headlines)
    return report


def ratings_table() -> pl.DataFrame:
    ratings = _state().ratings
    rows = [
        {
            "team": team,
            "name": TEAM_NAMES[team],
            "elo": round(elo, 1),
            "points_scored": round(ratings.points[team].scored, 1),
            "points_allowed": round(ratings.points[team].allowed, 1),
        }
        for team, elo in ratings.elo.items()
        if team in TEAM_NAMES
    ]
    return pl.DataFrame(rows).sort("elo", descending=True)


def backtest(test_seasons: set[int] | None = None) -> dict[str, float]:
    history = _state().history
    season = current_season()
    test_seasons = test_seasons or {season - 3, season - 2, season - 1}
    return model.evaluate(history, test_seasons)
