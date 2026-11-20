"""A deliberately simple game model.

Two ratings, updated game by game in chronological order, so every prediction
only uses games played before it (no look-ahead):

1. Elo rating per team (FiveThirtyEight-style, with margin-of-victory
   multiplier and a 1/3 reversion to the mean between seasons). Gives the
   expected point margin: Elo difference / 25.
2. Points rating per team: exponentially weighted points scored and allowed.
   Gives the expected score of each team, so the projected total.

The final margin blends both, then subtracts a fixed penalty for each star
the injury report or roster says will miss the game (see availability.py).
Win probability comes from a normal distribution of the margin around that
projection (NFL margins have a standard deviation of roughly 13.5 points).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import polars as pl

from nfl_analyst.availability import TeamAvailability

GameAvailability = dict[str, TeamAvailability]

ELO_MEAN = 1505.0
ELO_K = 20.0
ELO_HOME_ADV = 48.0  # Elo points, about 1.9 points of margin
ELO_POINTS_PER_POINT = 25.0
SEASON_REVERSION = 1 / 3

POINTS_ALPHA = 0.10  # weight of the latest game in the points rating (half-life ~6.6 games)
POINTS_HOME_ADV = 1.6  # home field, in points, for the points model
LEAGUE_POINTS_START = 22.0

MARGIN_SD = 13.5
TOTAL_SD = 13.0
ELO_WEIGHT = 0.5  # share of the Elo margin in the blended margin (see backtest)

# Points of margin lost per missing star. Fitted by least squares on the
# model's errors in 2013-2022 and checked on 2023-2025 (see README).
QB_OUT_PENALTY = 2.0
OFF_STAR_PENALTY = 1.1
DEF_STAR_PENALTY = 0.6


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def absence_penalty(team_availability: TeamAvailability | None) -> float:
    """Points of margin a team loses for its missing stars."""
    if team_availability is None:
        return 0.0
    return (
        QB_OUT_PENALTY * len(team_availability.missing("QB"))
        + OFF_STAR_PENALTY * len(team_availability.missing("OFF"))
        + DEF_STAR_PENALTY * len(team_availability.missing("DEF"))
    )


@dataclass
class TeamPoints:
    scored: float = LEAGUE_POINTS_START
    allowed: float = LEAGUE_POINTS_START


@dataclass
class Prediction:
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    neutral: bool
    home_elo: float
    away_elo: float
    elo_margin: float
    points_margin: float
    home_points: float
    away_points: float
    margin: float  # home minus away, blended, after absence penalties
    total: float
    home_win_prob: float
    elo_home_win_prob: float
    home_absence_penalty: float = 0.0
    away_absence_penalty: float = 0.0
    availability: GameAvailability | None = None

    @property
    def away_win_prob(self) -> float:
        return 1 - self.home_win_prob

    def prob_home_covers(self, home_line: float) -> float:
        """P(home margin + line > 0); home_line is the home spread, e.g. -3.5."""
        return 1 - normal_cdf((-home_line - self.margin) / MARGIN_SD)

    def prob_over(self, total_line: float) -> float:
        return 1 - normal_cdf((total_line - self.total) / TOTAL_SD)


@dataclass
class Ratings:
    elo_weight: float = ELO_WEIGHT
    elo: dict[str, float] = field(default_factory=dict)
    points: dict[str, TeamPoints] = field(default_factory=dict)
    league_points: float = LEAGUE_POINTS_START
    season: int | None = None
    games_processed: int = 0
    last_game: str | None = None

    # -- season boundaries --------------------------------------------------

    def _start_season(self, season: int) -> None:
        if self.season is not None and season != self.season:
            for team, rating in self.elo.items():
                self.elo[team] = rating + SEASON_REVERSION * (ELO_MEAN - rating)
            for team_points in self.points.values():
                team_points.scored += SEASON_REVERSION * (self.league_points - team_points.scored)
                team_points.allowed += SEASON_REVERSION * (self.league_points - team_points.allowed)
        self.season = season

    def _team(self, team: str) -> tuple[float, TeamPoints]:
        self.elo.setdefault(team, ELO_MEAN)
        self.points.setdefault(team, TeamPoints(self.league_points, self.league_points))
        return self.elo[team], self.points[team]

    # -- prediction ---------------------------------------------------------

    def predict(self, game: dict, availability: GameAvailability | None = None) -> Prediction:
        self._start_season(int(game["season"]))
        home, away = game["home_team"], game["away_team"]
        neutral = (game.get("location") or "") == "Neutral"
        home_elo, home_pts = self._team(home)
        away_elo, away_pts = self._team(away)

        elo_diff = home_elo - away_elo + (0 if neutral else ELO_HOME_ADV)
        elo_margin = elo_diff / ELO_POINTS_PER_POINT
        elo_prob = 1 / (1 + 10 ** (-elo_diff / 400))

        hfa = 0 if neutral else POINTS_HOME_ADV / 2
        home_points = home_pts.scored + away_pts.allowed - self.league_points + hfa
        away_points = away_pts.scored + home_pts.allowed - self.league_points - hfa
        points_margin = home_points - away_points

        availability = availability or {}
        home_penalty = absence_penalty(availability.get(home))
        away_penalty = absence_penalty(availability.get(away))
        margin = (
            self.elo_weight * elo_margin
            + (1 - self.elo_weight) * points_margin
            - home_penalty
            + away_penalty
        )
        return Prediction(
            game_id=game["game_id"],
            season=int(game["season"]),
            week=int(game["week"]),
            home_team=home,
            away_team=away,
            neutral=neutral,
            home_elo=home_elo,
            away_elo=away_elo,
            elo_margin=elo_margin,
            points_margin=points_margin,
            home_points=home_points,
            away_points=away_points,
            margin=margin,
            total=home_points + away_points,
            home_win_prob=normal_cdf(margin / MARGIN_SD),
            elo_home_win_prob=elo_prob,
            home_absence_penalty=home_penalty,
            away_absence_penalty=away_penalty,
            availability=availability or None,
        )

    # -- update after a game is played ---------------------------------------

    def update(self, game: dict, prediction: Prediction | None = None) -> None:
        prediction = prediction or self.predict(game)
        home, away = game["home_team"], game["away_team"]
        home_score, away_score = float(game["home_score"]), float(game["away_score"])
        mov = home_score - away_score

        # Elo
        expected = prediction.elo_home_win_prob
        actual = 1.0 if mov > 0 else 0.0 if mov < 0 else 0.5
        winner_diff = (prediction.elo_margin * ELO_POINTS_PER_POINT) * (1 if mov >= 0 else -1)
        multiplier = math.log(abs(mov) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
        shift = ELO_K * multiplier * (actual - expected)
        self.elo[home] += shift
        self.elo[away] -= shift

        # Points
        hfa = 0 if prediction.neutral else POINTS_HOME_ADV / 2
        a = POINTS_ALPHA
        home_pts, away_pts = self.points[home], self.points[away]
        home_pts.scored += a * (home_score - hfa - home_pts.scored)
        home_pts.allowed += a * (away_score + hfa - home_pts.allowed)
        away_pts.scored += a * (away_score + hfa - away_pts.scored)
        away_pts.allowed += a * (home_score - hfa - away_pts.allowed)
        self.league_points += (a / 16) * ((home_score + away_score) / 2 - self.league_points)

        self.games_processed += 1
        self.last_game = game["game_id"]


def is_played(game: dict) -> bool:
    return game.get("home_score") is not None and game.get("away_score") is not None


def run_history(
    games: pl.DataFrame,
    elo_weight: float = ELO_WEIGHT,
    availability: dict[str, GameAvailability] | None = None,
) -> tuple[Ratings, list[tuple[dict, Prediction]]]:
    """Walk through every game in order. Returns final ratings and the
    pre-game prediction of every played game (walk-forward, no leakage).
    `availability` maps game_id to each team's stars (availability.py)."""
    availability = availability or {}
    ratings = Ratings(elo_weight=elo_weight)
    history: list[tuple[dict, Prediction]] = []
    for game in games.iter_rows(named=True):
        if not is_played(game):
            continue
        prediction = ratings.predict(game, availability.get(game["game_id"]))
        history.append((game, prediction))
        ratings.update(game, prediction)
    return ratings, history


def predict_upcoming(
    games: pl.DataFrame,
    ratings: Ratings,
    availability: dict[str, GameAvailability] | None = None,
) -> list[Prediction]:
    availability = availability or {}
    return [
        ratings.predict(g, availability.get(g["game_id"]))
        for g in games.iter_rows(named=True)
        if not is_played(g)
    ]


def evaluate(history: list[tuple[dict, Prediction]], seasons: set[int]) -> dict[str, float]:
    """Accuracy and calibration of pre-game predictions, compared with the
    closing market line published by nflverse."""
    rows = [(g, p) for g, p in history if g["season"] in seasons]
    if not rows:
        return {}
    n = len(rows)
    brier = log_loss = correct = margin_err = total_err = 0.0
    market_margin_err = market_total_err = 0.0
    market_n = 0
    for game, pred in rows:
        mov = game["home_score"] - game["away_score"]
        total = game["home_score"] + game["away_score"]
        outcome = 1.0 if mov > 0 else 0.0 if mov < 0 else 0.5
        p = min(max(pred.home_win_prob, 1e-6), 1 - 1e-6)
        brier += (p - outcome) ** 2
        log_loss -= outcome * math.log(p) + (1 - outcome) * math.log(1 - p)
        correct += 1.0 if (p > 0.5) == (mov > 0) and mov != 0 else 0.0
        margin_err += abs(pred.margin - mov)
        total_err += abs(pred.total - total)
        if game.get("spread_line") is not None and game.get("total_line") is not None:
            market_n += 1
            market_margin_err += abs(game["spread_line"] - mov)  # positive = home favored
            market_total_err += abs(game["total_line"] - total)
    return {
        "games": n,
        "accuracy": correct / n,
        "brier": brier / n,
        "log_loss": log_loss / n,
        "margin_mae": margin_err / n,
        "total_mae": total_err / n,
        "market_margin_mae": market_margin_err / market_n if market_n else float("nan"),
        "market_total_mae": market_total_err / market_n if market_n else float("nan"),
    }
