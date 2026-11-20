"""Who are each team's stars, and are they playing?

Before every game, using only the team's previous games (no look-ahead):

- QB1: the quarterback with the most dropbacks in the team's last 10 games.
- Offensive stars: the 2 running backs, receivers or tight ends with the most
  PPR fantasy points in those games.
- Defensive stars: the 2 defenders with the most impact plays in those games
  (sacks, QB hits, tackles for loss, interceptions, passes defended, forced
  fumbles).

A star counts as absent when the official injury report lists him Out or
Doubtful for that week, or when the weekly roster has him on injured reserve,
released, or no longer on the team. Game-day inactives are ignored on
purpose: they are announced 90 minutes before kickoff, too late for us.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import nflreadpy as nfl
import polars as pl

from nfl_analyst.teams import canonical

LOOKBACK_GAMES = 10
OFF_STARS = 2
DEF_STARS = 2
OFF_GROUPS = {"RB", "WR", "TE"}
DEF_GROUPS = {"DL", "LB", "DB"}
ABSENT_REPORT = {"Out", "Doubtful"}
PRESENT_ROSTER = {"ACT", "INA"}
FIRST_INJURY_SEASON = 2012


@dataclass
class Star:
    player_id: str
    name: str
    position: str
    role: str  # "QB", "OFF" or "DEF"
    value: float  # dropbacks, PPR points or impact plays in the lookback window
    played_last_game: bool = True
    epa_per_play: float = 0.0  # QB only: (passing + rushing EPA) per dropback
    absent: bool = False
    reason: str = ""


@dataclass
class TeamAvailability:
    team: str
    stars: list[Star] = field(default_factory=list)

    def missing(self, role: str) -> list[Star]:
        return [s for s in self.stars if s.role == role and s.absent]

    @property
    def qb_out(self) -> bool:
        return bool(self.missing("QB"))


# Codes used only by the weekly roster files.
ROSTER_CODES = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LA"}


def _canon(col: str) -> pl.Expr:
    return pl.col(col).map_elements(
        lambda code: canonical(ROSTER_CODES.get(code, code)), return_dtype=pl.Utf8
    )


def load_player_games(seasons: list[int]) -> pl.DataFrame:
    ps = nfl.load_player_stats(seasons, summary_level="week")
    return ps.select(
        "game_id",
        "player_id",
        pl.col("player_display_name").alias("name"),
        "position",
        "position_group",
        _canon("team").alias("team"),
        (pl.col("attempts").fill_null(0) + pl.col("sacks_suffered").fill_null(0)).alias(
            "dropbacks"
        ),
        pl.col("fantasy_points_ppr").fill_null(0).alias("ppr"),
        (
            2 * pl.col("def_sacks").fill_null(0)
            + pl.col("def_qb_hits").fill_null(0)
            + pl.col("def_tackles_for_loss").fill_null(0)
            + 3 * pl.col("def_interceptions").fill_null(0)
            + pl.col("def_pass_defended").fill_null(0)
            + 2 * pl.col("def_fumbles_forced").fill_null(0)
        ).alias("impact"),
        (pl.col("passing_epa").fill_null(0) + pl.col("rushing_epa").fill_null(0)).alias("epa"),
    )


def load_absence_sources(seasons: list[int]) -> tuple[dict, dict]:
    """(injury report Out/Doubtful, weekly roster status) keyed by season/week/team."""
    seasons = [s for s in seasons if s >= FIRST_INJURY_SEASON]
    injuries = nfl.load_injuries(seasons).filter(pl.col("report_status").is_in(ABSENT_REPORT))
    report: dict[tuple[int, int, str], dict[str, str]] = defaultdict(dict)
    for r in injuries.with_columns(_canon("team").alias("team")).iter_rows(named=True):
        report[(r["season"], r["week"], r["team"])][r["gsis_id"]] = r["report_status"]

    rosters = nfl.load_rosters_weekly(seasons).select(
        "season", "week", _canon("team").alias("team"), "gsis_id", "status"
    )
    roster: dict[tuple[int, int, str], dict[str, str]] = defaultdict(dict)
    for r in rosters.iter_rows(named=True):
        if r["gsis_id"]:
            roster[(r["season"], r["week"], r["team"])][r["gsis_id"]] = r["status"]
    return dict(report), dict(roster)


def _top(players: dict[str, dict], key: str, groups: set[str], n: int) -> list[dict]:
    pool = [p for p in players.values() if p["group"] in groups and p[key] > 0]
    return sorted(pool, key=lambda p: p[key], reverse=True)[:n]


def _stars_from_window(window: list[list[dict]]) -> list[Star]:
    players: dict[str, dict] = {}
    last_ids = {r["player_id"] for r in window[-1]} if window else set()
    for game_rows in window:
        for r in game_rows:
            p = players.setdefault(
                r["player_id"],
                {
                    "id": r["player_id"],
                    "name": r["name"],
                    "position": r["position"],
                    "group": r["position_group"],
                    "dropbacks": 0.0,
                    "ppr": 0.0,
                    "impact": 0.0,
                    "epa": 0.0,
                },
            )
            p["dropbacks"] += r["dropbacks"]
            p["ppr"] += r["ppr"]
            p["impact"] += r["impact"]
            p["epa"] += r["epa"]
    stars = [
        Star(
            p["id"],
            p["name"],
            p["position"],
            "QB",
            p["dropbacks"],
            played_last_game=p["id"] in last_ids,
            epa_per_play=p["epa"] / p["dropbacks"],
        )
        for p in _top(players, "dropbacks", {"QB"}, 1)
    ]
    stars += [
        Star(p["id"], p["name"], p["position"], "OFF", round(p["ppr"], 1), p["id"] in last_ids)
        for p in _top(players, "ppr", OFF_GROUPS, OFF_STARS)
    ]
    stars += [
        Star(p["id"], p["name"], p["position"], "DEF", round(p["impact"], 1), p["id"] in last_ids)
        for p in _top(players, "impact", DEF_GROUPS, DEF_STARS)
    ]
    return stars


def _mark_absences(
    stars: list[Star], season: int, week: int, team: str, report: dict, roster: dict
) -> None:
    listed = report.get((season, week, team), {})
    # Upcoming weeks may not have a roster yet: use the latest one of the season.
    roster_week = week
    while roster_week > 0 and (season, roster_week, team) not in roster:
        roster_week -= 1
    team_roster = roster.get((season, roster_week, team))
    for star in stars:
        if star.player_id in listed:
            star.absent, star.reason = True, f"Reporte: {listed[star.player_id]}"
        elif team_roster is not None:
            status = team_roster.get(star.player_id)
            if status is None:
                star.absent, star.reason = True, "Ya no está en el roster"
            elif status not in PRESENT_ROSTER:
                label = {"RES": "Lista de lesionados (IR)", "CUT": "Liberado"}.get(status, status)
                star.absent, star.reason = True, label


def availability_by_game(
    games: pl.DataFrame, player_games: pl.DataFrame, report: dict, roster: dict
) -> dict[str, dict[str, TeamAvailability]]:
    """For every game (played or not): {game_id: {team: TeamAvailability}}.

    `games` must be in chronological order (as load_games returns it)."""
    rows_by_game: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in player_games.iter_rows(named=True):
        rows_by_game[(r["team"], r["game_id"])].append(r)

    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=LOOKBACK_GAMES))
    result: dict[str, dict[str, TeamAvailability]] = {}
    for g in games.iter_rows(named=True):
        result[g["game_id"]] = {}
        for team in (g["home_team"], g["away_team"]):
            stars = _stars_from_window(list(history[team]))
            _mark_absences(stars, g["season"], g["week"], team, report, roster)
            result[g["game_id"]][team] = TeamAvailability(team, stars)
        if g.get("home_score") is not None:
            for team in (g["home_team"], g["away_team"]):
                history[team].append(rows_by_game.get((team, g["game_id"]), []))
    return result
