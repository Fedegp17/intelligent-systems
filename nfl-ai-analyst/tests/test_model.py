import polars as pl
import pytest

from nfl_analyst import model
from nfl_analyst.news import parse_rss
from nfl_analyst.teams import canonical


def _games(rows):
    base = {"location": "Home", "spread_line": None, "total_line": None}
    return pl.DataFrame([{**base, **r} for r in rows])


def test_canonical_maps_relocations():
    assert canonical("oak") == "LV"
    assert canonical("SD") == "LAC"
    with pytest.raises(ValueError):
        canonical("XXX")


def test_new_teams_start_even_with_home_edge():
    ratings = model.Ratings()
    p = ratings.predict(
        {
            "game_id": "g",
            "season": 2024,
            "week": 1,
            "home_team": "KC",
            "away_team": "BAL",
            "location": "Home",
        }
    )
    assert 0.5 < p.home_win_prob < 0.6
    assert abs(p.away_win_prob + p.home_win_prob - 1) < 1e-9


def test_neutral_site_has_no_home_edge():
    ratings = model.Ratings()
    p = ratings.predict(
        {
            "game_id": "g",
            "season": 2024,
            "week": 1,
            "home_team": "KC",
            "away_team": "BAL",
            "location": "Neutral",
        }
    )
    assert p.home_win_prob == pytest.approx(0.5)


def test_winner_gains_elo_and_no_lookahead():
    games = _games(
        [
            {
                "game_id": "a",
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "home_score": 35,
                "away_score": 10,
            },
            {
                "game_id": "b",
                "season": 2024,
                "week": 2,
                "home_team": "BAL",
                "away_team": "KC",
                "home_score": None,
                "away_score": None,
            },
        ]
    )
    ratings, history = model.run_history(games)
    assert [g["game_id"] for g, _ in history] == ["a"]
    # The first game's prediction was made before its result was known.
    assert history[0][1].home_elo == model.ELO_MEAN
    assert ratings.elo["KC"] > ratings.elo["BAL"]
    upcoming = model.predict_upcoming(games, ratings)
    assert upcoming[0].away_win_prob > 0.5  # KC, now rated higher, is the away team


def test_season_reverts_toward_mean():
    ratings = model.Ratings()
    ratings.elo = {"KC": 1700.0}
    ratings.season = 2023
    ratings._start_season(2024)
    assert ratings.elo["KC"] == pytest.approx(1700 - (1700 - model.ELO_MEAN) / 3)


def test_line_probabilities():
    ratings = model.Ratings()
    p = ratings.predict(
        {
            "game_id": "g",
            "season": 2024,
            "week": 1,
            "home_team": "KC",
            "away_team": "BAL",
            "location": "Neutral",
        }
    )
    assert p.prob_home_covers(0) == pytest.approx(0.5)
    assert p.prob_over(p.total) == pytest.approx(0.5)


def test_parse_rss_sorts_newest_first():
    xml = """<rss><channel>
      <item><title>Old</title><link>u1</link><pubDate>Mon, 17 Nov 2026 10:00:00 GMT</pubDate>
        <source url="x">ESPN</source></item>
      <item><title>New</title><link>u2</link><pubDate>Wed, 19 Nov 2026 10:00:00 GMT</pubDate>
        <source url="y">NFL.com</source></item>
    </channel></rss>"""
    items = parse_rss(xml, "KC")
    assert [h.title for h in items] == ["New", "Old"]
    assert items[0].source == "NFL.com"


def test_missing_stars_lower_the_team_margin():
    from nfl_analyst.availability import Star, TeamAvailability

    game = {
        "game_id": "g",
        "season": 2024,
        "week": 1,
        "home_team": "KC",
        "away_team": "BAL",
        "location": "Neutral",
    }
    kc = TeamAvailability(
        "KC",
        [
            Star("1", "QB uno", "QB", "QB", 300, absent=True, reason="Reporte: Out"),
            Star("2", "WR uno", "WR", "OFF", 150),
            Star("3", "DE uno", "DE", "DEF", 20, absent=True, reason="Lista de lesionados (IR)"),
        ],
    )
    p = model.Ratings().predict(game, {"KC": kc, "BAL": TeamAvailability("BAL")})
    assert p.home_absence_penalty == pytest.approx(model.QB_OUT_PENALTY + model.DEF_STAR_PENALTY)
    assert p.margin == pytest.approx(-p.home_absence_penalty)
    assert p.away_win_prob > 0.5


def test_stars_come_only_from_previous_games():
    from nfl_analyst import availability as av

    def row(game_id, pid, group, **stats):
        base = {"player_id": pid, "name": pid, "position": group, "position_group": group}
        return {
            "game_id": game_id,
            "team": "KC",
            "dropbacks": 0,
            "ppr": 0,
            "impact": 0,
            "epa": 0,
            **base,
            **stats,
        }

    games = pl.DataFrame(
        [
            {
                "game_id": "a",
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "home_score": 20,
                "away_score": 10,
            },
            {
                "game_id": "b",
                "season": 2024,
                "week": 2,
                "home_team": "KC",
                "away_team": "BAL",
                "home_score": None,
                "away_score": None,
            },
        ]
    )
    players = pl.DataFrame(
        [row("a", "qb1", "QB", dropbacks=35, epa=5), row("a", "wr1", "WR", ppr=25)]
    )
    report = {(2024, 2, "KC"): {"qb1": "Out"}}
    result = av.availability_by_game(games, players, report, {})
    assert result["a"]["KC"].stars == []  # nothing known before the first game
    kc = result["b"]["KC"]
    assert [s.name for s in kc.stars] == ["qb1", "wr1"]
    assert kc.qb_out and not kc.missing("OFF")
