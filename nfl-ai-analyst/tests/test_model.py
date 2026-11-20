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
