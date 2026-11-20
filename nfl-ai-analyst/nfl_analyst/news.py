"""Injury reports (official, via nflverse) and recent news headlines per team.

Headlines come from Google News RSS. When ANTHROPIC_API_KEY is set, Claude
summarizes what matters for the game. Claude only describes the news; it
never changes the model's numbers.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import httpx
import polars as pl

from nfl_analyst.data import load_injuries
from nfl_analyst.teams import TEAM_NAMES

log = logging.getLogger(__name__)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
DEFAULT_MODEL = "claude-opus-5-5"
REPORT_STATUS_ORDER = {"Out": 0, "Doubtful": 1, "Questionable": 2}


@dataclass
class Headline:
    team: str
    title: str
    source: str
    url: str
    published: datetime | None


# ---------------------------------------------------------------------------
# Injuries
# ---------------------------------------------------------------------------


def injury_report(
    teams: list[str], season: int | None = None, week: int | None = None
) -> pl.DataFrame:
    """Latest official injury report for each team (players with a game status)."""
    injuries = load_injuries(season).filter(pl.col("team").is_in(teams))
    if injuries.is_empty():
        return injuries
    if week is None:
        week = injuries["week"].max()
    latest = injuries.filter(pl.col("week") == week)
    # Early in the week only practice participation exists; game status comes Friday.
    return (
        latest.filter(
            pl.col("report_status").is_not_null()
            | pl.col("practice_status").str.contains("(?i)did not|limited")
        )
        .with_columns(
            pl.col("report_status")
            .fill_null("")
            .replace_strict(REPORT_STATUS_ORDER, default=9)
            .alias("_order")
        )
        .sort(["team", "_order", "position"])
        .select(
            "team",
            "week",
            "full_name",
            "position",
            "report_status",
            "report_primary_injury",
            "practice_status",
        )
    )


# ---------------------------------------------------------------------------
# Headlines
# ---------------------------------------------------------------------------


def fetch_headlines(
    team: str, days: int = 3, limit: int = 8, client: httpx.Client | None = None
) -> list[Headline]:
    query = quote_plus(f'"{TEAM_NAMES[team]}" when:{days}d')
    client = client or httpx.Client(timeout=15, follow_redirects=True)
    response = client.get(GOOGLE_NEWS_RSS.format(query=query))
    response.raise_for_status()
    return parse_rss(response.text, team)[:limit]


def parse_rss(xml_text: str, team: str) -> list[Headline]:
    root = ET.fromstring(xml_text)
    items = []
    for item in root.iter("item"):
        published = item.findtext("pubDate")
        source = item.find("source")
        try:
            published_at = parsedate_to_datetime(published) if published else None
        except (TypeError, ValueError):
            published_at = None
        items.append(
            Headline(
                team=team,
                title=(item.findtext("title") or "").strip(),
                source=source.text.strip() if source is not None and source.text else "",
                url=(item.findtext("link") or "").strip(),
                published=published_at,
            )
        )
    epoch = datetime.min.replace(tzinfo=UTC)
    return sorted(items, key=lambda h: h.published or epoch, reverse=True)


def headlines_for(teams: list[str]) -> tuple[list[Headline], list[str]]:
    """Headlines for every team; network errors are reported, not raised."""
    headlines, errors = [], []
    with httpx.Client(timeout=15, follow_redirects=True) as client:
        for team in teams:
            try:
                headlines.extend(fetch_headlines(team, client=client))
            except Exception as exc:  # noqa: BLE001 - news is optional context
                errors.append(f"{team}: no se pudieron leer noticias ({type(exc).__name__})")
    return headlines, errors


# ---------------------------------------------------------------------------
# Claude summary
# ---------------------------------------------------------------------------

SUMMARY_INSTRUCTIONS = """Eres analista de NFL. Con el reporte oficial de lesiones y los titulares
recientes de abajo, escribe en español un resumen breve (máximo 8 viñetas) de lo que
importa para el partido {matchup}: bajas o dudas relevantes (sobre todo QB, receptores,
línea ofensiva, defensivos clave), cambios de titulares, clima u otras noticias.
Reglas: usa solo la información dada; si un titular no es claro, dilo; no inventes
estados de jugadores; no des probabilidades ni recomendaciones de apuesta."""


def summarize(matchup: str, injuries: pl.DataFrame, headlines: list[Headline]) -> str | None:
    """Claude summary of injuries + headlines, or None when no API key is set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    injury_lines = [
        f"- {r['team']} {r['position']} {r['full_name']}: "
        f"{r['report_status'] or 'estatus pendiente'}, práctica: {r['practice_status'] or 'n/d'} "
        f"({r['report_primary_injury'] or 'sin detalle'})"
        for r in injuries.iter_rows(named=True)
    ] or ["- (sin reporte de lesiones)"]
    headline_lines = [
        f"- [{h.team}] {h.title} ({h.source}, {h.published:%Y-%m-%d})"
        if h.published
        else f"- [{h.team}] {h.title} ({h.source})"
        for h in headlines
    ] or ["- (sin titulares)"]
    content = (
        "REPORTE OFICIAL DE LESIONES\n"
        + "\n".join(injury_lines)
        + "\n\nTITULARES RECIENTES\n"
        + "\n".join(headline_lines)
    )
    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=os.environ.get("NFL_ANALYST_MODEL", DEFAULT_MODEL),
            max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SUMMARY_INSTRUCTIONS.format(matchup=matchup),
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.APIError as exc:
        log.warning("Claude summary failed: %s", exc)
        return f"(No se pudo generar el resumen con Claude: {type(exc).__name__})"
    if response.stop_reason == "refusal":
        return "(Claude no generó el resumen para esta solicitud.)"
    return "".join(b.text for b in response.content if b.type == "text").strip()
