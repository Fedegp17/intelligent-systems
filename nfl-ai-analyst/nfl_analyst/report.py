"""Plain-text / Markdown output."""

from __future__ import annotations

from nfl_analyst.analysis import GameReport
from nfl_analyst.teams import TEAM_NAMES


def _spread(team: str, margin: float) -> str:
    """Projected line for the favorite, e.g. 'KC -3.5'."""
    rounded = round(abs(margin) * 2) / 2
    return "Pick'em" if rounded == 0 else f"{team} -{rounded:g}"


def game_markdown(report: GameReport) -> str:
    p = report.prediction
    favorite = p.home_team if p.margin >= 0 else p.away_team
    lines = [
        f"## {TEAM_NAMES[p.away_team]} @ {TEAM_NAMES[p.home_team]}",
        f"Semana {p.week} · {report.kickoff}" + (" · campo neutral" if p.neutral else ""),
        "",
        "| | Visitante | Local |",
        "|---|---|---|",
        f"| Equipo | {p.away_team} | {p.home_team} |",
        f"| Probabilidad de ganar | {p.away_win_prob:.1%} | {p.home_win_prob:.1%} |",
        f"| Puntos proyectados | {p.away_points:.1f} | {p.home_points:.1f} |",
        f"| Elo | {p.away_elo:.0f} | {p.home_elo:.0f} |",
        "",
        (
            f"- **Margen proyectado:** {_spread(favorite, p.margin)} "
            f"(Elo {p.elo_margin:+.1f}, modelo de puntos {p.points_margin:+.1f}, positivo = local)"
        ),
        f"- **Total proyectado:** {p.total:.1f} puntos",
    ]
    if report.injuries is not None:
        lines += ["", "### Reporte oficial de lesiones"]
        if report.injuries.is_empty():
            lines.append("Sin jugadores con estatus en el último reporte.")
        else:
            lines += [
                "| Equipo | Jugador | Pos | Estatus juego | Práctica | Lesión |",
                "|---|---|---|---|---|---|",
            ]
            for r in report.injuries.iter_rows(named=True):
                lines.append(
                    f"| {r['team']} | {r['full_name']} | {r['position']} | "
                    f"{r['report_status'] or 'pendiente'} | {r['practice_status'] or ''} | "
                    f"{r['report_primary_injury'] or ''} |"
                )
    if report.summary:
        lines += ["", "### Resumen de noticias (Claude)", report.summary]
    if report.headlines:
        lines += ["", "### Titulares recientes"]
        for h in report.headlines:
            date = f"{h.published:%Y-%m-%d}" if h.published else ""
            lines.append(f"- [{h.team}] [{h.title}]({h.url}) · {h.source} {date}")
    for error in report.news_errors:
        lines.append(f"- _{error}_")
    return "\n".join(lines)


def week_markdown(reports: list[GameReport]) -> str:
    if not reports:
        return "No hay partidos pendientes esta temporada."
    lines = [
        f"## Semana {reports[0].prediction.week}",
        "",
        "| Partido | Fecha | Favorito | Prob. | Margen | Total |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        p = r.prediction
        fav = p.home_team if p.home_win_prob >= 0.5 else p.away_team
        prob = max(p.home_win_prob, p.away_win_prob)
        lines.append(
            f"| {p.away_team} @ {p.home_team} | {r.kickoff} | {fav} | {prob:.0%} | "
            f"{_spread(p.home_team if p.margin >= 0 else p.away_team, p.margin)} | {p.total:.1f} |"
        )
    return "\n".join(lines)


def backtest_markdown(metrics: dict[str, float], seasons: str) -> str:
    return "\n".join(
        [
            f"## Validación del modelo ({seasons}, {metrics['games']} partidos)",
            "",
            "| Métrica | Modelo | Mercado (línea de cierre) |",
            "|---|---|---|",
            f"| Aciertos de ganador | {metrics['accuracy']:.1%} | |",
            f"| Brier score (menor es mejor) | {metrics['brier']:.3f} | |",
            f"| Error medio del margen (pts) | {metrics['margin_mae']:.2f} | {metrics['market_margin_mae']:.2f} |",
            f"| Error medio del total (pts) | {metrics['total_mae']:.2f} | {metrics['market_total_mae']:.2f} |",
        ]
    )
