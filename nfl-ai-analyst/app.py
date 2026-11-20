"""Local UI to see what data the analyst has and what it predicts.

streamlit run app.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import polars as pl
import streamlit as st
from dotenv import load_dotenv

from nfl_analyst import analysis, report
from nfl_analyst.data import current_season, load_games, load_injuries
from nfl_analyst.teams import TEAM_NAMES

load_dotenv()
st.set_page_config(page_title="NFL AI Analyst", layout="wide")
st.title("NFL AI Analyst")
st.caption("Todo se calcula cuando lo pides. Datos: nflverse. Noticias: Google News + Claude.")


@st.cache_data(ttl=6 * 3600, show_spinner="Cargando datos de nflverse...")
def games_df() -> pl.DataFrame:
    return load_games()


@st.cache_data(ttl=6 * 3600)
def injuries_df() -> pl.DataFrame:
    return load_injuries()


def win_probability_chart(week_reports: list[analysis.GameReport]) -> alt.Chart:
    """One bar per game: away team share on the left, home team share on the right."""
    rows = []
    for r in week_reports:
        p = r.prediction
        label = f"{p.away_team} @ {p.home_team}"
        rows.append(
            {
                "game": label,
                "team": p.away_team,
                "side": "Visitante",
                "prob": p.away_win_prob,
                "start": 0.0,
                "end": p.away_win_prob,
            }
        )
        rows.append(
            {
                "game": label,
                "team": p.home_team,
                "side": "Local",
                "prob": p.home_win_prob,
                "start": p.away_win_prob,
                "end": 1.0,
            }
        )
    data = pd.DataFrame(rows)
    order = [f"{r.prediction.away_team} @ {r.prediction.home_team}" for r in week_reports]
    y = alt.Y("game:N", sort=order, title=None)
    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "start:Q", axis=alt.Axis(format="%"), title=None, scale=alt.Scale(domain=[0, 1])
            ),
            x2="end:Q",
            y=y,
            color=alt.Color(
                "side:N",
                scale=alt.Scale(domain=["Visitante", "Local"], range=["#9aa5b1", "#1f6feb"]),
                title=None,
            ),
            tooltip=["game", "team", alt.Tooltip("prob:Q", format=".1%")],
        )
    )
    labels = (
        alt.Chart(data)
        .mark_text(color="white", fontWeight="bold")
        .encode(x=alt.X("mid:Q"), y=y, text="label:N")
        .transform_calculate(
            mid="(datum.start + datum.end) / 2",
            label="datum.team + ' ' + format(datum.prob, '.0%')",
        )
    )
    rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
    return (bars + labels + rule).properties(height=32 * len(order))


if st.sidebar.button("Recargar datos"):
    st.cache_data.clear()
    analysis.refresh()

games = games_df()
season = current_season()
played = games.filter(pl.col("home_score").is_not_null())

tab_data, tab_week, tab_game, tab_ratings, tab_validation = st.tabs(
    ["Datos cargados", "Semana", "Partido", "Ratings", "Validación"]
)

with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temporadas", f"{games['season'].min()}–{games['season'].max()}")
    c2.metric("Partidos jugados", f"{played.height:,}")
    c3.metric(
        "Pendientes esta temporada",
        games.filter((pl.col("season") == season) & pl.col("home_score").is_null()).height,
    )
    last = played.tail(1).row(0, named=True)
    c4.metric(
        "Último resultado",
        f"{last['away_team']} {last['away_score']}–{last['home_score']} {last['home_team']}",
    )

    st.subheader("Partidos por temporada")
    per_season = played.group_by("season").agg(pl.len().alias("partidos")).sort("season")
    st.bar_chart(per_season.to_pandas(), x="season", y="partidos")

    st.subheader("Explorar partidos")
    f1, f2 = st.columns(2)
    sel_season = f1.selectbox("Temporada", sorted(games["season"].unique(), reverse=True))
    sel_team = f2.selectbox("Equipo", ["Todos", *sorted(TEAM_NAMES)])
    view = games.filter(pl.col("season") == sel_season)
    if sel_team != "Todos":
        view = view.filter((pl.col("home_team") == sel_team) | (pl.col("away_team") == sel_team))
    st.dataframe(
        view.select(
            "week",
            "gameday",
            "away_team",
            "away_score",
            "home_team",
            "home_score",
            "spread_line",
            "total_line",
            "roof",
            "temp",
            "wind",
        ).to_pandas(),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader(f"Reportes de lesiones {season}")
    inj = injuries_df()
    st.write(f"{inj.height:,} registros, semanas {inj['week'].min()}–{inj['week'].max()}")
    st.dataframe(
        inj.filter(pl.col("week") == inj["week"].max())
        .select(
            "team",
            "full_name",
            "position",
            "report_status",
            "practice_status",
            "report_primary_injury",
        )
        .to_pandas(),
        hide_index=True,
        use_container_width=True,
    )

with tab_week:
    pending = games.filter((pl.col("season") == season) & pl.col("home_score").is_null())
    weeks = sorted(pending["week"].unique())
    if not weeks:
        st.info("No hay partidos pendientes esta temporada.")
    else:
        week = st.selectbox("Semana", weeks)
        week_reports = analysis.predict_week(week)
        st.subheader("Probabilidad de ganar")
        st.altair_chart(win_probability_chart(week_reports), use_container_width=True)
        st.markdown(report.week_markdown(week_reports))

with tab_game:
    pending = games.filter((pl.col("season") == season) & pl.col("home_score").is_null())
    options = {
        f"Sem {g['week']}: {g['away_team']} @ {g['home_team']} ({g['gameday']})": g
        for g in pending.iter_rows(named=True)
    }
    if options:
        choice = st.selectbox("Partido", list(options))
        game = options[choice]
        with_news = st.checkbox("Incluir lesiones y noticias", value=True)
        if st.button("Analizar"):
            with st.spinner("Calculando y buscando noticias..."):
                result = analysis.analyze_game(game["home_team"], game["away_team"], with_news)
            st.markdown(report.game_markdown(result))
            if with_news and result.summary is None:
                st.caption("Agrega ANTHROPIC_API_KEY en .env para el resumen con Claude.")

with tab_ratings:
    ratings = analysis.ratings_table()
    st.bar_chart(ratings.to_pandas(), x="team", y="elo", sort="-elo")
    st.dataframe(ratings.to_pandas(), hide_index=True, use_container_width=True)

with tab_validation:
    st.write(
        "Cada predicción usa solo partidos jugados antes de ella. "
        "La línea de cierre del mercado sirve como referencia."
    )
    metrics = analysis.backtest()
    st.markdown(report.backtest_markdown(metrics, f"{season - 3}–{season - 1}"))
