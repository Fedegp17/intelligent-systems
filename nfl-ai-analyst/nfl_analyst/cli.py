"""On-demand commands.

nfl-analyst game KC LV        prediction + injuries + news for one game
nfl-analyst week [--week 12]  predictions for every game of a week
nfl-analyst ratings           current Elo and points ratings
nfl-analyst backtest          how the model did in recent seasons
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from nfl_analyst import analysis, report


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="nfl-analyst", description="NFL AI Analyst")
    sub = parser.add_subparsers(dest="command", required=True)

    game = sub.add_parser("game", help="Analizar un partido: nfl-analyst game KC LV")
    game.add_argument("team_a")
    game.add_argument("team_b")
    game.add_argument(
        "--no-news", action="store_true", help="Solo el cálculo, sin lesiones ni noticias"
    )

    week = sub.add_parser("week", help="Predicciones de la semana")
    week.add_argument("--week", type=int)

    sub.add_parser("ratings", help="Ratings actuales de los equipos")
    sub.add_parser("backtest", help="Validación del modelo en temporadas recientes")

    args = parser.parse_args(argv)
    try:
        if args.command == "game":
            print(
                report.game_markdown(
                    analysis.analyze_game(args.team_a, args.team_b, not args.no_news)
                )
            )
        elif args.command == "week":
            print(report.week_markdown(analysis.predict_week(args.week)))
        elif args.command == "ratings":
            with __import__("polars").Config(tbl_rows=40):
                print(analysis.ratings_table())
        elif args.command == "backtest":
            season = analysis.current_season()
            metrics = analysis.backtest()
            print(report.backtest_markdown(metrics, f"{season - 3}–{season - 1}"))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
