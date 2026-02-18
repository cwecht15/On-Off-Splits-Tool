#!/usr/bin/env python
"""Load required NFL CSVs into typed Postgres tables for the Streamlit app.

Usage (PowerShell):
    $env:POSTGRES_DSN = "postgresql://user:pass@host/db?sslmode=require"
    python scripts/load_postgres_typed.py --csv-dir .
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Callable

import psycopg


ParserFn = Callable[[str | None], object]


def as_text(value: str | None) -> object:
    if value is None:
        return None
    v = value.strip()
    return v if v else None


def as_int(value: str | None) -> object:
    v = as_text(value)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return int(float(v))


def as_float(value: str | None) -> object:
    v = as_text(value)
    if v is None:
        return None
    return float(v)


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def create_table(cur: psycopg.Cursor, table_name: str, columns: list[tuple[str, str]]) -> None:
    cur.execute(f"DROP TABLE IF EXISTS {qident(table_name)}")
    ddl_cols = ", ".join(f"{qident(col)} {col_type}" for col, col_type in columns)
    cur.execute(f"CREATE TABLE {qident(table_name)} ({ddl_cols})")


def load_table(
    conn: psycopg.Connection,
    csv_path: Path,
    table_name: str,
    columns: list[tuple[str, str]],
    parsers: dict[str, ParserFn],
) -> None:
    print(f"Loading {csv_path.name} -> {table_name}")
    with conn.cursor() as cur:
        create_table(cur, table_name, columns)
    conn.commit()

    col_names = [c[0] for c in columns]
    copy_sql = (
        f"COPY {qident(table_name)} ({', '.join(qident(c) for c in col_names)}) "
        "FROM STDIN"
    )

    loaded = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        with conn.cursor() as cur:
            with cur.copy(copy_sql) as cp:
                for row in reader:
                    out = []
                    for col in col_names:
                        raw = row.get(col)
                        parser = parsers.get(col, as_text)
                        try:
                            out.append(parser(raw))
                        except Exception as exc:  # pragma: no cover
                            raise ValueError(
                                f"Failed parsing column '{col}' value '{raw}' in table '{table_name}'"
                            ) from exc
                    cp.write_row(tuple(out))
                    loaded += 1
                    if loaded % 250000 == 0:
                        print(f"  {table_name}: {loaded:,} rows...")
        conn.commit()

    print(f"  {table_name}: done ({loaded:,} rows)")


def create_indexes(conn: psycopg.Connection) -> None:
    print("Creating indexes...")
    idx_sql = [
        'CREATE INDEX IF NOT EXISTS idx_part_season_week_team_role ON "participation" ("season","week","team","role")',
        'CREATE INDEX IF NOT EXISTS idx_part_game_play ON "participation" ("gameId","playId")',
        'CREATE INDEX IF NOT EXISTS idx_pbp_season_week ON "play_by_play_data" ("season","week")',
        'CREATE INDEX IF NOT EXISTS idx_pbp_off_def ON "play_by_play_data" ("offense","defense")',
        'CREATE INDEX IF NOT EXISTS idx_pbp_game_play ON "play_by_play_data" ("game_id","play_id")',
        'CREATE INDEX IF NOT EXISTS idx_epa_game_play ON "epa" ("gameId","playId")',
        'CREATE INDEX IF NOT EXISTS idx_roster_season_week_team ON "weekly_rosters" ("Season","Week","CurrentClub")',
        'CREATE INDEX IF NOT EXISTS idx_pp_play_id ON "pp_data" ("play id")',
    ]
    with conn.cursor() as cur:
        for stmt in idx_sql:
            cur.execute(stmt)
    conn.commit()
    with conn.cursor() as cur:
        for table in ["participation", "play_by_play_data", "epa", "weekly_rosters", "pp_data"]:
            cur.execute(f"ANALYZE {qident(table)}")
    conn.commit()
    print("Indexes + ANALYZE complete.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default=os.environ.get("POSTGRES_DSN", "").strip())
    parser.add_argument("--csv-dir", default=".", help="Directory containing CSV files")
    args = parser.parse_args()

    if not args.dsn:
        print("Missing DSN. Set --dsn or POSTGRES_DSN.", file=sys.stderr)
        return 2

    base = Path(args.csv_dir).resolve()
    files = {
        "participation": base / "participation.csv",
        "play_by_play_data": base / "play_by_play_data.csv",
        "epa": base / "epa.csv",
        "weekly_rosters": base / "weekly_rosters.csv",
        "pp_data": base / "pp_data.csv",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        print("Missing required CSV files:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 2

    participation_cols = [
        ("gameId", "TEXT"),
        ("season", "INTEGER"),
        ("seasonType", "TEXT"),
        ("week", "INTEGER"),
        ("playId", "TEXT"),
        ("team", "TEXT"),
        ("role", "TEXT"),
    ] + [(f"P{i}", "TEXT") for i in range(1, 16)]
    participation_parsers = {"season": as_int, "week": as_int}

    pbp_cols = [
        ("season", "INTEGER"),
        ("seas_type", "TEXT"),
        ("week", "INTEGER"),
        ("game_id", "TEXT"),
        ("play_id", "TEXT"),
        ("offense", "TEXT"),
        ("defense", "TEXT"),
        ("passer_id", "TEXT"),
        ("target", "TEXT"),
        ("receiver_id", "TEXT"),
        ("runner_id", "TEXT"),
        ("no_rush_player_id", "TEXT"),
        ("no_reception_player_1_id", "TEXT"),
        ("no_reception_player_2_id", "TEXT"),
        ("fumble_lost_player_1_id", "TEXT"),
        ("fumble_lost_player_2_id", "TEXT"),
        ("offense_score", "DOUBLE PRECISION"),
        ("defense_score", "DOUBLE PRECISION"),
        ("quarter", "INTEGER"),
        ("down", "INTEGER"),
        ("distance", "DOUBLE PRECISION"),
        ("yards_to_score", "DOUBLE PRECISION"),
        ("pass_play", "INTEGER"),
        ("run_play", "INTEGER"),
        ("no_play", "INTEGER"),
        ("spiked_ball", "INTEGER"),
        ("kneel_down", "INTEGER"),
        ("two_point_att", "INTEGER"),
        ("kickoff", "INTEGER"),
        ("punt", "INTEGER"),
        ("field_goal", "INTEGER"),
        ("extra_point", "INTEGER"),
        ("attempt", "INTEGER"),
        ("dropback", "INTEGER"),
        ("sack", "INTEGER"),
        ("scramble", "INTEGER"),
        ("rec_yards", "DOUBLE PRECISION"),
        ("reception", "INTEGER"),
        ("intercepted", "INTEGER"),
        ("passing_touchdown", "INTEGER"),
        ("rush_attempt", "INTEGER"),
        ("rushing_touchdown", "INTEGER"),
        ("no_rush_yards", "DOUBLE PRECISION"),
        ("no_rush_touchdown", "INTEGER"),
        ("no_reception_yards_1", "DOUBLE PRECISION"),
        ("no_reception_yards_2", "DOUBLE PRECISION"),
        ("no_reception_touchdown_1", "INTEGER"),
        ("no_reception_touchdown_2", "INTEGER"),
        ("passing_yards", "DOUBLE PRECISION"),
        ("rushing_yards", "DOUBLE PRECISION"),
    ]
    pbp_int_cols = {
        "season",
        "week",
        "quarter",
        "down",
        "pass_play",
        "run_play",
        "no_play",
        "spiked_ball",
        "kneel_down",
        "two_point_att",
        "kickoff",
        "punt",
        "field_goal",
        "extra_point",
        "attempt",
        "dropback",
        "sack",
        "scramble",
        "reception",
        "intercepted",
        "passing_touchdown",
        "rush_attempt",
        "rushing_touchdown",
        "no_rush_touchdown",
        "no_reception_touchdown_1",
        "no_reception_touchdown_2",
    }
    pbp_parsers = {k: as_int for k in pbp_int_cols}
    for c, _ in pbp_cols:
        pbp_parsers.setdefault(c, as_float if c not in pbp_int_cols and c not in {
            "seas_type",
            "game_id",
            "play_id",
            "offense",
            "defense",
            "passer_id",
            "target",
            "receiver_id",
            "runner_id",
            "no_rush_player_id",
            "no_reception_player_1_id",
            "no_reception_player_2_id",
            "fumble_lost_player_1_id",
            "fumble_lost_player_2_id",
        } else as_text)

    epa_cols = [("gameId", "TEXT"), ("playId", "TEXT"), ("epa", "DOUBLE PRECISION")]
    epa_parsers = {"epa": as_float}

    roster_cols = [
        ("Season", "INTEGER"),
        ("SeasonType", "TEXT"),
        ("Week", "INTEGER"),
        ("GsisID", "TEXT"),
        ("FirstName", "TEXT"),
        ("LastName", "TEXT"),
        ("FootballName", "TEXT"),
        ("CurrentClub", "TEXT"),
        ("Position", "TEXT"),
        ("StatusShortDescription", "TEXT"),
    ]
    roster_parsers = {"Season": as_int, "Week": as_int}

    pp_cols = [("play id", "BIGINT"), ("offense", "TEXT"), ("personnel", "INTEGER")]
    pp_parsers = {"play id": as_int, "personnel": as_int}

    with psycopg.connect(args.dsn, connect_timeout=20) as conn:
        load_table(conn, files["participation"], "participation", participation_cols, participation_parsers)
        load_table(conn, files["play_by_play_data"], "play_by_play_data", pbp_cols, pbp_parsers)
        load_table(conn, files["epa"], "epa", epa_cols, epa_parsers)
        load_table(conn, files["weekly_rosters"], "weekly_rosters", roster_cols, roster_parsers)
        load_table(conn, files["pp_data"], "pp_data", pp_cols, pp_parsers)
        create_indexes(conn)

    print("All tables loaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
