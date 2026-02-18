import io
import os
import re
import tempfile
from urllib.error import URLError
from urllib.request import urlopen
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import psycopg

try:
    from fpdf import FPDF
except Exception:  # pragma: no cover
    FPDF = None

st.set_page_config(page_title="NFL On/Off Splits", layout="wide")

DATA_DIR = Path(__file__).resolve().parent
DATA_CACHE_DIR = Path.home() / ".cache" / "on_off_tool_data"
REQUIRED_DATA_FILES = [
    "participation.csv",
    "play_by_play_data.csv",
    "epa.csv",
    "weekly_rosters.csv",
    "pp_data.csv",
]
PG_TABLE_ENV_PREFIX = "POSTGRES_TABLE_"


def normalize_team(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def normalize_str(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def normalize_position(series: pd.Series) -> pd.Series:
    pos = normalize_str(series).str.upper()
    pos = pos.replace(
        {
            "KICKER": "K",
            "PK": "K",
            "PUNTER": "P",
        }
    )
    return pos


def map_display_week(week: pd.Series, season_type: pd.Series) -> pd.Series:
    week_num = pd.to_numeric(week, errors="coerce")
    season_type_norm = normalize_str(season_type).str.upper()
    return week_num + np.where(season_type_norm == "POST", 18, 0)


def _data_url_key(filename: str) -> str:
    return f"DATA_URL_{filename.replace('.', '_').upper()}"


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _table_ident(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Postgres table name is empty.")
    parts = name.strip().split(".")
    for p in parts:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p):
            raise ValueError(f"Invalid Postgres table identifier part '{p}'.")
    return ".".join(f'"{p}"' for p in parts)


def _extract_columns(
    raw_df: pd.DataFrame,
    column_candidates: dict[str, list[str]],
    dataset_label: str,
) -> pd.DataFrame:
    normalized_lookup = {_normalize_key(c): c for c in raw_df.columns}
    out = pd.DataFrame(index=raw_df.index)
    missing: list[str] = []
    for target, candidates in column_candidates.items():
        found = None
        for candidate in candidates:
            key = _normalize_key(candidate)
            if key in normalized_lookup:
                found = normalized_lookup[key]
                break
        if found is None:
            missing.append(target)
        else:
            out[target] = raw_df[found]

    if missing:
        preview = ", ".join(sorted(str(c) for c in raw_df.columns[:30]))
        raise KeyError(
            f"Postgres dataset '{dataset_label}' missing required columns: {missing}. "
            f"Available columns (first 30): {preview}"
        )
    return out


def get_postgres_config() -> dict[str, str] | None:
    dsn = os.environ.get("POSTGRES_DSN", "").strip()

    try:
        if "postgres" in st.secrets:
            pg_sec = dict(st.secrets["postgres"])
            dsn = str(pg_sec.get("dsn", dsn)).strip()
    except Exception:
        pg_sec = {}

    if not dsn:
        return None

    table_defaults = {
        "participation": "participation",
        "play_by_play_data": "play_by_play_data",
        "epa": "epa",
        "weekly_rosters": "weekly_rosters",
        "pp_data": "pp_data",
    }

    table_map = table_defaults.copy()
    for logical_name, default_table in table_defaults.items():
        env_key = f"{PG_TABLE_ENV_PREFIX}{logical_name.upper()}"
        if os.environ.get(env_key, "").strip():
            table_map[logical_name] = os.environ[env_key].strip()

    try:
        if "postgres" in st.secrets:
            pg_sec = dict(st.secrets["postgres"])
            for logical_name, default_table in table_defaults.items():
                table_map[logical_name] = str(pg_sec.get(f"{logical_name}_table", table_map[logical_name])).strip()
                if not table_map[logical_name]:
                    table_map[logical_name] = default_table
    except Exception:
        pass

    return {"dsn": dsn, **table_map}


def load_raw_data_from_postgres(pg_cfg: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def fetch_cols(conn: psycopg.Connection, table_name: str, cols: list[str]) -> pd.DataFrame:
        quoted_cols = ", ".join(f'"{c}"' for c in cols)
        sql = f"SELECT {quoted_cols} FROM {_table_ident(table_name)}"
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [desc.name for desc in cur.description]
        return pd.DataFrame(rows, columns=cols)

    part_cols = [
        "gameId",
        "season",
        "seasonType",
        "week",
        "playId",
        "team",
        "role",
    ] + [f"P{i}" for i in range(1, 16)]
    pbp_cols = [
        "season",
        "seas_type",
        "week",
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
        "offense_score",
        "defense_score",
        "quarter",
        "down",
        "distance",
        "yards_to_score",
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
        "rec_yards",
        "reception",
        "intercepted",
        "passing_touchdown",
        "rush_attempt",
        "rushing_touchdown",
        "no_rush_yards",
        "no_rush_touchdown",
        "no_reception_yards_1",
        "no_reception_yards_2",
        "no_reception_touchdown_1",
        "no_reception_touchdown_2",
        "passing_yards",
        "rushing_yards",
    ]
    epa_cols = ["gameId", "playId", "epa"]
    roster_cols = [
        "Season",
        "SeasonType",
        "Week",
        "GsisID",
        "FirstName",
        "LastName",
        "FootballName",
        "CurrentClub",
        "Position",
        "StatusShortDescription",
    ]
    pp_cols = ["play id", "offense", "personnel"]

    with psycopg.connect(pg_cfg["dsn"], connect_timeout=15) as conn:
        participation = fetch_cols(conn, pg_cfg["participation"], part_cols)
        pbp = fetch_cols(conn, pg_cfg["play_by_play_data"], pbp_cols)
        epa = fetch_cols(conn, pg_cfg["epa"], epa_cols)
        rosters = fetch_cols(conn, pg_cfg["weekly_rosters"], roster_cols)
        pp = fetch_cols(conn, pg_cfg["pp_data"], pp_cols)

    pbp = pbp.rename(columns={"game_id": "gameId", "play_id": "playId", "seas_type": "seasonType"})
    return participation, pbp, epa, rosters, pp


def get_data_url_map() -> dict[str, str]:
    url_map: dict[str, str] = {}

    # Streamlit secrets (preferred for cloud deploy)
    try:
        if "data_urls" in st.secrets:
            for k, v in dict(st.secrets["data_urls"]).items():
                if isinstance(k, str) and isinstance(v, str) and v.strip():
                    url_map[k.strip()] = v.strip()
    except Exception:
        pass

    # Environment variables
    for filename in REQUIRED_DATA_FILES:
        env_key = _data_url_key(filename)
        url_val = os.environ.get(env_key, "").strip()
        if url_val:
            url_map[filename] = url_val

    base_url = os.environ.get("DATA_BASE_URL", "").strip().rstrip("/")
    if base_url:
        for filename in REQUIRED_DATA_FILES:
            url_map.setdefault(filename, f"{base_url}/{filename}")

    return url_map


def ensure_data_file(filename: str) -> Path:
    local_path = DATA_DIR / filename
    if local_path.exists():
        return local_path

    url_map = get_data_url_map()
    url = url_map.get(filename, "").strip()
    if not url:
        raise FileNotFoundError(
            f"Missing required data file '{filename}'. "
            f"Provide it locally at '{local_path}' or configure URL via st.secrets['data_urls']['{filename}'] "
            f"or env var '{_data_url_key(filename)}'."
        )

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_path = DATA_CACHE_DIR / filename
    if cached_path.exists() and cached_path.stat().st_size > 0:
        return cached_path

    try:
        with urlopen(url, timeout=120) as resp:
            payload = resp.read()
    except URLError as exc:
        raise RuntimeError(f"Failed to download '{filename}' from '{url}': {exc}") from exc

    if not payload:
        raise RuntimeError(f"Downloaded empty payload for '{filename}' from '{url}'.")

    cached_path.write_bytes(payload)
    return cached_path


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pg_cfg = get_postgres_config()
    if pg_cfg:
        participation, pbp, epa, rosters, pp = load_raw_data_from_postgres(pg_cfg)
    else:
        part_path = ensure_data_file("participation.csv")
        pbp_path = ensure_data_file("play_by_play_data.csv")
        epa_path = ensure_data_file("epa.csv")
        roster_path = ensure_data_file("weekly_rosters.csv")
        pp_path = ensure_data_file("pp_data.csv")

        part_usecols = [
            "gameId",
            "season",
            "seasonType",
            "week",
            "playId",
            "team",
            "role",
        ] + [f"P{i}" for i in range(1, 16)]
        participation = pd.read_csv(
            part_path,
            usecols=part_usecols,
            dtype={
                "gameId": "string",
                "playId": "string",
                "season": "Int64",
                "seasonType": "string",
                "week": "Int64",
                "team": "string",
                "role": "string",
            },
            low_memory=False,
            memory_map=True,
        )

        pbp_usecols = [
            "season",
            "seas_type",
            "week",
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
            "offense_score",
            "defense_score",
            "quarter",
            "down",
            "distance",
            "yards_to_score",
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
            "rec_yards",
            "reception",
            "intercepted",
            "passing_touchdown",
            "rush_attempt",
            "rushing_touchdown",
            "no_rush_yards",
            "no_rush_touchdown",
            "no_reception_yards_1",
            "no_reception_yards_2",
            "no_reception_touchdown_1",
            "no_reception_touchdown_2",
            "passing_yards",
            "rushing_yards",
        ]
        pbp = pd.read_csv(
            pbp_path,
            usecols=pbp_usecols,
            dtype={
                "game_id": "string",
                "play_id": "string",
                "season": "Int64",
                "seas_type": "string",
                "week": "Int64",
                "offense": "string",
                "defense": "string",
                "passer_id": "string",
                "target": "string",
                "receiver_id": "string",
                "runner_id": "string",
                "no_rush_player_id": "string",
                "no_reception_player_1_id": "string",
                "no_reception_player_2_id": "string",
                "fumble_lost_player_1_id": "string",
                "fumble_lost_player_2_id": "string",
            },
            low_memory=False,
            memory_map=True,
        ).rename(
            columns={
                "game_id": "gameId",
                "play_id": "playId",
                "seas_type": "seasonType",
            }
        )

        epa = pd.read_csv(
            epa_path,
            usecols=["gameId", "playId", "epa"],
            dtype={"gameId": "string", "playId": "string", "epa": "float64"},
            low_memory=False,
            memory_map=True,
        )

        rosters = pd.read_csv(
            roster_path,
            usecols=[
                "Season",
                "SeasonType",
                "Week",
                "GsisID",
                "FirstName",
                "LastName",
                "FootballName",
                "CurrentClub",
                "Position",
                "StatusShortDescription",
            ],
            dtype={"GsisID": "string", "CurrentClub": "string", "SeasonType": "string"},
            low_memory=False,
            memory_map=True,
        )

        pp = pd.read_csv(
            pp_path,
            usecols=["play id", "offense", "personnel"],
            dtype={"offense": "string"},
            low_memory=False,
            memory_map=True,
        )

    epa["gameId"] = normalize_str(epa["gameId"].astype("string"))
    epa["playId"] = normalize_str(epa["playId"].astype("string"))
    epa["epa"] = pd.to_numeric(epa["epa"], errors="coerce")

    num_cols = [
        "offense_score",
        "defense_score",
        "quarter",
        "down",
        "distance",
        "yards_to_score",
        "pass_play",
        "run_play",
        "attempt",
        "dropback",
        "sack",
        "scramble",
        "rec_yards",
        "reception",
        "intercepted",
        "passing_touchdown",
        "rush_attempt",
        "rushing_touchdown",
        "no_rush_yards",
        "no_rush_touchdown",
        "no_reception_yards_1",
        "no_reception_yards_2",
        "no_reception_touchdown_1",
        "no_reception_touchdown_2",
        "no_play",
        "spiked_ball",
        "kneel_down",
        "two_point_att",
        "kickoff",
        "punt",
        "field_goal",
        "extra_point",
        "passing_yards",
        "rushing_yards",
    ]
    for col in num_cols:
        pbp[col] = pd.to_numeric(pbp[col], errors="coerce")
    pbp["season"] = pd.to_numeric(pbp["season"], errors="coerce").astype("Int64")
    pbp["week"] = pd.to_numeric(pbp["week"], errors="coerce").astype("Int64")
    participation["season"] = pd.to_numeric(participation["season"], errors="coerce").astype("Int64")
    participation["week"] = pd.to_numeric(participation["week"], errors="coerce").astype("Int64")

    pbp["seasonType"] = normalize_str(pbp["seasonType"]).str.upper()
    pbp["offense"] = normalize_team(pbp["offense"])
    pbp["defense"] = normalize_team(pbp["defense"])
    pbp["passer_id"] = normalize_str(pbp["passer_id"].astype("string"))
    pbp["target"] = normalize_str(pbp["target"].astype("string"))
    pbp["receiver_id"] = normalize_str(pbp["receiver_id"].astype("string"))
    pbp["runner_id"] = normalize_str(pbp["runner_id"].astype("string"))
    pbp["no_rush_player_id"] = normalize_str(pbp["no_rush_player_id"].astype("string"))
    pbp["no_reception_player_1_id"] = normalize_str(pbp["no_reception_player_1_id"].astype("string"))
    pbp["no_reception_player_2_id"] = normalize_str(pbp["no_reception_player_2_id"].astype("string"))
    pbp["fumble_lost_player_1_id"] = normalize_str(pbp["fumble_lost_player_1_id"].astype("string"))
    pbp["fumble_lost_player_2_id"] = normalize_str(pbp["fumble_lost_player_2_id"].astype("string"))
    pbp["yards_gained"] = pbp["passing_yards"].fillna(0) + pbp["rushing_yards"].fillna(0)
    pbp["display_week"] = map_display_week(pbp["week"], pbp["seasonType"])

    plays = pbp.merge(epa, on=["gameId", "playId"], how="inner")

    valid_play = (
        (plays["pass_play"] == 1) | (plays["run_play"] == 1)
    ) & (
        (plays["no_play"] != 1)
        & (plays["spiked_ball"] != 1)
        & (plays["kneel_down"] != 1)
        & (plays["two_point_att"] != 1)
        & (plays["kickoff"] != 1)
        & (plays["punt"] != 1)
        & (plays["field_goal"] != 1)
        & (plays["extra_point"] != 1)
    )
    plays = plays.loc[valid_play].copy()

    pp = pp.rename(columns={"play id": "playId"})
    pp["playId"] = pd.to_numeric(pp["playId"], errors="coerce").astype("Int64").astype("string")
    pp["offense"] = normalize_team(pp["offense"])
    pp["personnel"] = pd.to_numeric(pp["personnel"], errors="coerce").astype("Int64")
    pp = pp.dropna(subset=["playId"]).drop_duplicates(subset=["playId"], keep="first")

    plays = plays.merge(pp[["playId", "personnel"]], on="playId", how="left")

    player_cols = [f"P{i}" for i in range(1, 16)]
    participation["seasonType"] = normalize_str(participation["seasonType"]).str.upper()
    participation["team"] = normalize_team(participation["team"])
    participation["role"] = normalize_str(participation["role"]).str.lower()
    participation["display_week"] = map_display_week(participation["week"], participation["seasonType"])
    for col in player_cols:
        participation[col] = normalize_str(participation[col].astype("string"))

    rosters = rosters.rename(
        columns={
            "Season": "season",
            "SeasonType": "seasonType",
            "Week": "week",
            "GsisID": "player_id",
            "CurrentClub": "team",
            "Position": "position",
            "StatusShortDescription": "status",
        }
    )
    rosters["season"] = pd.to_numeric(rosters["season"], errors="coerce").astype("Int64")
    rosters["week"] = pd.to_numeric(rosters["week"], errors="coerce").astype("Int64")
    rosters["seasonType"] = normalize_str(rosters["seasonType"]).str.upper()
    rosters["team"] = normalize_team(rosters["team"])
    rosters["player_id"] = normalize_str(rosters["player_id"])
    rosters["position"] = normalize_position(rosters["position"])
    rosters["status"] = normalize_str(rosters["status"]).str.upper()
    rosters["display_week"] = map_display_week(rosters["week"], rosters["seasonType"])

    for col in ["seasonType", "offense", "defense"]:
        plays[col] = plays[col].astype("category")
    for col in ["seasonType", "team", "role"]:
        participation[col] = participation[col].astype("category")
    for col in ["seasonType", "team", "position"]:
        rosters[col] = rosters[col].astype("category")

    return plays, participation, rosters


def build_success_flag(df: pd.DataFrame) -> pd.Series:
    down = pd.to_numeric(df["down"], errors="coerce")
    distance = pd.to_numeric(df["distance"], errors="coerce")
    gain = pd.to_numeric(df["yards_gained"], errors="coerce").fillna(0)

    first = (down == 1) & (gain >= 0.4 * distance)
    second = (down == 2) & (gain >= 0.6 * distance)
    third_fourth = (down >= 3) & (gain >= distance)

    return (first | second | third_fourth) & (distance > 0)


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "Plays": 0,
            "EPA/play": np.nan,
            "Success Rate": np.nan,
            "Pass Rate": np.nan,
            "Run Rate": np.nan,
        }

    success = build_success_flag(df)
    return {
        "Plays": int(len(df)),
        "EPA/play": float(df["epa"].mean()),
        "Success Rate": float(success.mean()),
        "Pass Rate": float((df["pass_play"] == 1).mean()),
        "Run Rate": float((df["run_play"] == 1).mean()),
    }


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["Plays", "EPA/play", "Success Rate", "Pass Rate", "Run Rate"])

    tmp = df.copy()
    tmp["success_flag"] = build_success_flag(tmp).astype(float)
    grouped = (
        tmp.groupby(group_cols, dropna=False)
        .agg(
            Plays=("epa", "size"),
            **{
                "EPA/play": ("epa", "mean"),
                "Success Rate": ("success_flag", "mean"),
                "Pass Rate": ("pass_play", "mean"),
                "Run Rate": ("run_play", "mean"),
            },
        )
        .reset_index()
    )
    grouped["Pass Rate"] = grouped["Pass Rate"].astype(float)
    grouped["Run Rate"] = grouped["Run Rate"].astype(float)
    return grouped


def apply_common_filters(
    plays: pd.DataFrame,
    selected_downs: list[int],
    distance_range: tuple[int, int],
    selected_quarters: list[int],
    red_zone_filter: str,
    inside_10_filter: str,
) -> pd.DataFrame:
    filtered = plays

    if selected_downs:
        filtered = filtered[filtered["down"].isin(selected_downs)]

    if selected_quarters:
        filtered = filtered[filtered["quarter"].isin(selected_quarters)]

    min_dist, max_dist = distance_range
    filtered = filtered[(filtered["distance"] >= min_dist) & (filtered["distance"] <= max_dist)]

    if red_zone_filter == "Red Zone (<=20)":
        filtered = filtered[filtered["yards_to_score"] <= 20]
    elif red_zone_filter == "Outside Red Zone (>20)":
        filtered = filtered[filtered["yards_to_score"] > 20]

    if inside_10_filter == "Inside 10":
        filtered = filtered[filtered["yards_to_score"] <= 10]
    elif inside_10_filter == "Outside 10":
        filtered = filtered[filtered["yards_to_score"] > 10]

    return filtered


def role_margin(df: pd.DataFrame, role: str) -> pd.Series:
    if role == "offense":
        return df["offense_score"].fillna(0) - df["defense_score"].fillna(0)
    return df["defense_score"].fillna(0) - df["offense_score"].fillna(0)


def apply_margin_filter(df: pd.DataFrame, role: str, margin_range: tuple[int, int]) -> pd.DataFrame:
    low, high = margin_range
    margin = role_margin(df, role)
    return df[(margin >= low) & (margin <= high)]


def get_on_keys(
    part_wide: pd.DataFrame,
    team: str,
    role: str,
    selected_player_ids: list[str],
    on_mode: str,
) -> pd.DataFrame:
    if not selected_player_ids:
        return pd.DataFrame(columns=["gameId", "playId", "is_on"])

    player_cols = [f"P{i}" for i in range(1, 16)]
    subset = part_wide[
        (part_wide["team"] == team)
        & (part_wide["role"] == role)
    ][["gameId", "playId"] + player_cols]

    if subset.empty:
        return pd.DataFrame(columns=["gameId", "playId", "is_on"])

    clean_ids = [pid for pid in sorted(set(selected_player_ids)) if isinstance(pid, str) and pid]
    if not clean_ids:
        return pd.DataFrame(columns=["gameId", "playId", "is_on"])

    if on_mode == "All selected players on field":
        masks = [subset[player_cols].eq(pid).any(axis=1) for pid in clean_ids]
        on_mask = np.logical_and.reduce(masks) if masks else pd.Series(False, index=subset.index)
    else:
        on_mask = subset[player_cols].isin(clean_ids).any(axis=1)

    keys = subset.loc[on_mask, ["gameId", "playId"]].drop_duplicates()

    keys["is_on"] = 1
    return keys


def split_for_role(
    plays: pd.DataFrame,
    part_wide: pd.DataFrame,
    team: str,
    selected_player_ids: list[str],
    role: str,
    on_mode: str,
    margin_range: tuple[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if role == "offense":
        team_plays = plays[plays["offense"] == team]
    else:
        team_plays = plays[plays["defense"] == team]

    team_plays = apply_margin_filter(team_plays, role, margin_range)

    if team_plays.empty:
        return pd.DataFrame(columns=["Split", "Plays", "EPA/play", "Success Rate", "Pass Rate", "Run Rate"]), pd.DataFrame()

    on_keys = get_on_keys(part_wide, team, role, selected_player_ids, on_mode)
    team_plays = team_plays.merge(on_keys, on=["gameId", "playId"], how="left")
    team_plays["is_on"] = team_plays["is_on"].fillna(0).astype(int)

    on_summary = summarize(team_plays[team_plays["is_on"] == 1])
    off_summary = summarize(team_plays[team_plays["is_on"] == 0])

    split_table = pd.DataFrame(
        [
            {"Split": "On", **on_summary},
            {"Split": "Off", **off_summary},
            {
                "Split": "On - Off",
                "Plays": on_summary["Plays"] - off_summary["Plays"],
                "EPA/play": on_summary["EPA/play"] - off_summary["EPA/play"],
                "Success Rate": on_summary["Success Rate"] - off_summary["Success Rate"],
                "Pass Rate": on_summary["Pass Rate"] - off_summary["Pass Rate"],
                "Run Rate": on_summary["Run Rate"] - off_summary["Run Rate"],
            },
        ]
    )

    team_plays["personnel_label"] = team_plays["personnel"].astype("Int64").astype("string").fillna("Unknown")
    personnel_group = summarize_group(team_plays, ["personnel_label", "is_on"])
    if not personnel_group.empty:
        personnel_group["Split"] = np.where(personnel_group["is_on"] == 1, "On", "Off")
        personnel_group = personnel_group.drop(columns=["is_on"]).rename(columns={"personnel_label": "Personnel"})
        personnel_group = personnel_group.sort_values(["Personnel", "Split"])

    return split_table, personnel_group


def baseline_table(
    plays: pd.DataFrame,
    team: str,
    role: str,
    margin_range: tuple[int, int],
) -> pd.DataFrame:
    if role == "offense":
        team_df = plays[plays["offense"] == team]
        league_df = plays
    else:
        team_df = plays[plays["defense"] == team]
        league_df = plays

    team_df = apply_margin_filter(team_df, role, margin_range)
    league_df = apply_margin_filter(league_df, role, margin_range)

    return pd.DataFrame(
        [
            {"Baseline": "Team", **summarize(team_df)},
            {"Baseline": "League", **summarize(league_df)},
        ]
    )


def trend_table(
    plays: pd.DataFrame,
    part_wide: pd.DataFrame,
    team: str,
    selected_player_ids: list[str],
    role: str,
    on_mode: str,
    margin_range: tuple[int, int],
) -> pd.DataFrame:
    if role == "offense":
        team_plays = plays[plays["offense"] == team]
    else:
        team_plays = plays[plays["defense"] == team]

    team_plays = apply_margin_filter(team_plays, role, margin_range)
    if team_plays.empty:
        return pd.DataFrame(columns=["display_week", "On EPA/play", "Off EPA/play", "On Plays", "Off Plays"])

    on_keys = get_on_keys(part_wide, team, role, selected_player_ids, on_mode)
    team_plays = team_plays.merge(on_keys, on=["gameId", "playId"], how="left")
    team_plays["is_on"] = team_plays["is_on"].fillna(0).astype(int)

    agg = (
        team_plays.groupby(["display_week", "is_on"], dropna=False)
        .agg(plays=("epa", "size"), epa=("epa", "mean"))
        .reset_index()
    )

    epa_pivot = agg.pivot(index="display_week", columns="is_on", values="epa").rename(columns={1: "On EPA/play", 0: "Off EPA/play"})
    play_pivot = agg.pivot(index="display_week", columns="is_on", values="plays").rename(columns={1: "On Plays", 0: "Off Plays"})

    out = epa_pivot.join(play_pivot, how="outer").reset_index().sort_values("display_week")
    return out


def build_player_lookup(rosters: pd.DataFrame) -> pd.DataFrame:
    if rosters.empty:
        return pd.DataFrame(columns=["player_id", "player_name", "position"])

    preferred_first = normalize_str(rosters["FootballName"])
    fallback_first = normalize_str(rosters["FirstName"])
    last_name = normalize_str(rosters["LastName"])
    first_name = preferred_first.where(preferred_first.str.len() > 0, fallback_first)
    full_name = (first_name + " " + last_name).str.strip()
    full_name = full_name.where(full_name.str.len() > 0, first_name)

    lookup = pd.DataFrame(
        {
            "player_id": normalize_str(rosters["player_id"]),
            "player_name": full_name,
            "position": normalize_position(rosters["position"].astype("string")),
        }
    )
    lookup = lookup[lookup["player_id"].str.len() > 0]
    lookup = lookup[lookup["player_name"].str.len() > 0]
    lookup = lookup.drop_duplicates(subset=["player_id"], keep="first")
    return lookup


def top_player_diffs(
    plays: pd.DataFrame,
    part_wide: pd.DataFrame,
    rosters: pd.DataFrame,
    role: str,
    min_plays: int,
    leaderboard_team: str | None = None,
    top_n: int = 50,
) -> pd.DataFrame:
    def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        den = den.replace(0, np.nan)
        return num / den

    team_col = "offense" if role == "offense" else "defense"
    team_plays = plays.copy()
    if leaderboard_team:
        team_plays = team_plays[team_plays[team_col] == leaderboard_team]
    if team_plays.empty:
        return pd.DataFrame()

    team_plays["success_flag"] = build_success_flag(team_plays).astype(int)
    team_plays["pass_flag"] = (team_plays["pass_play"] == 1).astype(int)
    team_plays["run_flag"] = (team_plays["run_play"] == 1).astype(int)
    team_plays["pass_epa"] = np.where(team_plays["pass_flag"] == 1, team_plays["epa"], 0.0)
    team_plays["run_epa"] = np.where(team_plays["run_flag"] == 1, team_plays["epa"], 0.0)

    totals = (
        team_plays.groupby(team_col, dropna=False)
        .agg(
            total_plays=("epa", "size"),
            total_epa_sum=("epa", "sum"),
            total_success_sum=("success_flag", "sum"),
            total_pass_sum=("pass_flag", "sum"),
            total_run_sum=("run_flag", "sum"),
            total_pass_epa_sum=("pass_epa", "sum"),
            total_run_epa_sum=("run_epa", "sum"),
        )
        .reset_index()
        .rename(columns={team_col: "team"})
    )

    player_cols = [f"P{i}" for i in range(1, 16)]
    role_part = part_wide[part_wide["role"] == role][["gameId", "playId", "team"] + player_cols]
    play_core = team_plays[
        ["gameId", "playId", team_col, "epa", "success_flag", "pass_flag", "run_flag", "pass_epa", "run_epa"]
    ].rename(
        columns={team_col: "team"}
    )
    role_part = role_part.merge(play_core, on=["gameId", "playId", "team"], how="inner")
    if role_part.empty:
        return pd.DataFrame()

    # Aggregate slot-by-slot to avoid building one massive melted dataframe for multi-season selections.
    slot_aggs: list[pd.DataFrame] = []
    stat_cols = ["team", "epa", "success_flag", "pass_flag", "run_flag", "pass_epa", "run_epa"]
    for slot_col in player_cols:
        slot_df = role_part[stat_cols + [slot_col]].rename(columns={slot_col: "player_id"})
        slot_df = slot_df.dropna(subset=["player_id"])
        if slot_df.empty:
            continue
        slot_df["player_id"] = normalize_str(slot_df["player_id"].astype("string"))
        slot_df = slot_df[slot_df["player_id"].str.len() > 0]
        if slot_df.empty:
            continue
        slot_agg = (
            slot_df.groupby(["team", "player_id"], dropna=False)
            .agg(
                on_plays=("epa", "size"),
                on_epa_sum=("epa", "sum"),
                on_success_sum=("success_flag", "sum"),
                on_pass_sum=("pass_flag", "sum"),
                on_run_sum=("run_flag", "sum"),
                on_pass_epa_sum=("pass_epa", "sum"),
                on_run_epa_sum=("run_epa", "sum"),
            )
            .reset_index()
        )
        slot_aggs.append(slot_agg)

    if not slot_aggs:
        return pd.DataFrame()

    on_stats = (
        pd.concat(slot_aggs, ignore_index=True)
        .groupby(["team", "player_id"], dropna=False)
        .agg(
            on_plays=("on_plays", "sum"),
            on_epa_sum=("on_epa_sum", "sum"),
            on_success_sum=("on_success_sum", "sum"),
            on_pass_sum=("on_pass_sum", "sum"),
            on_run_sum=("on_run_sum", "sum"),
            on_pass_epa_sum=("on_pass_epa_sum", "sum"),
            on_run_epa_sum=("on_run_epa_sum", "sum"),
        )
        .reset_index()
    )

    out = on_stats.merge(totals, on="team", how="left")
    out["off_plays"] = out["total_plays"] - out["on_plays"]
    out = out[(out["on_plays"] > 0) & (out["off_plays"] > 0)]
    out = out[(out["on_plays"] >= min_plays) & (out["off_plays"] >= min_plays)]
    if out.empty:
        return pd.DataFrame()

    out["on_epa_play"] = out["on_epa_sum"] / out["on_plays"]
    out["off_epa_play"] = (out["total_epa_sum"] - out["on_epa_sum"]) / out["off_plays"]
    out["on_success_rate"] = out["on_success_sum"] / out["on_plays"]
    out["off_success_rate"] = (out["total_success_sum"] - out["on_success_sum"]) / out["off_plays"]
    out["on_pass_rate"] = out["on_pass_sum"] / out["on_plays"]
    out["off_pass_rate"] = (out["total_pass_sum"] - out["on_pass_sum"]) / out["off_plays"]
    out["on_run_rate"] = out["on_run_sum"] / out["on_plays"]
    out["off_run_rate"] = (out["total_run_sum"] - out["on_run_sum"]) / out["off_plays"]
    out["off_pass_plays"] = out["total_pass_sum"] - out["on_pass_sum"]
    out["off_run_plays"] = out["total_run_sum"] - out["on_run_sum"]
    out["on_pass_epa_play"] = safe_div(out["on_pass_epa_sum"], out["on_pass_sum"])
    out["off_pass_epa_play"] = safe_div(out["total_pass_epa_sum"] - out["on_pass_epa_sum"], out["off_pass_plays"])
    out["on_run_epa_play"] = safe_div(out["on_run_epa_sum"], out["on_run_sum"])
    out["off_run_epa_play"] = safe_div(out["total_run_epa_sum"] - out["on_run_epa_sum"], out["off_run_plays"])

    out["epa_delta"] = out["on_epa_play"] - out["off_epa_play"]
    out["pass_epa_delta"] = out["on_pass_epa_play"] - out["off_pass_epa_play"]
    out["run_epa_delta"] = out["on_run_epa_play"] - out["off_run_epa_play"]
    out["success_delta"] = out["on_success_rate"] - out["off_success_rate"]
    out["abs_epa_delta"] = out["epa_delta"].abs()

    player_lookup = build_player_lookup(rosters)
    out = out.merge(player_lookup, on="player_id", how="left")
    out["player_name"] = out["player_name"].fillna("Unknown")
    out["position"] = out["position"].fillna("")
    out["Player"] = out["player_name"] + " (" + out["player_id"] + ")"

    out = out.sort_values(["abs_epa_delta", "on_plays"], ascending=[False, False]).head(top_n).copy()
    return out[
        [
            "Player",
            "team",
            "position",
            "on_plays",
            "off_plays",
            "on_epa_play",
            "off_epa_play",
            "epa_delta",
            "pass_epa_delta",
            "run_epa_delta",
            "on_success_rate",
            "off_success_rate",
            "success_delta",
            "on_pass_rate",
            "off_pass_rate",
            "on_run_rate",
            "off_run_rate",
        ]
    ].rename(
        columns={
            "team": "Team",
            "position": "Pos",
            "on_plays": "On Plays",
            "off_plays": "Off Plays",
            "on_epa_play": "On EPA/play",
            "off_epa_play": "Off EPA/play",
            "epa_delta": "EPA Delta",
            "pass_epa_delta": "Pass EPA Delta",
            "run_epa_delta": "Rush EPA Delta",
            "on_success_rate": "On Success Rate",
            "off_success_rate": "Off Success Rate",
            "success_delta": "Success Delta",
            "on_pass_rate": "On Pass Rate",
            "off_pass_rate": "Off Pass Rate",
            "on_run_rate": "On Run Rate",
            "off_run_rate": "Off Run Rate",
        }
    )


def safe_div_scalar(num: float, den: float) -> float:
    if den == 0 or pd.isna(den):
        return np.nan
    return num / den


def game_status_split(
    team_plays: pd.DataFrame,
    part_wide: pd.DataFrame,
    team: str,
    split_player_id: str,
    min_snap_threshold: int,
) -> pd.DataFrame:
    games = team_plays[["gameId", "display_week"]].drop_duplicates()
    player_cols = [f"P{i}" for i in range(1, 16)]
    team_part = part_wide[(part_wide["team"] == team) & (part_wide["role"] == "offense")][["gameId", "playId"] + player_cols]
    if team_part.empty:
        games["split"] = f"<{min_snap_threshold} snaps"
        games["player_snaps"] = 0
        return games[["gameId", "split", "player_snaps"]]

    on_mask = team_part[player_cols].eq(split_player_id).any(axis=1)
    snap_counts = (
        team_part.loc[on_mask, ["gameId", "playId"]]
        .drop_duplicates()
        .groupby("gameId")
        .size()
        .reset_index(name="player_snaps")
    )
    games = games.merge(snap_counts, on="gameId", how="left")
    games["player_snaps"] = games["player_snaps"].fillna(0).astype(int)
    games["split"] = np.where(
        games["player_snaps"] < int(min_snap_threshold),
        f"<{min_snap_threshold} snaps",
        f">={min_snap_threshold} snaps",
    )
    return games[["gameId", "split", "player_snaps"]]


def team_active_inactive_table(team_plays: pd.DataFrame, games_split: pd.DataFrame) -> pd.DataFrame:
    df = team_plays.merge(games_split, on="gameId", how="inner")
    if df.empty:
        return pd.DataFrame()
    df["success_flag"] = build_success_flag(df).astype(int)
    out = (
        df.groupby("split", dropna=False)
        .agg(
            Plays=("epa", "size"),
            Games=("gameId", "nunique"),
            **{
                "EPA/play": ("epa", "mean"),
                "Success Rate": ("success_flag", "mean"),
                "Pass Rate": ("pass_play", "mean"),
                "Run Rate": ("run_play", "mean"),
            },
        )
        .reset_index()
        .rename(columns={"split": "Status"})
    )
    out["Plays/Game"] = out["Plays"] / out["Games"].replace(0, np.nan)
    return out[["Status", "Games", "Plays", "Plays/Game", "EPA/play", "Success Rate", "Pass Rate", "Run Rate"]]


def player_active_inactive_table(
    team_plays: pd.DataFrame,
    games_split: pd.DataFrame,
    player_id: str,
    position: str,
    reception_points: float,
) -> pd.DataFrame:
    df = team_plays.merge(games_split, on="gameId", how="inner")
    if df.empty:
        return pd.DataFrame()

    pos = (position or "").upper()
    df["target_id"] = df["target"]
    df["target_num"] = pd.to_numeric(df["target_id"], errors="coerce")
    # In this dataset, target is commonly a numeric flag (e.g., 1.0) rather than player ID.
    df["is_target_event"] = ((df["target_num"] == 1) | (df["target_id"] != "")).astype(int)
    df["receiver_id_norm"] = df["receiver_id"]
    df["passer_id_norm"] = df["passer_id"]
    df["runner_id_norm"] = df["runner_id"]
    # Player target: receiver-based match, with target-id fallback for datasets where target stores player IDs.
    df["player_target"] = (
        ((df["receiver_id_norm"] == player_id) | (df["target_id"] == player_id)) & (df["is_target_event"] == 1)
    ).astype(int)
    df["player_rec_yards_base"] = np.where(df["receiver_id_norm"] == player_id, df["rec_yards"].fillna(0), 0.0)
    df["player_no_rec_yards"] = np.where(
        df["no_reception_player_1_id"] == player_id,
        df["no_reception_yards_1"].fillna(0),
        0.0,
    ) + np.where(
        df["no_reception_player_2_id"] == player_id,
        df["no_reception_yards_2"].fillna(0),
        0.0,
    )
    df["player_rec_yards"] = df["player_rec_yards_base"] + df["player_no_rec_yards"]
    df["player_receptions"] = np.where((df["receiver_id_norm"] == player_id) & (df["reception"] == 1), 1, 0)
    df["player_rec_td_base"] = np.where((df["receiver_id_norm"] == player_id) & (df["passing_touchdown"] == 1), 1, 0)
    df["player_no_rec_td"] = np.where(
        df["no_reception_player_1_id"] == player_id,
        df["no_reception_touchdown_1"].fillna(0),
        0,
    ) + np.where(
        df["no_reception_player_2_id"] == player_id,
        df["no_reception_touchdown_2"].fillna(0),
        0,
    )
    df["player_rec_td"] = df["player_rec_td_base"] + df["player_no_rec_td"]
    df["player_rush_att"] = np.where((df["runner_id_norm"] == player_id) & (df["rush_attempt"] == 1), 1, 0)
    df["player_designed_rush_att"] = np.where(
        (df["runner_id_norm"] == player_id) & (df["rush_attempt"] == 1) & (df["scramble"] != 1),
        1,
        0,
    )
    df["player_rush_yards_base"] = np.where(df["runner_id_norm"] == player_id, df["rushing_yards"].fillna(0), 0.0)
    df["player_no_rush_yards"] = np.where(
        df["no_rush_player_id"] == player_id,
        df["no_rush_yards"].fillna(0),
        0.0,
    )
    df["player_rush_yards"] = df["player_rush_yards_base"] + df["player_no_rush_yards"]
    df["player_rush_td_base"] = np.where((df["runner_id_norm"] == player_id) & (df["rushing_touchdown"] == 1), 1, 0)
    df["player_no_rush_td"] = np.where(
        df["no_rush_player_id"] == player_id,
        df["no_rush_touchdown"].fillna(0),
        0,
    )
    df["player_rush_td"] = df["player_rush_td_base"] + df["player_no_rush_td"]
    df["player_pass_att"] = np.where((df["passer_id_norm"] == player_id) & (df["attempt"] == 1), 1, 0)
    df["player_pass_yards"] = np.where(df["passer_id_norm"] == player_id, df["passing_yards"].fillna(0), 0.0)
    df["player_pass_td"] = np.where((df["passer_id_norm"] == player_id) & (df["passing_touchdown"] == 1), 1, 0)
    df["player_int"] = np.where((df["passer_id_norm"] == player_id) & (df["intercepted"] == 1), 1, 0)
    df["player_fumble_lost"] = np.where(
        (df["fumble_lost_player_1_id"] == player_id) | (df["fumble_lost_player_2_id"] == player_id),
        1,
        0,
    )
    df["player_dropback"] = np.where((df["passer_id_norm"] == player_id) & (df["dropback"] == 1), 1, 0)
    df["player_sack"] = np.where((df["passer_id_norm"] == player_id) & (df["sack"] == 1), 1, 0)
    df["player_scramble"] = np.where((df["passer_id_norm"] == player_id) & (df["scramble"] == 1), 1, 0)
    df["team_targets"] = df["is_target_event"]
    df["team_rec_yards"] = (
        df["rec_yards"].fillna(0)
        + df["no_reception_yards_1"].fillna(0)
        + df["no_reception_yards_2"].fillna(0)
    )
    df["team_rush_att"] = (df["rush_attempt"] == 1).astype(int)
    df["team_designed_rush_att"] = ((df["rush_attempt"] == 1) & (df["scramble"] != 1)).astype(int)
    df["team_rush_td"] = (df["rushing_touchdown"] == 1).astype(int)
    df["team_rec_td"] = (df["passing_touchdown"] == 1).astype(int)

    rows = []
    for split_val, g in df.groupby("split", dropna=False):
        games = int(g["gameId"].nunique())
        targets = float(g["player_target"].sum())
        rec_yards = float(g["player_rec_yards"].sum())
        receptions = float(g["player_receptions"].sum())
        rec_tds = float(g["player_rec_td"].sum())
        rush_att = float(g["player_rush_att"].sum())
        designed_rush_att = float(g["player_designed_rush_att"].sum())
        rush_yards = float(g["player_rush_yards"].sum())
        rush_tds = float(g["player_rush_td"].sum())
        pass_att = float(g["player_pass_att"].sum())
        pass_yards = float(g["player_pass_yards"].sum())
        pass_tds = float(g["player_pass_td"].sum())
        ints = float(g["player_int"].sum())
        fumbles_lost = float(g["player_fumble_lost"].sum())
        dropbacks = float(g["player_dropback"].sum())
        sacks = float(g["player_sack"].sum())
        scrambles = float(g["player_scramble"].sum())
        team_targets = float(g["team_targets"].sum())
        team_rec_yards = float(g["team_rec_yards"].sum())
        team_rush_att = float(g["team_rush_att"].sum())
        team_designed_rush_att = float(g["team_designed_rush_att"].sum())
        team_rush_td = float(g["team_rush_td"].sum())
        team_rec_td = float(g["team_rec_td"].sum())

        base = {"Status": split_val, "Games": games}
        fantasy_pts = (
            pass_yards * 0.04
            + pass_tds * 4.0
            + rec_yards * 0.1
            + rush_yards * 0.1
            + (rec_tds + rush_tds) * 6.0
            + receptions * float(reception_points)
            - ints
            - fumbles_lost
        )
        base["Fantasy Pts/Game"] = safe_div_scalar(fantasy_pts, games)
        if pos.startswith("QB"):
            base.update(
                {
                    "Passing Yards/Game": safe_div_scalar(pass_yards, games),
                    "Sack Rate": safe_div_scalar(sacks, dropbacks),
                    "Scramble Rate": safe_div_scalar(scrambles, dropbacks),
                    "Passing Yards/Target": safe_div_scalar(pass_yards, pass_att),
                    "Passing TD/Target": safe_div_scalar(pass_tds, pass_att),
                    "Rushing Yards/Game": safe_div_scalar(rush_yards, games),
                    "Rushing Yards/Attempt": safe_div_scalar(rush_yards, rush_att),
                }
            )
        elif pos.startswith("RB"):
            base.update(
                {
                    "Avg Target Share": safe_div_scalar(targets, team_targets),
                    "Targets/Game": safe_div_scalar(targets, games),
                    "Yards/Target": safe_div_scalar(rec_yards, targets),
                    "Receiving Yardage Share": safe_div_scalar(rec_yards, team_rec_yards),
                    "Rushing Carries/Game": safe_div_scalar(rush_att, games),
                    "Rushing Carry Share": safe_div_scalar(designed_rush_att, team_designed_rush_att),
                    "Yards/Carry": safe_div_scalar(rush_yards, rush_att),
                    "Rushing TD/Carry": safe_div_scalar(rush_tds, rush_att),
                    "Rushing TD Share": safe_div_scalar(rush_tds, team_rush_td),
                }
            )
        else:
            base.update(
                {
                    "Avg Target Share": safe_div_scalar(targets, team_targets),
                    "Targets/Game": safe_div_scalar(targets, games),
                    "Yards/Target": safe_div_scalar(rec_yards, targets),
                    "Receiving Yardage Share": safe_div_scalar(rec_yards, team_rec_yards),
                    "Receiving TD/Game": safe_div_scalar(rec_tds, games),
                    "Receiving TD Share": safe_div_scalar(rec_tds, team_rec_td),
                }
            )
        rows.append(base)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    ordered_cols = ["Status", "Fantasy Pts/Game"] + [c for c in out.columns if c not in ["Status", "Fantasy Pts/Game"]]
    return out[ordered_cols]


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def format_report_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in ["EPA/play", "On EPA/play", "Off EPA/play"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    for col in ["Success Rate", "Pass Rate", "Run Rate"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    for col in ["Plays", "On Plays", "Off Plays", "display_week"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{int(x)}")
    return out


def render_table_image(df: pd.DataFrame, title: str, out_path: str, max_rows: int = 35) -> None:
    shown = format_report_df(df).head(max_rows).astype(str)
    n_rows = max(1, len(shown))
    fig_h = min(14.0, 1.2 + 0.38 * (n_rows + 1))
    fig_w = 11.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=10)
    tbl = ax.table(
        cellText=shown.values,
        colLabels=[str(c) for c in shown.columns],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_trend_chart_image(df: pd.DataFrame, title: str, out_path: str) -> bool:
    if df.empty:
        return False
    cols = [c for c in ["On EPA/play", "Off EPA/play"] if c in df.columns]
    if not cols:
        return False
    plot_df = df.copy()
    plot_df["display_week"] = pd.to_numeric(plot_df["display_week"], errors="coerce")
    plot_df = plot_df.dropna(subset=["display_week"])
    if plot_df.empty:
        return False

    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    for col in cols:
        ax.plot(plot_df["display_week"], pd.to_numeric(plot_df[col], errors="coerce"), marker="o", linewidth=1.6, label=col)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Week")
    ax.set_ylabel("EPA/play")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def build_pdf_bytes(
    selection_text: str,
    tables: dict[str, pd.DataFrame],
    trend_charts: dict[str, pd.DataFrame],
) -> bytes | None:
    if FPDF is None:
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, "NFL On/Off Splits Report", ln=1)
    pdf.set_font("Arial", size=9)
    usable_w = max(20, pdf.w - pdf.l_margin - pdf.r_margin)

    for line in selection_text.split("\n"):
        pdf.set_x(pdf.l_margin)
        if line:
            pdf.multi_cell(usable_w, 5, str(line))
        else:
            pdf.cell(0, 5, "", ln=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        img_idx = 0
        for title, df in tables.items():
            pdf.ln(2)
            pdf.set_font("Arial", style="B", size=10)
            pdf.cell(0, 6, title, ln=1)
            pdf.set_font("Arial", size=8)

            if df.empty:
                pdf.cell(0, 5, "No rows", ln=1)
                continue

            img_idx += 1
            path = os.path.join(tmpdir, f"table_{img_idx}.png")
            render_table_image(df, title, path)
            pdf.image(path, x=pdf.l_margin, w=usable_w)

        for title, trend_df in trend_charts.items():
            pdf.ln(2)
            pdf.set_font("Arial", style="B", size=10)
            pdf.cell(0, 6, title, ln=1)
            path = os.path.join(tmpdir, f"trend_{img_idx}.png")
            img_idx += 1
            ok = render_trend_chart_image(trend_df, title, path)
            if not ok:
                pdf.set_font("Arial", size=8)
                pdf.cell(0, 5, "No chart rows", ln=1)
            else:
                pdf.image(path, x=pdf.l_margin, w=usable_w)

    raw = pdf.output(dest="S")
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, str):
        return raw.encode("latin-1", errors="ignore")
    return raw


plays, part_wide, rosters = load_data()

st.title("NFL Player On/Off Splits")
st.caption("Weeks 1-18 are regular season and 19-22 are playoffs. Excludes no-play, spikes, kneels, 2PT attempts, and special teams plays.")

all_seasons = sorted([int(x) for x in plays["season"].dropna().unique()])
selected_season = st.sidebar.selectbox("Season", all_seasons, index=len(all_seasons) - 1 if all_seasons else 0)

base_filtered = plays
part_filtered = part_wide
roster_filtered = rosters

base_filtered = base_filtered[base_filtered["season"] == selected_season]
part_filtered = part_filtered[part_filtered["season"] == selected_season]
roster_filtered = roster_filtered[roster_filtered["season"] == selected_season]

week_options = sorted([int(x) for x in base_filtered["display_week"].dropna().unique()])
selected_weeks = st.sidebar.multiselect("Weeks", week_options, default=week_options)

if selected_weeks:
    base_filtered = base_filtered[base_filtered["display_week"].isin(selected_weeks)]
    part_filtered = part_filtered[part_filtered["display_week"].isin(selected_weeks)]
    roster_filtered = roster_filtered[roster_filtered["display_week"].isin(selected_weeks)]

# Leaderboards intentionally use only season/week filters.
leaderboard_plays = base_filtered.copy()

team_options = sorted([str(x) for x in roster_filtered["team"].dropna().unique() if str(x)])
if not team_options:
    team_options = sorted(set(base_filtered["offense"].astype(str).unique()) | set(base_filtered["defense"].astype(str).unique()))

if not team_options:
    st.warning("No team options available for selected filters.")
    st.stop()

team = st.sidebar.selectbox("Team", team_options)

player_df = roster_filtered[roster_filtered["team"] == team].copy()
if player_df.empty:
    st.warning("No players found for selected team and filters.")
    st.stop()

preferred_first = normalize_str(player_df["FootballName"])
fallback_first = normalize_str(player_df["FirstName"])
last_name = normalize_str(player_df["LastName"])
first_name = preferred_first.where(preferred_first.str.len() > 0, fallback_first)
name = (first_name + " " + last_name).str.strip()
name = name.where(name.str.len() > 0, first_name)

player_options = (
    pd.DataFrame(
        {
            "player_id": normalize_str(player_df["player_id"]),
            "name": name,
            "position": normalize_position(player_df["position"].astype("string")),
        }
    )
    .dropna(subset=["player_id"])
)
player_options = player_options[player_options["player_id"].str.len() > 0]
player_options = player_options.drop_duplicates(subset=["player_id"])
player_options["label"] = (
    player_options["name"].fillna("Unknown")
    + " ("
    + player_options["player_id"]
    + ")"
    + np.where(player_options["position"].str.len() > 0, " - " + player_options["position"], "")
)
player_options = player_options.sort_values(["name", "player_id"]).reset_index(drop=True)

selected_labels = st.sidebar.multiselect("Players", player_options["label"].tolist(), default=player_options["label"].tolist()[:1])
if not selected_labels:
    st.warning("Select at least one player.")
    st.stop()

selected_player_ids = (
    player_options[player_options["label"].isin(selected_labels)]["player_id"].drop_duplicates().tolist()
)
on_mode = st.sidebar.radio("On definition", ["Any selected player on field", "All selected players on field"], index=0)

team_scope = base_filtered[(base_filtered["offense"] == team) | (base_filtered["defense"] == team)]
if team_scope.empty:
    st.warning("No plays for selected team and season/week filters.")
    st.stop()

quarter_options = sorted([int(x) for x in team_scope["quarter"].dropna().unique()])
selected_quarters = st.sidebar.multiselect("Quarter", quarter_options, default=quarter_options)

down_options = sorted([int(x) for x in team_scope["down"].dropna().unique() if x in [1, 2, 3, 4]])
selected_downs = st.sidebar.multiselect("Down", down_options, default=down_options)

distance_vals = team_scope["distance"].dropna()
if distance_vals.empty:
    distance_range = (0, 99)
else:
    dist_min = int(max(0, np.floor(distance_vals.min())))
    dist_max = int(min(99, np.ceil(distance_vals.max())))
    distance_range = st.sidebar.slider("Distance to go", min_value=dist_min, max_value=dist_max, value=(dist_min, dist_max))

red_zone_filter = st.sidebar.selectbox("Red Zone", ["All", "Red Zone (<=20)", "Outside Red Zone (>20)"])
inside_10_filter = st.sidebar.selectbox("Inside 10", ["All", "Inside 10", "Outside 10"])

team_margin_series = np.where(
    team_scope["offense"] == team,
    team_scope["offense_score"].fillna(0) - team_scope["defense_score"].fillna(0),
    team_scope["defense_score"].fillna(0) - team_scope["offense_score"].fillna(0),
)
margin_min = int(np.floor(np.nanmin(team_margin_series)))
margin_max = int(np.ceil(np.nanmax(team_margin_series)))
score_margin_range = st.sidebar.slider("Score margin (team perspective)", min_value=margin_min, max_value=margin_max, value=(margin_min, margin_max))

min_play_threshold = st.sidebar.number_input("Minimum plays (On and Off)", min_value=0, value=0, step=10)
leaderboard_min_plays = st.sidebar.number_input("Leaderboard min plays (On and Off)", min_value=1, value=100, step=10)
leaderboard_team_options = sorted(
    set(leaderboard_plays["offense"].dropna().astype(str).unique()) | set(leaderboard_plays["defense"].dropna().astype(str).unique())
)
leaderboard_team_choice = st.sidebar.selectbox("Leaderboard team", ["All teams"] + leaderboard_team_options)
leaderboard_team_filter = None if leaderboard_team_choice == "All teams" else leaderboard_team_choice

base_filtered = apply_common_filters(
    base_filtered,
    selected_downs=selected_downs,
    distance_range=distance_range,
    selected_quarters=selected_quarters,
    red_zone_filter=red_zone_filter,
    inside_10_filter=inside_10_filter,
)
selection_payload = {
    "team": team,
    "players": selected_labels,
    "on_definition": on_mode,
    "season": selected_season,
    "weeks": selected_weeks,
    "down": selected_downs,
    "distance": distance_range,
    "quarters": selected_quarters,
    "red_zone": red_zone_filter,
    "inside_10": inside_10_filter,
    "score_margin": score_margin_range,
}

offense_table, offense_personnel = split_for_role(
    plays=base_filtered,
    part_wide=part_filtered,
    team=team,
    selected_player_ids=selected_player_ids,
    role="offense",
    on_mode=on_mode,
    margin_range=score_margin_range,
)
defense_table, defense_personnel = split_for_role(
    plays=base_filtered,
    part_wide=part_filtered,
    team=team,
    selected_player_ids=selected_player_ids,
    role="defense",
    on_mode=on_mode,
    margin_range=score_margin_range,
)

if min_play_threshold > 0:
    if not offense_table.empty and (offense_table[offense_table["Split"].isin(["On", "Off"])]["Plays"] < min_play_threshold).any():
        offense_table = pd.DataFrame()
        offense_personnel = pd.DataFrame()
    if not defense_table.empty and (defense_table[defense_table["Split"].isin(["On", "Off"])]["Plays"] < min_play_threshold).any():
        defense_table = pd.DataFrame()
        defense_personnel = pd.DataFrame()

offense_baseline = baseline_table(base_filtered, team=team, role="offense", margin_range=score_margin_range)
defense_baseline = baseline_table(base_filtered, team=team, role="defense", margin_range=score_margin_range)

offense_trend = trend_table(base_filtered, part_filtered, team, selected_player_ids, "offense", on_mode, score_margin_range)
defense_trend = trend_table(base_filtered, part_filtered, team, selected_player_ids, "defense", on_mode, score_margin_range)
top_offense = top_player_diffs(
    plays=leaderboard_plays,
    part_wide=part_filtered,
    rosters=roster_filtered,
    role="offense",
    min_plays=int(leaderboard_min_plays),
    leaderboard_team=leaderboard_team_filter,
    top_n=50,
)
top_defense = top_player_diffs(
    plays=leaderboard_plays,
    part_wide=part_filtered,
    rosters=roster_filtered,
    role="defense",
    min_plays=int(leaderboard_min_plays),
    leaderboard_team=leaderboard_team_filter,
    top_n=50,
)

formatters = {
    "EPA/play": "{:.3f}",
    "Plays/Game": "{:.1f}",
    "Success Rate": "{:.1%}",
    "Pass Rate": "{:.1%}",
    "Run Rate": "{:.1%}",
    "On EPA/play": "{:.3f}",
    "Off EPA/play": "{:.3f}",
    "EPA Delta": "{:.3f}",
    "Pass EPA Delta": "{:.3f}",
    "Rush EPA Delta": "{:.3f}",
    "On Success Rate": "{:.1%}",
    "Off Success Rate": "{:.1%}",
    "Success Delta": "{:.1%}",
    "On Pass Rate": "{:.1%}",
    "Off Pass Rate": "{:.1%}",
    "On Run Rate": "{:.1%}",
    "Off Run Rate": "{:.1%}",
    "Passing Yards/Game": "{:.1f}",
    "Sack Rate": "{:.1%}",
    "Scramble Rate": "{:.1%}",
    "Passing Yards/Target": "{:.2f}",
    "Passing TD/Target": "{:.3f}",
    "Rushing Yards/Game": "{:.1f}",
    "Rushing Yards/Attempt": "{:.2f}",
    "Avg Target Share": "{:.1%}",
    "Targets/Game": "{:.2f}",
    "Yards/Target": "{:.2f}",
    "Receiving Yardage Share": "{:.1%}",
    "Rushing Carries/Game": "{:.2f}",
    "Rushing Carry Share": "{:.1%}",
    "Yards/Carry": "{:.2f}",
    "Rushing TD/Carry": "{:.3f}",
    "Rushing TD Share": "{:.1%}",
    "Receiving TD/Game": "{:.2f}",
    "Receiving TD Share": "{:.1%}",
    "Fantasy Pts/Game": "{:.2f}",
}

tab_onoff, tab_status = st.tabs(["On/Off View", "Snap-Threshold Splits"])

with tab_onoff:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Offense On/Off")
        if offense_table.empty:
            st.info("No offense rows for this selection.")
        else:
            st.dataframe(offense_table.style.format(formatters), width="stretch")

        st.markdown("### Offense Baseline (Team vs League)")
        st.dataframe(offense_baseline.style.format(formatters), width="stretch")

    with col2:
        st.markdown("### Defense On/Off")
        if defense_table.empty:
            st.info("No defense rows for this selection.")
        else:
            st.dataframe(defense_table.style.format(formatters), width="stretch")

        st.markdown("### Defense Baseline (Team vs League)")
        st.dataframe(defense_baseline.style.format(formatters), width="stretch")

    st.markdown("### Offense Trend (EPA/play)")
    if offense_trend.empty:
        st.info("No offense trend data.")
    else:
        offense_plot = offense_trend.set_index("display_week")
        offense_cols = [c for c in ["On EPA/play", "Off EPA/play"] if c in offense_plot.columns]
        if offense_cols:
            st.line_chart(offense_plot[offense_cols])
        else:
            st.info("No offense On/Off EPA series available for current filters.")
        st.dataframe(offense_trend, width="stretch")

    st.markdown("### Defense Trend (EPA/play)")
    if defense_trend.empty:
        st.info("No defense trend data.")
    else:
        defense_plot = defense_trend.set_index("display_week")
        defense_cols = [c for c in ["On EPA/play", "Off EPA/play"] if c in defense_plot.columns]
        if defense_cols:
            st.line_chart(defense_plot[defense_cols])
        else:
            st.info("No defense On/Off EPA series available for current filters.")
        st.dataframe(defense_trend, width="stretch")

    st.markdown("### Offense Personnel Splits")
    if offense_personnel.empty:
        st.info("No offense personnel rows.")
    else:
        st.dataframe(offense_personnel.style.format(formatters), width="stretch")

    st.markdown("### Defense Personnel Splits (Opponent Offensive Personnel)")
    if defense_personnel.empty:
        st.info("No defense personnel rows.")
    else:
        st.dataframe(defense_personnel.style.format(formatters), width="stretch")

    st.markdown("### Top 50 Offensive Players By Absolute EPA Delta")
    if top_offense.empty:
        st.info("No offensive leaderboard rows for current filters.")
    else:
        st.dataframe(top_offense.style.format(formatters), width="stretch")

    st.markdown("### Top 50 Defensive Players By Absolute EPA Delta")
    if top_defense.empty:
        st.info("No defensive leaderboard rows for current filters.")
    else:
        st.dataframe(top_defense.style.format(formatters), width="stretch")

    st.markdown("### Export")
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button("Download Offense On/Off CSV", df_to_csv_bytes(offense_table), file_name="offense_on_off.csv", mime="text/csv")
        st.download_button("Download Defense On/Off CSV", df_to_csv_bytes(defense_table), file_name="defense_on_off.csv", mime="text/csv")
        st.download_button("Download Offense Personnel CSV", df_to_csv_bytes(offense_personnel), file_name="offense_personnel_splits.csv", mime="text/csv")
        st.download_button("Download Defense Personnel CSV", df_to_csv_bytes(defense_personnel), file_name="defense_personnel_splits.csv", mime="text/csv")

    with export_col2:
        selection_text = (
            f"Team: {selection_payload['team']}\n"
            f"Players: {', '.join(selection_payload['players'])}\n"
            f"On definition: {selection_payload['on_definition']}\n"
            f"Season: {selection_payload['season']}\n"
            f"Weeks: {selection_payload['weeks']}\n"
            f"Down: {selection_payload['down']}\n"
            f"Distance: {selection_payload['distance']}\n"
            f"Quarter: {selection_payload['quarters']}\n"
            f"Red Zone: {selection_payload['red_zone']}\n"
            f"Inside 10: {selection_payload['inside_10']}\n"
            f"Score margin: {selection_payload['score_margin']}"
        )
        if st.button("Generate PDF Report"):
            pdf_bytes = build_pdf_bytes(
                selection_text,
                {
                    "Offense On/Off": offense_table,
                    "Defense On/Off": defense_table,
                    "Offense Baseline (Team vs League)": offense_baseline,
                    "Defense Baseline (Team vs League)": defense_baseline,
                    "Offense Trend Table": offense_trend,
                    "Defense Trend Table": defense_trend,
                    "Offense Personnel Splits": offense_personnel,
                    "Defense Personnel Splits": defense_personnel,
                },
                {
                    "Offense Trend Chart": offense_trend,
                    "Defense Trend Chart": defense_trend,
                },
            )
            if pdf_bytes is None:
                st.info("Install fpdf2 to enable PDF export.")
            else:
                st.session_state["pdf_bytes"] = pdf_bytes

        if "pdf_bytes" in st.session_state and st.session_state["pdf_bytes"] is not None:
            st.download_button(
                "Download PDF Report",
                data=st.session_state["pdf_bytes"],
                file_name="on_off_report.pdf",
                mime="application/pdf",
            )

    with st.expander("Selection", expanded=False):
        st.write(selection_payload)

with tab_status:
    st.markdown("### Snap-Threshold Team + Player Splits")
    status_team_options = sorted([str(x) for x in roster_filtered["team"].dropna().astype(str).unique() if str(x)])
    if not status_team_options:
        st.info("No teams available for current season/week filters.")
    else:
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        with col_a:
            status_team = st.selectbox("Status Split Team", status_team_options, key="status_team")

        status_roster = roster_filtered[roster_filtered["team"] == status_team].copy()
        pref_first = normalize_str(status_roster["FootballName"])
        fb_first = normalize_str(status_roster["FirstName"])
        ln = normalize_str(status_roster["LastName"])
        fn = pref_first.where(pref_first.str.len() > 0, fb_first)
        fulln = (fn + " " + ln).str.strip().where((fn + " " + ln).str.strip().str.len() > 0, fn)
        status_roster["display_name"] = fulln
        status_roster["label"] = (
            status_roster["display_name"].fillna("Unknown")
            + " ("
            + status_roster["player_id"].astype(str)
            + ") - "
            + normalize_position(status_roster["position"]).astype(str)
        )
        status_roster = status_roster.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"]).reset_index(drop=True)

        if status_roster.empty:
            st.info("No roster rows available for this team and filter.")
        else:
            with col_b:
                split_player_label = st.selectbox(
                    "Player For Status Split",
                    status_roster["label"].tolist(),
                    key="status_split_player",
                )
            split_player_row = status_roster[status_roster["label"] == split_player_label].iloc[0]
            split_player_id = str(split_player_row["player_id"])

            with col_c:
                second_player_label = st.selectbox(
                    "Second Player For Player Stats",
                    status_roster["label"].tolist(),
                    key="status_second_player",
                )
            with col_d:
                min_snap_threshold = st.number_input(
                    "Low-snap threshold (< snaps)",
                    min_value=0,
                    value=1,
                    step=1,
                    key="status_snap_threshold",
                )
            with col_e:
                scoring_choice = st.selectbox(
                    "Fantasy Scoring",
                    ["PPR (1.0)", "Half PPR (0.5)"],
                    key="status_scoring",
                )
                reception_points = 1.0 if scoring_choice.startswith("PPR") else 0.5
            second_player_row = status_roster[status_roster["label"] == second_player_label].iloc[0]
            second_player_id = str(second_player_row["player_id"])
            second_player_pos = str(normalize_position(pd.Series([second_player_row["position"]])).iloc[0])

            status_team_plays = leaderboard_plays[leaderboard_plays["offense"] == status_team].copy()
            if status_team_plays.empty:
                st.info("No team offensive plays for selected season/week.")
            else:
                games_split = game_status_split(
                    status_team_plays,
                    part_filtered,
                    status_team,
                    split_player_id,
                    int(min_snap_threshold),
                )
                team_status_table = team_active_inactive_table(status_team_plays, games_split)
                player_status_table = player_active_inactive_table(
                    status_team_plays,
                    games_split,
                    second_player_id,
                    second_player_pos,
                    reception_points,
                )

                st.markdown("#### Team Results (Split By Selected Player Snaps)")
                if team_status_table.empty:
                    st.info("No team status split rows.")
                else:
                    st.dataframe(team_status_table.style.format(formatters), width="stretch")

                st.markdown("#### Player Results (Second Player)")
                st.caption(f"Second player position profile: {second_player_pos}")
                if player_status_table.empty:
                    st.info("No player status split rows.")
                else:
                    st.dataframe(player_status_table.style.format(formatters), width="stretch")

