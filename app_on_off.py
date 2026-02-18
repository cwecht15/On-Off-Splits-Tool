import io
import os
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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


def normalize_team(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def normalize_str(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def map_display_week(week: pd.Series, season_type: pd.Series) -> pd.Series:
    return week + np.where(season_type == "POST", 18, 0)


def _data_url_key(filename: str) -> str:
    return f"DATA_URL_{filename.replace('.', '_').upper()}"


def get_data_url_map() -> dict[str, str]:
    url_map: dict[str, str] = {}

    try:
        if "data_urls" in st.secrets:
            for k, v in dict(st.secrets["data_urls"]).items():
                if isinstance(k, str) and isinstance(v, str) and v.strip():
                    url_map[k.strip()] = v.strip()
    except Exception:
        pass

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

    num_cols = [
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
        "passing_yards",
        "rushing_yards",
    ]
    for col in num_cols:
        pbp[col] = pd.to_numeric(pbp[col], errors="coerce")

    pbp["seasonType"] = normalize_str(pbp["seasonType"]).str.upper()
    pbp["offense"] = normalize_team(pbp["offense"])
    pbp["defense"] = normalize_team(pbp["defense"])
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
        }
    )
    rosters["season"] = pd.to_numeric(rosters["season"], errors="coerce").astype("Int64")
    rosters["week"] = pd.to_numeric(rosters["week"], errors="coerce").astype("Int64")
    rosters["seasonType"] = normalize_str(rosters["seasonType"]).str.upper()
    rosters["team"] = normalize_team(rosters["team"])
    rosters["player_id"] = normalize_str(rosters["player_id"])
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
selected_seasons = st.sidebar.multiselect("Seasons", all_seasons, default=all_seasons)

base_filtered = plays
part_filtered = part_wide
roster_filtered = rosters

if selected_seasons:
    base_filtered = base_filtered[base_filtered["season"].isin(selected_seasons)]
    part_filtered = part_filtered[part_filtered["season"].isin(selected_seasons)]
    roster_filtered = roster_filtered[roster_filtered["season"].isin(selected_seasons)]

week_options = sorted([int(x) for x in base_filtered["display_week"].dropna().unique()])
selected_weeks = st.sidebar.multiselect("Weeks", week_options, default=week_options)

if selected_weeks:
    base_filtered = base_filtered[base_filtered["display_week"].isin(selected_weeks)]
    part_filtered = part_filtered[part_filtered["display_week"].isin(selected_weeks)]
    roster_filtered = roster_filtered[roster_filtered["display_week"].isin(selected_weeks)]

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
            "position": normalize_str(player_df["position"].astype("string")),
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

selected_labels = st.sidebar.multiselect(
    "Players",
    player_options["label"].tolist(),
    default=player_options["label"].tolist()[:1],
    key=f"players_{team}",
)
if not selected_labels:
    st.warning("Select at least one player.")
    st.stop()

selected_player_ids = (
    player_options[player_options["label"].isin(selected_labels)]["player_id"].drop_duplicates().tolist()
)

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
on_mode = st.sidebar.radio("On definition", ["Any selected player on field", "All selected players on field"], index=0)

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
    "seasons": selected_seasons,
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

formatters = {
    "EPA/play": "{:.3f}",
    "Success Rate": "{:.1%}",
    "Pass Rate": "{:.1%}",
    "Run Rate": "{:.1%}",
}

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
        f"Seasons: {selection_payload['seasons']}\n"
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

