import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_app_like_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    player_cols = [f"P{i}" for i in range(1, 16)]

    part = pd.read_csv(
        ROOT / "participation.csv",
        usecols=["gameId", "season", "seasonType", "week", "playId", "team", "role"] + player_cols,
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
    )

    pbp = pd.read_csv(
        ROOT / "play_by_play_data.csv",
        usecols=[
            "season",
            "seas_type",
            "week",
            "game_id",
            "play_id",
            "offense",
            "defense",
            "down",
            "distance",
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
        ],
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
    ).rename(columns={"game_id": "gameId", "play_id": "playId", "seas_type": "seasonType"})

    epa = pd.read_csv(
        ROOT / "epa.csv",
        usecols=["gameId", "playId", "epa"],
        dtype={"gameId": "string", "playId": "string"},
        low_memory=False,
    )

    pp = pd.read_csv(ROOT / "pp_data.csv", usecols=["play id", "personnel"], low_memory=False)
    pp = pp.rename(columns={"play id": "playId"})
    pp["playId"] = pd.to_numeric(pp["playId"], errors="coerce").astype("Int64").astype("string")
    pp["personnel"] = pd.to_numeric(pp["personnel"], errors="coerce").astype("Int64")
    pp = pp.dropna(subset=["playId"]).drop_duplicates(subset=["playId"], keep="first")

    for col in [
        "down",
        "distance",
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
    ]:
        pbp[col] = pd.to_numeric(pbp[col], errors="coerce")

    pbp["seasonType"] = pbp["seasonType"].fillna("").astype(str).str.strip().str.upper()
    pbp["offense"] = pbp["offense"].fillna("").astype(str).str.strip().str.upper()
    pbp["defense"] = pbp["defense"].fillna("").astype(str).str.strip().str.upper()
    pbp["yards_gained"] = pbp["passing_yards"].fillna(0) + pbp["rushing_yards"].fillna(0)
    pbp["display_week"] = pbp["week"] + (pbp["seasonType"] == "POST").astype(int) * 18

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
    plays = plays.merge(pp[["playId", "personnel"]], on="playId", how="left")

    part_long = (
        part.melt(
            id_vars=["gameId", "playId", "season", "seasonType", "week", "team", "role"],
            value_vars=player_cols,
            var_name="slot",
            value_name="player_id",
        )
        .drop(columns=["slot"])
        .dropna(subset=["player_id"])
    )
    part_long["player_id"] = part_long["player_id"].astype(str).str.strip()
    part_long = part_long[part_long["player_id"].str.len() > 0].copy()
    part_long["seasonType"] = part_long["seasonType"].fillna("").astype(str).str.strip().str.upper()
    part_long["team"] = part_long["team"].fillna("").astype(str).str.strip().str.upper()
    part_long["role"] = part_long["role"].fillna("").astype(str).str.strip().str.lower()
    part_long["display_week"] = part_long["week"] + (part_long["seasonType"] == "POST").astype(int) * 18

    return plays, part_long


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
        return {"Plays": 0, "EPA/play": float("nan"), "Success Rate": float("nan"), "Pass Rate": float("nan"), "Run Rate": float("nan")}
    success = build_success_flag(df)
    return {
        "Plays": int(len(df)),
        "EPA/play": float(df["epa"].mean()),
        "Success Rate": float(success.mean()),
        "Pass Rate": float((df["pass_play"] == 1).mean()),
        "Run Rate": float((df["run_play"] == 1).mean()),
    }


class TestGoldenRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        plays, part_long = load_app_like_frames()

        team = "ARZ"
        player_id = "00-0033553"
        role = "offense"
        season = 2021
        weeks = list(range(1, 19))

        plays = plays[(plays["season"] == season) & (plays["display_week"].isin(weeks))]
        part_long = part_long[(part_long["season"] == season) & (part_long["display_week"].isin(weeks))]

        team_plays = plays[plays["offense"] == team].copy()
        on_keys = (
            part_long[
                (part_long["team"] == team)
                & (part_long["role"] == role)
                & (part_long["player_id"] == player_id)
            ][["gameId", "playId"]]
            .drop_duplicates()
            .assign(is_on=1)
        )
        cls.team_plays = team_plays.merge(on_keys, on=["gameId", "playId"], how="left")
        cls.team_plays["is_on"] = cls.team_plays["is_on"].fillna(0).astype(int)

    def test_golden_on_off_split_metrics(self) -> None:
        on_summary = summarize(self.team_plays[self.team_plays["is_on"] == 1])
        off_summary = summarize(self.team_plays[self.team_plays["is_on"] == 0])

        self.assertEqual(on_summary["Plays"], 570)
        self.assertAlmostEqual(on_summary["EPA/play"], 0.05080732103157894, places=12)
        self.assertAlmostEqual(on_summary["Success Rate"], 0.5140350877192983, places=12)
        self.assertAlmostEqual(on_summary["Pass Rate"], 0.5333333333333333, places=12)
        self.assertAlmostEqual(on_summary["Run Rate"], 0.4666666666666667, places=12)

        self.assertEqual(off_summary["Plays"], 539)
        self.assertAlmostEqual(off_summary["EPA/play"], 0.1023514501038961, places=12)
        self.assertAlmostEqual(off_summary["Success Rate"], 0.5064935064935064, places=12)
        self.assertAlmostEqual(off_summary["Pass Rate"], 0.6641929499072357, places=12)
        self.assertAlmostEqual(off_summary["Run Rate"], 0.3358070500927644, places=12)

    def test_golden_trend_points(self) -> None:
        trend = (
            self.team_plays.groupby(["display_week", "is_on"], dropna=False)
            .agg(plays=("epa", "size"), epa=("epa", "mean"))
            .reset_index()
            .sort_values(["display_week", "is_on"])
        )

        self.assertEqual(len(trend), 32)

        checks = [
            (1, 1, 32, 0.19082630128125),
            (2, 0, 37, 0.32980789632432433),
            (2, 1, 23, -0.22669541656521738),
            (11, 1, 64, 0.151006553953125),
            (17, 0, 69, 0.178616424),
            (18, 1, 42, 0.04437689971428574),
        ]
        for week, is_on, exp_plays, exp_epa in checks:
            row = trend[(trend["display_week"] == week) & (trend["is_on"] == is_on)].iloc[0]
            self.assertEqual(int(row["plays"]), exp_plays)
            self.assertAlmostEqual(float(row["epa"]), exp_epa, places=12)

    def test_golden_personnel_points(self) -> None:
        personnel = (
            self.team_plays.assign(personnel_label=self.team_plays["personnel"].astype("Int64").astype("string").fillna("Unknown"))
            .groupby(["personnel_label", "is_on"], dropna=False)
            .agg(plays=("epa", "size"), epa=("epa", "mean"))
            .reset_index()
        )

        self.assertEqual(len(personnel), 21)

        checks = [
            ("11", 1, 308, 0.04121945704545455),
            ("11", 0, 309, -0.006944086974110032),
            ("12", 1, 129, -0.04350697660465115),
            ("12", 0, 98, 0.10628485345918368),
            ("Unknown", 0, 3, -0.9698276726666665),
        ]
        for personnel_label, is_on, exp_plays, exp_epa in checks:
            row = personnel[(personnel["personnel_label"] == personnel_label) & (personnel["is_on"] == is_on)].iloc[0]
            self.assertEqual(int(row["plays"]), exp_plays)
            self.assertAlmostEqual(float(row["epa"]), exp_epa, places=12)


if __name__ == "__main__":
    unittest.main()
