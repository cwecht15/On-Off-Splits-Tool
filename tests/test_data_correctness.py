import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_eligible_plays() -> pd.DataFrame:
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
        low_memory=False,
    ).rename(columns={"game_id": "gameId", "play_id": "playId", "seas_type": "seasonType"})

    epa = pd.read_csv(ROOT / "epa.csv", usecols=["gameId", "playId", "epa"], low_memory=False)

    for c in [
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
        pbp[c] = pd.to_numeric(pbp[c], errors="coerce").fillna(0)

    pbp["yards_gained"] = pbp["passing_yards"] + pbp["rushing_yards"]

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
    plays["offense"] = plays["offense"].fillna("").astype(str).str.strip().str.upper()
    plays["defense"] = plays["defense"].fillna("").astype(str).str.strip().str.upper()
    return plays


def build_success_flag(df: pd.DataFrame) -> pd.Series:
    down = pd.to_numeric(df["down"], errors="coerce")
    distance = pd.to_numeric(df["distance"], errors="coerce")
    gain = pd.to_numeric(df["yards_gained"], errors="coerce").fillna(0)

    first = (down == 1) & (gain >= 0.4 * distance)
    second = (down == 2) & (gain >= 0.6 * distance)
    third_fourth = (down >= 3) & (gain >= distance)
    return (first | second | third_fourth) & (distance > 0)


class TestOnOffDataCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plays = load_eligible_plays()

        part_cols = ["gameId", "playId", "team", "role"] + [f"P{i}" for i in range(1, 16)]
        cls.part = pd.read_csv(ROOT / "participation.csv", usecols=part_cols, low_memory=False)
        cls.part["team"] = cls.part["team"].fillna("").astype(str).str.strip().str.upper()
        cls.part["role"] = cls.part["role"].fillna("").astype(str).str.strip().str.lower()

    def test_success_rate_rule(self) -> None:
        df = pd.DataFrame(
            {
                "down": [1, 1, 2, 2, 3, 4],
                "distance": [10, 10, 10, 10, 3, 5],
                "yards_gained": [4, 3, 6, 5, 3, 4],
            }
        )
        got = build_success_flag(df).tolist()
        expected = [True, False, True, False, True, False]
        self.assertEqual(got, expected)

    def test_on_off_partitions_team_plays(self) -> None:
        first_off = self.part[self.part["role"] == "offense"].iloc[0]
        team = first_off["team"]
        player_id = str(first_off["P1"]).strip()

        team_plays = self.plays[self.plays["offense"] == team].copy()

        part_long = (
            self.part.melt(
                id_vars=["gameId", "playId", "team", "role"],
                value_vars=[f"P{i}" for i in range(1, 16)],
                value_name="player_id",
            )
            .drop(columns=["variable"])
            .dropna(subset=["player_id"])
        )
        part_long["player_id"] = part_long["player_id"].astype(str).str.strip()

        on_keys = (
            part_long[
                (part_long["team"] == team)
                & (part_long["role"] == "offense")
                & (part_long["player_id"] == player_id)
            ][["gameId", "playId"]]
            .drop_duplicates()
            .assign(is_on=1)
        )

        merged = team_plays.merge(on_keys, on=["gameId", "playId"], how="left")
        merged["is_on"] = merged["is_on"].fillna(0).astype(int)

        on_n = int((merged["is_on"] == 1).sum())
        off_n = int((merged["is_on"] == 0).sum())
        total_n = int(len(merged))

        self.assertEqual(on_n + off_n, total_n)
        self.assertGreater(on_n, 0)
        self.assertGreater(off_n, 0)

    def test_team_epa_equals_weighted_on_off_epa(self) -> None:
        first_off = self.part[self.part["role"] == "offense"].iloc[0]
        team = first_off["team"]
        player_id = str(first_off["P1"]).strip()

        team_plays = self.plays[self.plays["offense"] == team].copy()

        part_long = (
            self.part.melt(
                id_vars=["gameId", "playId", "team", "role"],
                value_vars=[f"P{i}" for i in range(1, 16)],
                value_name="player_id",
            )
            .drop(columns=["variable"])
            .dropna(subset=["player_id"])
        )
        part_long["player_id"] = part_long["player_id"].astype(str).str.strip()

        on_keys = (
            part_long[
                (part_long["team"] == team)
                & (part_long["role"] == "offense")
                & (part_long["player_id"] == player_id)
            ][["gameId", "playId"]]
            .drop_duplicates()
            .assign(is_on=1)
        )

        merged = team_plays.merge(on_keys, on=["gameId", "playId"], how="left")
        merged["is_on"] = merged["is_on"].fillna(0).astype(int)

        on_df = merged[merged["is_on"] == 1]
        off_df = merged[merged["is_on"] == 0]

        team_epa = merged["epa"].mean()
        weighted_epa = (
            (on_df["epa"].mean() * len(on_df) + off_df["epa"].mean() * len(off_df)) / len(merged)
        )

        self.assertTrue(np.isfinite(team_epa))
        self.assertTrue(np.isfinite(weighted_epa))
        self.assertAlmostEqual(float(team_epa), float(weighted_epa), places=10)

    def test_only_pass_or_run_in_filtered_plays(self) -> None:
        bad = self.plays[(self.plays["pass_play"] != 1) & (self.plays["run_play"] != 1)]
        self.assertEqual(len(bad), 0)


if __name__ == "__main__":
    unittest.main()
