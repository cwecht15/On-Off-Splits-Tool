import unittest

from tests.test_golden_regression import load_app_like_frames, summarize


class TestGoldenRegressionDefense(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        plays, part_long = load_app_like_frames()

        team = "BUF"
        player_id = "00-0036914"
        role = "defense"
        season = 2021
        weeks = list(range(1, 19))

        plays = plays[(plays["season"] == season) & (plays["display_week"].isin(weeks))]
        part_long = part_long[(part_long["season"] == season) & (part_long["display_week"].isin(weeks))]

        team_plays = plays[plays["defense"] == team].copy()
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

    def test_golden_defense_on_off_split_metrics(self) -> None:
        on_summary = summarize(self.team_plays[self.team_plays["is_on"] == 1])
        off_summary = summarize(self.team_plays[self.team_plays["is_on"] == 0])

        self.assertEqual(on_summary["Plays"], 497)
        self.assertAlmostEqual(on_summary["EPA/play"], -0.05588849302213278, places=12)
        self.assertAlmostEqual(on_summary["Success Rate"], 0.4164989939637827, places=12)
        self.assertAlmostEqual(on_summary["Pass Rate"], 0.5271629778672032, places=12)
        self.assertAlmostEqual(on_summary["Run Rate"], 0.47283702213279677, places=12)

        self.assertEqual(off_summary["Plays"], 506)
        self.assertAlmostEqual(off_summary["EPA/play"], -0.22831257288142293, places=12)
        self.assertAlmostEqual(off_summary["Success Rate"], 0.41106719367588934, places=12)
        self.assertAlmostEqual(off_summary["Pass Rate"], 0.6798418972332015, places=12)
        self.assertAlmostEqual(off_summary["Run Rate"], 0.3201581027667984, places=12)

    def test_golden_defense_trend_points(self) -> None:
        trend = (
            self.team_plays.groupby(["display_week", "is_on"], dropna=False)
            .agg(plays=("epa", "size"), epa=("epa", "mean"))
            .reset_index()
            .sort_values(["display_week", "is_on"])
        )

        self.assertEqual(len(trend), 34)

        checks = [
            (1, 1, 28, -0.14285164046428572),
            (1, 0, 24, 0.034738525583333346),
            (2, 1, 46, -0.3202340243695652),
            (2, 0, 23, -0.724631878826087),
            (6, 1, 23, 0.5863730211304348),
            (6, 0, 27, -0.03261404733333329),
            (13, 1, 20, -0.043960028800000016),
            (13, 0, 27, 0.0038890937037037022),
            (18, 1, 24, -0.587560586125),
            (18, 0, 22, -0.44372319936363636),
        ]
        for week, is_on, exp_plays, exp_epa in checks:
            row = trend[(trend["display_week"] == week) & (trend["is_on"] == is_on)].iloc[0]
            self.assertEqual(int(row["plays"]), exp_plays)
            self.assertAlmostEqual(float(row["epa"]), exp_epa, places=12)

    def test_golden_defense_personnel_points(self) -> None:
        personnel = (
            self.team_plays.assign(
                personnel_label=self.team_plays["personnel"].astype("Int64").astype("string").fillna("Unknown")
            )
            .groupby(["personnel_label", "is_on"], dropna=False)
            .agg(plays=("epa", "size"), epa=("epa", "mean"))
            .reset_index()
        )

        self.assertEqual(len(personnel), 23)

        checks = [
            ("11", 1, 273, -0.0356171500989011),
            ("11", 0, 310, -0.2508667121548387),
            ("12", 1, 118, -0.16589189840677965),
            ("12", 0, 108, -0.21061445039814813),
            ("22", 1, 26, 0.026119737192307663),
            ("22", 0, 20, 0.17298820205),
        ]
        for personnel_label, is_on, exp_plays, exp_epa in checks:
            row = personnel[(personnel["personnel_label"] == personnel_label) & (personnel["is_on"] == is_on)].iloc[0]
            self.assertEqual(int(row["plays"]), exp_plays)
            self.assertAlmostEqual(float(row["epa"]), exp_epa, places=12)


if __name__ == "__main__":
    unittest.main()
