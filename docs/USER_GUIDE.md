# User Guide

## Purpose

This repo now exposes three separate Streamlit apps:
- `app_on_off.py`: team-level on/off splits for selected player(s)
- `app_snap_threshold.py`: snap-threshold split workflow
- `app_leaderboard.py`: top-50 on/off delta leaderboard

The workflow below describes `app_on_off.py`.

Use cases for `app_on_off.py`:
- quantify impact when a player (or group of players) is on field vs off field
- compare offense and defense effects side-by-side
- inspect weekly trend behavior
- inspect splits by offensive personnel

## Data Sources

The app reads from local CSV files in the project root:
- `participation.csv`
- `play_by_play_data.csv`
- `epa.csv`
- `weekly_rosters.csv`
- `pp_data.csv`

## Sidebar Workflow

1. Select `Seasons`
2. Select `Weeks` (regular season = `1-18`, playoffs = `19-22`)
3. Select `Team`
4. Select `Players` (multi-select)
5. Set `Quarter`, `Down`, `Distance to go`, `Red Zone`, `Inside 10`, `Score margin`
6. Set `Minimum plays (On and Off)`
7. Set `On definition`:
   - `Any selected player on field`
   - `All selected players on field`

## Main Output Sections

- `Offense On/Off`
- `Defense On/Off`
- `Offense Baseline (Team vs League)`
- `Defense Baseline (Team vs League)`
- `Offense Trend (EPA/play)`
- `Defense Trend (EPA/play)`
- `Offense Personnel Splits`
- `Defense Personnel Splits (Opponent Offensive Personnel)`
- `Export`:
  - CSV downloads
  - PDF report generation and download

## Interpreting The Tables

- `On`: team plays where selected player condition is true
- `Off`: same team plays where condition is false
- `On - Off`: difference

Metrics:
- `Plays`
- `EPA/play`
- `Success Rate`
- `Pass Rate`
- `Run Rate`

## Notes

- Results are always relative to the selected team and filters.
- Lower EPA allowed on defense is generally better.
- Very small samples can be noisy; use `Minimum plays` to stabilize.
