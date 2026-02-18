# NFL On/Off Splits Tool

Internal Streamlit dashboards for NFL player/team split analysis.

The repo now supports three deployable app entrypoints:
- `app_on_off.py` (core offense/defense on-off workflow)
- `app_snap_threshold.py` (snap-threshold split workflow)
- `app_leaderboard.py` (top-50 leaderboard workflow)

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app_on_off.py
```

Run other app variants locally:

```powershell
python -m streamlit run app_snap_threshold.py
python -m streamlit run app_leaderboard.py
```

## Data Setup (Current Recommended)

Use URL-based CSV hosting (for example S3 object URLs):

- `participation.csv`
- `play_by_play_data.csv`
- `epa.csv`
- `weekly_rosters.csv`
- `pp_data.csv`

Set these in Streamlit Cloud secrets under `[data_urls]` (see `docs/DEPLOYMENT_STREAMLIT_CLOUD.md`).

## Documentation

- Product and workflow: `docs/USER_GUIDE.md`
- Metrics, filters, and data logic: `docs/METRICS_AND_DATA_LOGIC.md`
- Streamlit Cloud deployment + troubleshooting: `docs/DEPLOYMENT_STREAMLIT_CLOUD.md`
- Test coverage and commands: `docs/TESTING.md`

## Core Files

- On/Off app: `app_on_off.py`
- Snap-threshold app: `app_snap_threshold.py`
- Leaderboard app: `app_leaderboard.py`
- Shared multi-feature module: `app_multi.py`
- Legacy combined app file: `app.py`
- Requirements: `requirements.txt`
- Streamlit config: `.streamlit/config.toml`
- Runtime pin: `runtime.txt`
- Tests: `tests/test_data_correctness.py`, `tests/test_golden_regression.py`, `tests/test_golden_regression_defense.py`
