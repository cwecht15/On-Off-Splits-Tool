# NFL On/Off Splits Tool

Internal Streamlit dashboard for NFL player on/off splits with offense and defense views, trend charts, personnel splits, and export.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Data Setup

Data source priority:

1. Postgres (if configured)
2. External CSV URLs/local files fallback

### Postgres (recommended)

Set `[postgres]` secrets (see `.streamlit/secrets.example.toml`):

- `dsn`
- optional table overrides:
  - `participation_table`
  - `play_by_play_data_table`
  - `epa_table`
  - `weekly_rosters_table`
  - `pp_data_table`

Before deploying on Postgres, load typed tables:

```powershell
$env:POSTGRES_DSN = "postgresql://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require"
python scripts/load_postgres_typed.py --csv-dir .
```

### CSV URL fallback

If Postgres is not configured, the app requires external CSV URLs (CSV files are not committed to git):

- `participation.csv`
- `play_by_play_data.csv`
- `epa.csv`
- `weekly_rosters.csv`
- `pp_data.csv`

Set these in Streamlit Cloud secrets under `[data_urls]` (see `docs/DEPLOYMENT_STREAMLIT_CLOUD.md`), or via env vars like `DATA_URL_PARTICIPATION_CSV`, or use `DATA_BASE_URL`.

## Documentation

- Product and workflow: `docs/USER_GUIDE.md`
- Metrics, filters, and data logic: `docs/METRICS_AND_DATA_LOGIC.md`
- Streamlit Cloud deployment + troubleshooting: `docs/DEPLOYMENT_STREAMLIT_CLOUD.md`
- Test coverage and commands: `docs/TESTING.md`

## Core Files

- App: `app.py`
- Requirements: `requirements.txt`
- Streamlit config: `.streamlit/config.toml`
- Runtime pin: `runtime.txt`
- Tests: `tests/test_data_correctness.py`, `tests/test_golden_regression.py`, `tests/test_golden_regression_defense.py`
