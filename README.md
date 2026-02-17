# NFL On/Off Splits Tool

Internal Streamlit dashboard for NFL player on/off splits with offense and defense views, trend charts, personnel splits, and export.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

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
