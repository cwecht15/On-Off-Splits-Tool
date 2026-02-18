# Deployment: Streamlit Community Cloud

## Prerequisites

- GitHub account
- Repo containing this project

## Repository Setup

From project root:

```powershell
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<user-or-org>/<repo>.git
git push -u origin main
```

Important: CSV data files are intentionally not tracked in git.
The app uses Postgres when configured, otherwise it downloads required files from URLs you provide in secrets/environment.

## Postgres Configuration (Recommended)

In Streamlit Cloud app settings, add secrets:

```toml
[postgres]
dsn = "postgresql://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require"
participation_table = "participation"
play_by_play_data_table = "play_by_play_data"
epa_table = "epa"
weekly_rosters_table = "weekly_rosters"
pp_data_table = "pp_data"
```

Notes:

- `dsn` can also be set as env var `POSTGRES_DSN`.
- Table names are optional if you use defaults above.
- Per-table env vars are also supported (`POSTGRES_TABLE_PARTICIPATION`, etc.).

## Data Source Configuration

Required files:

- `participation.csv`
- `play_by_play_data.csv`
- `epa.csv`
- `weekly_rosters.csv`
- `pp_data.csv`

Configure one of these options when Postgres is not set:

1. Streamlit secrets (recommended)
2. Per-file env vars (`DATA_URL_PARTICIPATION_CSV`, etc.)
3. Base URL env var (`DATA_BASE_URL`) that serves all required filenames

Example `.streamlit/secrets.toml`:

```toml
[data_urls]
participation.csv = "https://<your-storage>/participation.csv"
play_by_play_data.csv = "https://<your-storage>/play_by_play_data.csv"
epa.csv = "https://<your-storage>/epa.csv"
weekly_rosters.csv = "https://<your-storage>/weekly_rosters.csv"
pp_data.csv = "https://<your-storage>/pp_data.csv"
```

## Deploy Steps

1. Open `https://share.streamlit.io`
2. Click `Create app`
3. Select repo and branch `main`
4. Set main file path to `app.py`
5. Deploy

## Sharing

- Keep repo private if data is sensitive.
- Use Streamlit app sharing controls to add users.
- Send team the app URL.

## Config Used In This Project

- `.streamlit/config.toml`
- `runtime.txt`
- `requirements.txt`

## Dependency Compatibility Notes

The app is pinned for current cloud compatibility:
- modern Streamlit and Altair compatibility
- Python 3.13-compatible package set

If deployment fails, review recent `requirements.txt` changes first.

## Common Errors And Fixes

`ModuleNotFoundError: altair.vegalite.v4`
- Cause: old Streamlit + new Altair mismatch.
- Fix: upgrade Streamlit and pin compatible Altair (already handled in this repo).

`ModuleNotFoundError: imghdr`
- Cause: old Streamlit on Python 3.13.
- Fix: upgrade Streamlit (already handled in this repo).

App still using stale packages:
- In Streamlit Cloud: `Clear cache` and `Reboot app`.

Git LFS budget / clone failures:
- Remove tracked CSVs from git and keep them in external storage.
- Confirm Streamlit secrets include valid URLs for all required files.

## Update Workflow

```powershell
git add .
git commit -m "Describe change"
git push
```

Streamlit Cloud auto-redeploys after push.
