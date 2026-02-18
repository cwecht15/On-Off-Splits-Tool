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
Apps download required CSV files from URLs provided in secrets/environment.

## Deploy Three Separate Apps

Create three Streamlit apps pointing to the same repo/branch, each with a different main file:

1. `app_on_off.py`
2. `app_snap_threshold.py`
3. `app_leaderboard.py`

## Data Source Configuration

Required files:

- `participation.csv`
- `play_by_play_data.csv`
- `epa.csv`
- `weekly_rosters.csv`
- `pp_data.csv`

Example `.streamlit/secrets.toml`:

```toml
[data_urls]
participation.csv = "https://<your-storage>/participation.csv"
play_by_play_data.csv = "https://<your-storage>/play_by_play_data.csv"
epa.csv = "https://<your-storage>/epa.csv"
weekly_rosters.csv = "https://<your-storage>/weekly_rosters.csv"
pp_data.csv = "https://<your-storage>/pp_data.csv"
```

Important:
- Each Streamlit app has its own secrets store.
- Paste the same `[data_urls]` block into all deployed app instances.

## Deploy Steps

1. Open `https://share.streamlit.io`
2. Click `Create app`
3. Select repo and branch `main`
4. Set main file path to one of:
   - `app_on_off.py`
   - `app_snap_threshold.py`
   - `app_leaderboard.py`
5. Deploy

## Sharing

- Keep repo private if data is sensitive.
- Use Streamlit app sharing controls to add users.
- Send team the relevant app URL(s).

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

`Missing required data file ...`
- Cause: missing/incomplete `[data_urls]` block in this specific app instance.
- Fix: set all five file URLs in secrets, then clear cache and reboot.

## Update Workflow

```powershell
git add .
git commit -m "Describe change"
git push
```

Streamlit Cloud auto-redeploys after push.
