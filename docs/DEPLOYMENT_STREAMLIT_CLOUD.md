# Deployment: Streamlit Community Cloud

## Prerequisites

- GitHub account
- Repo containing this project
- Git LFS installed locally (required for large CSV files)

## Repository Setup

From project root:

```powershell
git init
git lfs install
git lfs track "*.csv"
git add .gitattributes
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<user-or-org>/<repo>.git
git push -u origin main
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

Large-file push failures on GitHub:
- Ensure `git lfs track "*.csv"` and re-push.

## Update Workflow

```powershell
git add .
git commit -m "Describe change"
git push
```

Streamlit Cloud auto-redeploys after push.
