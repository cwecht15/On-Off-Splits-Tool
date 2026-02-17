# NFL On/Off Splits Tool

Internal Streamlit app for NFL player on/off splits using:
- `participation.csv`
- `play_by_play_data.csv`
- `epa.csv`
- `weekly_rosters.csv`
- `pp_data.csv`

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy To Streamlit Community Cloud (Option A)

### 1) Create a GitHub repo and push this project

Because your CSV files are large, use Git LFS.

```powershell
git init
git lfs install
git lfs track "*.csv"
git add .gitattributes
git add .
git commit -m "Initial NFL on/off app"
git branch -M main
git remote add origin https://github.com/<your-org-or-user>/<repo-name>.git
git push -u origin main
```

Notes:
- GitHub rejects files over 100 MB without LFS.
- Keep the repo private if data is sensitive.

### 2) Deploy on Streamlit

1. Go to `https://share.streamlit.io`
2. Click `Create app`
3. Select your GitHub repo + branch `main`
4. Set main file path to `app.py`
5. Click `Deploy`

### 3) Share with your team

- Use app sharing controls in Streamlit to grant access.
- Share the generated app URL.

## Tests

```powershell
python -m unittest tests\test_data_correctness.py tests\test_golden_regression.py tests\test_golden_regression_defense.py -v
```

