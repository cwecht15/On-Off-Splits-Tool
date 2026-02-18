import os

os.environ["APP_MODE"] = "leaderboard"
os.environ.setdefault("ENABLE_POSTGRES", "0")

import app  # noqa: F401,E402
