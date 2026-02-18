import os

os.environ["APP_MODE"] = "snap_threshold"
os.environ.setdefault("ENABLE_POSTGRES", "0")

import app_multi  # noqa: F401,E402
