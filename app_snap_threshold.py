import os

os.environ["APP_MODE"] = "snap_threshold"
os.environ.setdefault("ENABLE_POSTGRES", "0")

import app  # noqa: F401,E402
