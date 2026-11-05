import functools
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=1)
def commit() -> str | None:
    """Return the commit of the app: OCRSD_COMMIT from the Docker build, else `git rev-parse HEAD`, else None."""
    env_commit = os.environ.get("OCRSD_COMMIT", "").strip()
    if env_commit and env_commit != "unknown":
        return env_commit
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                text=True, timeout=2, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
