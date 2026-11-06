import json
import threading
import time
import urllib.request

REPOSITORY = "siqueiratf/ocrsd"
SUCCESS_TTL = 3600.0
FAILURE_TTL = 600.0
TIMEOUT = 3.0

_lock = threading.Lock()
_cache: dict = {"stars": None, "expires": 0.0}


def fetch_stars() -> int | None:
    request = urllib.request.Request(f"https://api.github.com/repos/{REPOSITORY}",
                                     headers={"Accept": "application/vnd.github+json", "User-Agent": "ocrsd"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            stars = json.load(response).get("stargazers_count")
    except (OSError, ValueError):
        return None
    return stars if isinstance(stars, int) and stars >= 0 else None


def stars(now: float | None = None) -> int | None:
    """Return the cached star count of the repository; refresh it at most once an hour (10 minutes after a failure)."""
    now = time.time() if now is None else now
    with _lock:
        if now < _cache["expires"]:
            return _cache["stars"]
        value = fetch_stars()
        if value is not None:
            _cache["stars"] = value
        _cache["expires"] = now + (SUCCESS_TTL if value is not None else FAILURE_TTL)
        return _cache["stars"]


def compact(value: int) -> str:
    if value < 1000:
        return str(value)
    return f"{value / 1000:.1f}".rstrip("0").rstrip(".").replace(".", ",") + " mil"
