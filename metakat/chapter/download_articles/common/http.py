from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

USER_AGENT = "MetaKat-article-sampler/0.1"

# Requests are sequential, and at least this many seconds apart, across the whole process.
_min_interval = 1.0
_last_request = 0.0


def set_min_interval(seconds: float) -> None:
    global _min_interval
    _min_interval = seconds


def http_get(url: str, timeout: float = 120, retries: int = 5, backoff: float = 10) -> bytes:
    """GET with retries, at most one request per ``set_min_interval`` seconds.

    A server that asks to slow down (429/503) is waited for as long as its Retry-After says, at least
    ``backoff`` seconds times the attempt. Refusals (401/403/404/410) are raised at once.
    """
    global _last_request
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        time.sleep(max(0.0, _last_request + _min_interval - time.monotonic()))
        _last_request = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:
            code = getattr(error, "code", None)
            if attempt == retries or code in (401, 403, 404, 410):
                raise
            wait = backoff * attempt
            if code in (429, 503):
                retry_after = error.headers.get("Retry-After", "")
                wait = max(wait, float(retry_after) if retry_after.isdigit() else 0)
            logger.warning(f"GET {url} failed ({error}), waiting {wait:.0f} s, attempt {attempt}/{retries}")
            time.sleep(wait)
    raise AssertionError("unreachable")
