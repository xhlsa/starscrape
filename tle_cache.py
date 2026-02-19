"""TLE cache management for Starlink satellites.

Fetches TLE data from CelesTrak and caches locally as JSON.
Re-fetches automatically when the cache is older than CACHE_MAX_AGE_HOURS.
Flags individual TLEs whose epoch is older than TLE_EPOCH_WARN_DAYS as
having degraded accuracy (satellite position may drift from propagated value).
"""

import json
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

CELESTRAK_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
)
CACHE_FILE = Path(__file__).parent / ".tle_cache.json"
CACHE_MAX_AGE_HOURS = 2
TLE_EPOCH_WARN_DAYS = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tles(force_refresh: bool = False) -> list[dict]:
    """Return Starlink TLE records, using a local cache when fresh.

    Each record is a dict with keys:
        name            – satellite name string
        norad_id        – int NORAD catalog number
        line1 / line2   – raw TLE strings
        epoch_age_days  – float days since TLE epoch (recomputed at load time)
        degraded        – bool True when epoch_age_days > TLE_EPOCH_WARN_DAYS
    """
    if not force_refresh:
        cached = _load_cache()
        if cached is not None:
            return _annotate_epoch_ages(cached)

    tles = _fetch_and_parse()
    _save_cache(tles)
    return _annotate_epoch_ages(tles)


# ---------------------------------------------------------------------------
# Fetch / parse
# ---------------------------------------------------------------------------

def _fetch_and_parse() -> list[dict]:
    """Download TLEs from CelesTrak and return raw records (no epoch_age)."""
    try:
        resp = requests.get(CELESTRAK_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch TLEs from CelesTrak: {exc}") from exc

    return _parse_tle_text(resp.text)


def _parse_tle_text(text: str) -> list[dict]:
    """Parse raw TLE text into a list of minimal record dicts."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    records = []
    i = 0
    while i + 2 <= len(lines) - 1:
        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]
        if line1.startswith("1 ") and line2.startswith("2 "):
            try:
                norad_id = int(line1[2:7])
            except ValueError:
                i += 1
                continue
            records.append(
                {
                    "name": name,
                    "norad_id": norad_id,
                    "line1": line1,
                    "line2": line2,
                    "is_dtc": "[DTC]" in name,
                }
            )
            i += 3
        else:
            i += 1
    return records


# ---------------------------------------------------------------------------
# Epoch age
# ---------------------------------------------------------------------------

def _tle_epoch_age_days(line1: str) -> float:
    """Compute how many days ago the TLE epoch was, relative to UTC now."""
    # TLE line 1 epoch occupies columns 19–32 (1-indexed) = indices 18:32
    # Format: YYddd.dddddddd  (2-digit year + day-of-year with decimal)
    year_2d = int(line1[18:20])
    day_frac = float(line1[20:32])
    year = (2000 + year_2d) if year_2d < 57 else (1900 + year_2d)
    epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_frac - 1.0)
    age = (datetime.now(timezone.utc) - epoch).total_seconds() / 86400.0
    return age


def _annotate_epoch_ages(tles: list[dict]) -> list[dict]:
    """Recompute epoch_age_days, degraded, and is_dtc in-place (based on now)."""
    for rec in tles:
        age = _tle_epoch_age_days(rec["line1"])
        rec["epoch_age_days"] = age
        rec["degraded"] = age > TLE_EPOCH_WARN_DAYS
        rec["is_dtc"] = "[DTC]" in rec["name"]
    return tles


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _load_cache() -> list[dict] | None:
    """Return cached TLE records if the cache file is fresh, else None."""
    if not CACHE_FILE.exists():
        return None
    try:
        with CACHE_FILE.open() as fh:
            data = json.load(fh)
        fetch_time = datetime.fromisoformat(data["fetch_time"])
        age_h = (datetime.now(timezone.utc) - fetch_time).total_seconds() / 3600.0
        if age_h > CACHE_MAX_AGE_HOURS:
            return None
        return data["tles"]
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _save_cache(tles: list[dict]) -> None:
    """Persist TLE records with a UTC timestamp."""
    payload = {
        "fetch_time": datetime.now(timezone.utc).isoformat(),
        "tles": tles,
    }
    with CACHE_FILE.open("w") as fh:
        json.dump(payload, fh)
