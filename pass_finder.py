"""Coarse and fine pass detection for Starlink satellites.

Algorithm
---------
1. Coarse scan: 60-second grid.  Any timestep with elevation > min_elevation
   starts a potential pass; the pass ends when the satellite drops below.
2. Fine refinement:
   - Rise / set times: linear interpolation between the last below-horizon
     and first above-horizon samples (and vice versa).  60-second grid steps
     give < 30 s error before interpolation.
   - Peak time / elevation: 3-point parabolic interpolation around the grid
     maximum.
3. Peak azimuth is taken at the grid index of maximum elevation.
4. Minimum slant range is the nanmin over the above-horizon window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from propagator import orbital_altitude_km


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Pass:
    """A single satellite pass over the observer."""

    sat_name: str
    norad_id: int
    rise_time: datetime
    peak_time: datetime
    set_time: datetime
    peak_el_deg: float
    peak_az_deg: float
    min_slant_km: float
    orbital_alt_km: float
    epoch_age_days: float
    degraded: bool
    link_likelihood: str = ""
    link_note: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_passes(
    az_deg: np.ndarray,
    el_deg: np.ndarray,
    slant_km: np.ndarray,
    tles: list[dict],
    times: list[datetime],
    min_elevation_deg: float = 0.0,
) -> list[Pass]:
    """Detect all satellite passes from pre-computed topocentric arrays.

    Parameters
    ----------
    az_deg           : shape (n_sats, n_times)
    el_deg           : shape (n_sats, n_times)
    slant_km         : shape (n_sats, n_times)
    tles             : TLE record dicts, same order as sat dimension
    times            : datetime list, same order as time dimension
    min_elevation_deg: minimum elevation to count as a pass (default 0°)

    Returns
    -------
    Unsorted list of Pass objects.
    """
    passes: list[Pass] = []
    n_sats = el_deg.shape[0]

    for sat_idx in range(n_sats):
        el = el_deg[sat_idx]     # (n_times,)
        az = az_deg[sat_idx]
        sl = slant_km[sat_idx]
        tle = tles[sat_idx]

        sat_passes = _detect_satellite_passes(
            el, az, sl, tle, times, min_elevation_deg
        )
        passes.extend(sat_passes)

    return passes


# ---------------------------------------------------------------------------
# Per-satellite pass detection
# ---------------------------------------------------------------------------

def _detect_satellite_passes(
    el: np.ndarray,
    az: np.ndarray,
    sl: np.ndarray,
    tle: dict,
    times: list[datetime],
    min_el: float,
) -> list[Pass]:
    """Find all passes for a single satellite."""
    n = len(el)

    # NaN from failed SGP4 propagations → treat as below horizon
    above = np.where(np.isnan(el), False, el > min_el)

    # Detect transitions using a sentinel-padded array
    # padded[i+1] == above[i], padded[0] = padded[n+1] = False
    padded = np.empty(n + 2, dtype=bool)
    padded[0] = False
    padded[1 : n + 1] = above
    padded[n + 1] = False

    diff = np.diff(padded.view(np.int8))
    # diff[i] == 1  → above[i] is first True  (satellite rising through min_el)
    # diff[i] == -1 → above[i-1] is last True  (satellite setting)
    rise_indices = np.where(diff == 1)[0]
    set_indices  = np.where(diff == -1)[0]

    if rise_indices.size == 0:
        return []

    alt_km = orbital_altitude_km(tle["line2"])
    passes: list[Pass] = []

    for ri, si in zip(rise_indices, set_indices):
        # ri: first above-horizon index in el[]
        # si: first below-horizon index after the pass (== ri if degenerate)
        if si <= ri:
            continue

        # Peak elevation within the pass window
        window_el = el[ri:si]
        valid_mask = ~np.isnan(window_el)
        if not valid_mask.any():
            continue

        local_peak = int(np.nanargmax(window_el))
        peak_idx = ri + local_peak

        # Refined times
        peak_time = _refine_peak(times, el, peak_idx)
        rise_time = _refine_crossing(times, el, ri - 1, ri, min_el)
        set_time  = _refine_crossing(times, el, si - 1, si, min_el)

        # Peak az at grid peak index; min slant over pass
        peak_az  = float(az[peak_idx]) if not np.isnan(az[peak_idx]) else 0.0
        min_sl   = float(np.nanmin(sl[ri:si]))
        peak_el  = float(el[peak_idx])

        passes.append(
            Pass(
                sat_name=tle["name"],
                norad_id=tle["norad_id"],
                rise_time=rise_time,
                peak_time=peak_time,
                set_time=set_time,
                peak_el_deg=peak_el,
                peak_az_deg=peak_az,
                min_slant_km=min_sl,
                orbital_alt_km=alt_km,
                epoch_age_days=tle["epoch_age_days"],
                degraded=tle["degraded"],
            )
        )

    return passes


# ---------------------------------------------------------------------------
# Time-refinement helpers
# ---------------------------------------------------------------------------

def _refine_peak(times: list[datetime], el: np.ndarray, idx: int) -> datetime:
    """Refine peak time via 3-point parabolic interpolation.

    Falls back to the grid time if the index is at a boundary or if
    neighbouring samples are NaN.
    """
    if idx <= 0 or idx >= len(el) - 1:
        return times[idx]

    y0 = el[idx - 1]
    y1 = el[idx]
    y2 = el[idx + 1]
    if np.isnan(y0) or np.isnan(y2):
        return times[idx]

    denom = 2.0 * (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-10:
        return times[idx]

    # Fractional offset from idx, in units of grid step
    t_frac = (y0 - y2) / denom          # ∈ (−0.5, 0.5)
    dt_s = (times[idx + 1] - times[idx]).total_seconds()
    return times[idx] + timedelta(seconds=t_frac * dt_s)


def _refine_crossing(
    times: list[datetime],
    el: np.ndarray,
    lo: int,
    hi: int,
    threshold: float,
) -> datetime:
    """Refine a horizon crossing via linear interpolation between grid points.

    Parameters
    ----------
    lo, hi    : indices bracketing the crossing (lo < threshold ≤ hi  or
                hi < threshold ≤ lo for set crossings — handled symmetrically)
    threshold : elevation angle to find (degrees)
    """
    n = len(el)
    lo = max(0, lo)
    hi = min(n - 1, hi)
    if lo == hi:
        return times[lo]

    el_lo = el[lo] if not np.isnan(el[lo]) else threshold - 1.0
    el_hi = el[hi] if not np.isnan(el[hi]) else threshold - 1.0

    span = el_hi - el_lo
    if abs(span) < 1e-10:
        return times[lo]

    frac = np.clip((threshold - el_lo) / span, 0.0, 1.0)
    dt_total = (times[hi] - times[lo]).total_seconds()
    return times[lo] + timedelta(seconds=float(frac) * dt_total)
