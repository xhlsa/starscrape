"""Visualization data export for starscrape.

Produces viz_data.json consumed by viz.html.  Called from main.py when the
--viz flag is set.

Satellite positions are computed from observer-relative topocentric
coordinates (az/el/slant) using a spherical-Earth ENU→ECEF→geodetic
transform.  The result is the satellite's geocentric lat/lon (== sub-
satellite point on a sphere), suitable for Leaflet map display.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from analyzer import AnalysisResult
from link_budget import classify_shell
from pass_finder import Pass

# Spherical Earth radius (km) — consistent with geometry in link_budget.py
_RE_KM = 6371.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _sat_latlon(
    obs_lat_deg: float,
    obs_lon_deg: float,
    az_deg: np.ndarray,    # shape (n,)
    el_deg: np.ndarray,    # shape (n,)
    slant_km: np.ndarray,  # shape (n,)
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat_deg, lon_deg) arrays for satellite positions.

    Uses a spherical-Earth model.  Observer is at (obs_lat, obs_lon, 0 km).
    The satellite's geocentric lat/lon equals its sub-satellite point on a
    sphere, which is what we plot on the Leaflet map.
    """
    lat = math.radians(obs_lat_deg)
    lon = math.radians(obs_lon_deg)

    # Observer ECEF (spherical Earth)
    Ox = _RE_KM * math.cos(lat) * math.cos(lon)
    Oy = _RE_KM * math.cos(lat) * math.sin(lon)
    Oz = _RE_KM * math.sin(lat)

    # Topocentric ENU components
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    E = slant_km * np.cos(el) * np.sin(az)
    N = slant_km * np.cos(el) * np.cos(az)
    U = slant_km * np.sin(el)

    # ENU → ECEF via local rotation matrix
    slat, clat = math.sin(lat), math.cos(lat)
    slon, clon = math.sin(lon), math.cos(lon)
    dx = -slon * E - slat * clon * N + clat * clon * U
    dy =  clon * E - slat * slon * N + clat * slon * U
    dz =             clat *       N + slat *       U

    Sx, Sy, Sz = Ox + dx, Oy + dy, Oz + dz

    # Geocentric → geodetic (identical on a sphere)
    r = np.sqrt(Sx**2 + Sy**2 + Sz**2)
    sat_lat = np.degrees(np.arcsin(np.clip(Sz / r, -1.0, 1.0)))
    sat_lon = np.degrees(np.arctan2(Sy, Sx))
    return sat_lat, sat_lon


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def build_viz_data(
    obs_lat: float,
    obs_lon: float,
    times: list,              # list[datetime]
    all_passes: list[Pass],   # likely + marginal passes
    result: AnalysisResult,
    az_arr: np.ndarray,       # (n_sats, n_times)
    el_arr: np.ndarray,       # (n_sats, n_times)
    slant_arr: np.ndarray,    # (n_sats, n_times)
    valid_tles: list[dict],
) -> dict:
    """Build the complete visualization payload.

    Returns a dict ready for json.dump.  Key sections:
      observer          – lat/lon of the ground station
      window_start/end  – ISO-8601 timestamps
      timestep_sec      – propagation grid step (60 s)
      n_steps           – total number of timesteps
      summary           – high-level coverage stats
      handoff_schedule  – list of HandoffEntry dicts
      timesteps         – per-step compact stats (n_likely, n_marginal, best_el)
      trajectories      – per-pass list with per-step satellite positions
    """
    n_times = len(times)
    step_s  = (times[1] - times[0]).total_seconds() if n_times > 1 else 60.0
    t0      = times[0]

    norad_to_idx: dict[int, int] = {
        tle["norad_id"]: i for i, tle in enumerate(valid_tles)
    }

    # ── Satellite trajectories ─────────────────────────────────────────────
    trajectories: list[dict] = []

    for pass_ in all_passes:
        idx = norad_to_idx.get(pass_.norad_id)
        if idx is None:
            continue

        # Grid indices bracketing this pass
        ri = max(0, int((pass_.rise_time - t0).total_seconds() / step_s))
        si = min(n_times - 1,
                 int((pass_.set_time  - t0).total_seconds() / step_s) + 1)
        if ri > si:
            continue

        length   = si - ri + 1
        el_sl    = el_arr[idx,    ri:ri + length]
        az_sl    = az_arr[idx,    ri:ri + length]
        slant_sl = slant_arr[idx, ri:ri + length]

        # Only include timesteps where satellite is above the horizon
        above = ~np.isnan(el_sl) & (el_sl > 0.0)
        if not np.any(above):
            continue

        # Compute positions only for above-horizon steps (avoids NaN in trig)
        sel_el    = el_sl[above]
        sel_az    = az_sl[above]
        sel_slant = slant_sl[above]
        sat_lat, sat_lon = _sat_latlon(obs_lat, obs_lon,
                                       sel_az, sel_el, sel_slant)

        local_idxs = np.where(above)[0]
        points = [
            {
                "ti":    int(ri + li),
                "lat":   round(float(sat_lat[j]),   3),
                "lon":   round(float(sat_lon[j]),   3),
                "el":    round(float(sel_el[j]),    1),
                "az":    round(float(sel_az[j]),    1),
                "slant": int(round(float(sel_slant[j]))),
            }
            for j, li in enumerate(local_idxs)
        ]

        if not points:
            continue

        trajectories.append({
            "sat_name":       pass_.sat_name,
            "norad_id":       pass_.norad_id,
            "link":           pass_.link_likelihood,
            "shell":          classify_shell(pass_.orbital_alt_km),
            "orbital_alt_km": round(pass_.orbital_alt_km, 1),
            "peak_el_deg":    round(pass_.peak_el_deg, 1),
            "peak_az_deg":    round(pass_.peak_az_deg, 1),
            "min_slant_km":   int(round(pass_.min_slant_km)),
            "rise_time":      pass_.rise_time.isoformat(),
            "peak_time":      pass_.peak_time.isoformat(),
            "set_time":       pass_.set_time.isoformat(),
            "link_note":      pass_.link_note,
            "points":         points,
        })

    # ── Per-timestep summary (compact — no per-sat data here) ─────────────
    timesteps = []
    for i in range(n_times):
        el = result.best_el_deg[i]
        timesteps.append({
            "n_likely":   int(result.n_in_footprint[i]),
            "n_marginal": int(result.n_marginal[i]),
            "best_el":    round(float(el), 1) if not np.isnan(el) else None,
        })

    # ── Handoff schedule ──────────────────────────────────────────────────
    schedule: list[dict] = []
    for e in result.handoff_schedule:
        rec: dict = {
            "start":            e.start_time.isoformat(),
            "end":              e.end_time.isoformat(),
            "duration_min":     round(e.duration_min, 1),
            "is_gap":           e.is_gap,
            "is_marginal_fill": e.is_marginal_fill,
        }
        if not e.is_gap:
            rec.update({
                "sat_name":       e.sat_name,
                "norad_id":       e.norad_id,
                "peak_el_deg":    round(e.peak_el_deg, 1),
                "orbital_alt_km": round(e.orbital_alt_km, 1),
                "shell":          e.shell,
                "link":           e.link_likelihood,
            })
        schedule.append(rec)

    # ── Summary stats ──────────────────────────────────────────────────────
    summary = {
        "window_min":            result.window_min,
        "covered_min":           result.covered_min,
        "coverage_pct":          round(result.coverage_pct, 1),
        "any_coverage_min":      result.any_coverage_min,
        "any_coverage_pct":      round(result.any_coverage_pct, 1),
        "avg_sats_in_footprint": round(result.avg_sats_in_footprint, 2),
        "peak_simultaneous":     result.peak_simultaneous,
        "longest_gap_min":       round(result.longest_gap_min, 1),
        "avg_pass_duration_min": round(result.avg_pass_duration_min, 1),
    }

    return {
        "observer":         {"lat": obs_lat, "lon": obs_lon},
        "window_start":     times[0].isoformat(),
        "window_end":       times[-1].isoformat(),
        "timestep_sec":     int(step_s),
        "n_steps":          n_times,
        "summary":          summary,
        "handoff_schedule": schedule,
        "timesteps":        timesteps,
        "trajectories":     trajectories,
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

# Sentinel that separates the embedded-data script tag in viz.html.
# Must match exactly what is in viz.html.
_EMBED_SENTINEL = "window.VIZ_DATA = null;"


def write_viz_standalone(html_template: str, data: dict, out_path: str) -> None:
    """Inject data into viz.html and write a zero-server standalone file.

    Replaces the ``window.VIZ_DATA = null;`` sentinel in the template with
    the actual payload so the result can be opened directly via file://.
    """
    payload = json.dumps(data, separators=(",", ":"))
    injected = html_template.replace(
        _EMBED_SENTINEL,
        f"window.VIZ_DATA = {payload};",
        1,  # replace only the first occurrence
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(injected)


def write_viz_json(path: str, data: dict) -> None:
    """Write viz_data.json and a self-contained viz_out.html, then print
    instructions for both usage modes."""
    # ── viz_data.json (for fetch-based / server workflow) ─────────────────
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    json_kb = Path(path).stat().st_size // 1024

    # ── viz_out.html (self-contained, no server needed) ───────────────────
    here         = Path(__file__).parent
    template_path = here / "viz.html"
    out_path      = here / "viz_out.html"

    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
        if _EMBED_SENTINEL in template:
            write_viz_standalone(template, data, str(out_path))
            html_kb   = out_path.stat().st_size // 1024
            html_line = f"  Standalone: open {out_path} directly in any browser ({html_kb} KB)"
        else:
            html_line = f"  (viz.html sentinel not found — edit manually)"
    else:
        html_line = f"  (viz.html not found — cannot generate standalone)"

    parent = str(here.resolve())
    print(
        f"\n  {html_line}\n"
        f"\n"
        f"  Server mode: cd {parent!r} && python3 -m http.server 8000\n"
        f"               then http://localhost:8000/viz.html  ({json_kb} KB JSON)",
        file=sys.stderr,
    )
