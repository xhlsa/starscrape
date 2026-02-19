"""Starlink Pass Predictor — CLI entry point.

Usage examples
--------------
  python main.py --lat 37.7749 --lon -122.4194
  python main.py --lat 51.5074 --lon -0.1278 --hours 12 --format json
  python main.py --lat 40.7128 --lon -74.0060 --min-elevation 10 --verbose
  python main.py --lat 37.7749 --lon -122.4194 --all --verbose
"""

from __future__ import annotations

import argparse
import sys
import time as time_mod
from datetime import datetime, timedelta, timezone

import numpy as np

from tle_cache import get_tles
from propagator import geodetic_to_ecef, parse_tles, propagate_batch
from pass_finder import Pass, find_passes
from link_budget import classify_shell, estimate_link_likelihood
from analyzer import (
    analyze,
    format_handoff_schedule,
    format_summary,
    format_analysis_json,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Predict Starlink satellite passes over a ground location.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lat",  type=float, required=True,
                   help="Observer latitude, degrees North")
    p.add_argument("--lon",  type=float, required=True,
                   help="Observer longitude, degrees East")
    p.add_argument("--hours", type=float, default=24.0,
                   help="Prediction window length in hours")
    p.add_argument("--min-elevation", type=float, default=0.0, dest="min_el",
                   help="Minimum pass peak elevation to report, degrees")
    p.add_argument("--format", choices=["table", "json"], default="table",
                   help="Output format")
    p.add_argument("--all", action="store_true", dest="show_all",
                   help="Include 'unlikely' passes in analysis "
                        "(default: only likely/marginal)")
    p.add_argument("--verbose", action="store_true",
                   help="Also print the full per-pass table after the summary")
    p.add_argument("--bin-hours", type=float, default=1.0, dest="bin_hours",
                   help="Width of density histogram bins in hours")
    p.add_argument("--hysteresis", type=float, default=5.0,
                   help="Min elevation gain (°) required to trigger a handoff "
                        "(0 = show every minute-level transition)")
    p.add_argument("--include-dtc", action="store_true", dest="include_dtc",
                   help="Include Direct-to-Cell sats (different service, excluded by default)")
    p.add_argument("--refresh", action="store_true",
                   help="Force re-fetch of TLEs from CelesTrak (ignore cache)")
    p.add_argument("--chunk-size", type=int, default=500, dest="chunk_size",
                   help="Satellites per SGP4 batch (trade RAM vs speed)")
    return p


# ---------------------------------------------------------------------------
# Time-grid construction
# ---------------------------------------------------------------------------

def build_time_grid(
    start: datetime,
    hours: float,
    step_s: int = 60,
) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    """Build a uniform time grid and corresponding Julian date arrays.

    The J2000 epoch (JD 2451545.0) corresponds to 2000-01-01T12:00:00 UTC.

    Returns
    -------
    times    : list of datetime (length n_steps)
    jd_whole : ndarray (n_steps,) — integer (floor) Julian day
    jd_frac  : ndarray (n_steps,) — fractional Julian day
    """
    n_steps = int(hours * 3600.0 / step_s) + 1
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    times: list[datetime] = []
    jd_whole = np.empty(n_steps, dtype=np.float64)
    jd_frac  = np.empty(n_steps, dtype=np.float64)

    for i in range(n_steps):
        t = start + timedelta(seconds=i * step_s)
        times.append(t)
        jd = 2451545.0 + (t - j2000).total_seconds() / 86400.0
        jd_whole[i] = np.floor(jd)
        jd_frac[i]  = jd - jd_whole[i]

    return times, jd_whole, jd_frac


# ---------------------------------------------------------------------------
# Pass-table formatter (verbose mode)
# ---------------------------------------------------------------------------

def _fmt_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _format_pass_table(passes: list[Pass]) -> str:
    if not passes:
        return "No passes in the specified window."

    hdr = (
        f"{'#':<4} {'Satellite':<24} {'NORAD':>6}  "
        f"{'Rise (UTC)':<19}  {'Peak (UTC)':<19}  {'Set (UTC)':<19}  "
        f"{'PkEl':>5}  {'PkAz':>5}  {'Slant':>7}  {'Shell':<22}  "
        f"{'Link':<8}  Note"
    )
    sep = "─" * len(hdr)
    rows = [hdr, sep]

    for i, p in enumerate(passes, 1):
        warn  = "  ⚠ DEGRADED TLE" if p.degraded else ""
        shell = classify_shell(p.orbital_alt_km)
        rows.append(
            f"{i:<4} {p.sat_name:<24} {p.norad_id:>6}  "
            f"{_fmt_time(p.rise_time):<19}  "
            f"{_fmt_time(p.peak_time):<19}  "
            f"{_fmt_time(p.set_time):<19}  "
            f"{p.peak_el_deg:>4.1f}°  "
            f"{p.peak_az_deg:>4.1f}°  "
            f"{p.min_slant_km:>6.0f}km  "
            f"{shell:<22}  "
            f"{p.link_likelihood:<8}  "
            f"{p.link_note}{warn}"
        )

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

    # ── 1. TLEs ───────────────────────────────────────────────────────────
    _log("Fetching Starlink TLEs…")
    tles = get_tles(force_refresh=args.refresh)
    n_degraded = sum(1 for t in tles if t["degraded"])
    n_dtc = sum(1 for t in tles if t.get("is_dtc"))
    if not args.include_dtc:
        tles = [t for t in tles if not t.get("is_dtc")]
        _log(f"  {len(tles)} TLEs loaded ({n_degraded} degraded, {n_dtc} DTC excluded — use --include-dtc to include).")
    else:
        _log(f"  {len(tles)} TLEs loaded ({n_degraded} degraded, {n_dtc} DTC included).")

    # ── 2. Parse to Satrec objects ────────────────────────────────────────
    _log("Parsing TLEs…")
    satrecs, valid_tles = parse_tles(tles)
    _log(f"  {len(satrecs)} valid satellites.")

    # ── 3. Time grid ──────────────────────────────────────────────────────
    start_utc = datetime.now(timezone.utc).replace(microsecond=0)
    times, jd_whole, jd_frac = build_time_grid(start_utc, args.hours)
    _log(
        f"Time window: {_fmt_time(start_utc)} → "
        f"{_fmt_time(times[-1])} UTC  ({len(times)} steps @ 60 s)"
    )

    # ── 4. Observer ECEF ──────────────────────────────────────────────────
    obs_ecef = geodetic_to_ecef(args.lat, args.lon)

    # ── 5. Vectorised SGP4 propagation ────────────────────────────────────
    _log(f"Propagating {len(satrecs)} sats × {len(times)} steps…")
    t0 = time_mod.perf_counter()
    az_deg, el_deg, slant_km = propagate_batch(
        satrecs, jd_whole, jd_frac, obs_ecef,
        args.lat, args.lon,
        chunk_size=args.chunk_size,
    )
    elapsed = time_mod.perf_counter() - t0
    _log(f"  Done in {elapsed:.2f} s.")

    # ── 6. Pass detection ─────────────────────────────────────────────────
    _log("Detecting passes…")
    passes = find_passes(
        az_deg, el_deg, slant_km,
        valid_tles, times,
        min_elevation_deg=args.min_el,
    )

    # ── 7. Link budget annotation ─────────────────────────────────────────
    for p in passes:
        p.link_likelihood, p.link_note = estimate_link_likelihood(
            p.peak_el_deg, p.orbital_alt_km
        )

    # ── 8. Filter and sort ────────────────────────────────────────────────
    passes = [p for p in passes if p.peak_el_deg >= args.min_el]
    if args.show_all:
        passes.sort(key=lambda p: p.rise_time)
        marginal_passes: list = []
    else:
        marginal_passes = sorted(
            [p for p in passes if p.link_likelihood == "marginal"],
            key=lambda p: p.rise_time,
        )
        passes = sorted(
            [p for p in passes if p.link_likelihood == "likely"],
            key=lambda p: p.rise_time,
        )
    _log(f"  {len(passes)} likely passes found"
         + (f", {len(marginal_passes)} marginal (shown only if they fill a gap).\n"
            if marginal_passes else ".\n"))

    # ── 9. Coverage analysis ──────────────────────────────────────────────
    _log("Running coverage analysis…")
    result = analyze(passes, el_deg, az_deg, valid_tles, times,
                     bin_hours=args.bin_hours,
                     hysteresis_deg=args.hysteresis,
                     marginal_passes=marginal_passes)
    _log("")

    # ── 10. Output ────────────────────────────────────────────────────────
    if args.format == "json":
        print(format_analysis_json(result, passes, include_passes=args.verbose))
        return

    # Table mode — header
    dtc_note = f", {n_dtc} DTC excluded" if not args.include_dtc else f", {n_dtc} DTC included"
    print(
        f"Starlink Coverage  ({args.lat:.4f}°N, {args.lon:.4f}°E)\n"
        f"Window  : {_fmt_time(start_utc)} → {_fmt_time(times[-1])} UTC\n"
        f"TLEs    : {len(valid_tles)} sats ({n_degraded} degraded{dtc_note})  "
        f"| Filter : {'all' if args.show_all else 'likely/marginal'}\n"
    )

    # Handoff schedule (primary output)
    print("Handoff Schedule")
    print(format_handoff_schedule(result, args.lat, args.lon))

    # Summary + density
    print()
    print(format_summary(result, passes))

    # Optional full pass table
    if args.verbose:
        print()
        print("Full Pass List")
        print(_format_pass_table(passes))


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
