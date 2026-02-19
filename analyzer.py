"""Coverage analysis and actionable summary for Starlink passes.

Takes the filtered pass list (likely/marginal) plus the raw elevation arrays
and produces:

  - Per-minute best-satellite tracking (using actual propagated elevation, not
    just pass peak).
  - Handoff schedule — contiguous slots where one satellite is the highest-
    elevation option.  Gaps (no beam coverage) are explicit entries.
  - Summary statistics — coverage percentage, gap analysis, simultaneous sat
    counts, average pass duration.
  - Time-binned density — passes per hour, average best elevation, coverage
    fraction per bin.

All analysis is done at the propagation grid resolution (60 s) so no
additional SGP4 calls are needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

import numpy as np

from link_budget import classify_shell
from propagator import orbital_altitude_km


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class HandoffEntry:
    """One contiguous slot where a single satellite is the best connection,
    or a coverage gap."""
    start_time: datetime
    end_time: datetime
    duration_min: float
    is_gap: bool
    # Fields below are empty-string / 0 for gaps
    sat_name: str = ""
    norad_id: int = 0
    peak_el_deg: float = 0.0
    orbital_alt_km: float = 0.0
    shell: str = ""
    link_likelihood: str = ""


@dataclass
class DensityBin:
    start: datetime
    end: datetime
    n_passes: int              # passes whose peak falls in this bin
    avg_best_el_deg: float     # mean of best-sat elevation over minutes in bin
    coverage_pct: float        # % of minutes with ≥1 sat in footprint


@dataclass
class AnalysisResult:
    # Per-timestep (parallel to `times` list)
    best_el_deg: np.ndarray      # NaN when no coverage
    best_az_deg: np.ndarray      # NaN when no coverage
    best_sat_idx: np.ndarray     # index into valid_tles; -1 = gap
    best_link: list[str]         # link_likelihood of best sat at each step

    n_in_footprint: np.ndarray   # count of 'likely' sats per step
    n_marginal: np.ndarray       # count of 'marginal' sats per step

    handoff_schedule: list[HandoffEntry]
    density_bins: list[DensityBin]

    # Summary stats
    window_min: int
    covered_min: int             # steps with ≥1 sat in footprint (likely)
    any_coverage_min: int        # steps with ≥1 sat (likely OR marginal)
    coverage_pct: float          # covered_min / window_min * 100
    any_coverage_pct: float
    avg_sats_in_footprint: float
    peak_simultaneous: int
    gaps: list[tuple[datetime, datetime]]
    longest_gap_min: float
    avg_pass_duration_min: float


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze(
    passes: list,                # list[pass_finder.Pass] — already filtered
    el_deg: np.ndarray,          # (n_valid_sats, n_times)
    az_deg: np.ndarray,          # (n_valid_sats, n_times)
    valid_tles: list[dict],      # matches axis-0 of el_deg
    times: list[datetime],
    bin_hours: float = 1.0,
    hysteresis_deg: float = 5.0,
) -> AnalysisResult:
    """Build complete coverage analysis from pre-propagated arrays.

    hysteresis_deg
        A handoff is only recorded when a candidate satellite exceeds the
        current best by at least this many degrees.  Prevents rapid churn
        when several sats hover at similar elevations.  Set to 0 to see
        every minute-by-minute transition.
    """
    n_times = len(times)
    step_s = (times[1] - times[0]).total_seconds() if n_times > 1 else 60.0
    start_time = times[0]

    # Map NORAD ID → index in valid_tles / el_deg
    norad_to_idx: dict[int, int] = {
        tle["norad_id"]: i for i, tle in enumerate(valid_tles)
    }

    # ── Per-timestep tracking arrays ─────────────────────────────────────
    best_el  = np.full(n_times, np.nan, dtype=np.float64)
    best_az  = np.full(n_times, np.nan, dtype=np.float64)
    best_sat_idx = np.full(n_times, -1, dtype=np.intp)
    best_link: list[str] = [""] * n_times
    n_footprint = np.zeros(n_times, dtype=np.int32)
    n_marg      = np.zeros(n_times, dtype=np.int32)

    for pass_ in passes:
        sat_idx = norad_to_idx.get(pass_.norad_id)
        if sat_idx is None:
            continue

        # Grid indices bracketing this pass (clipped to valid range)
        ri = max(0, int((pass_.rise_time - start_time).total_seconds() / step_s))
        si = min(n_times, int((pass_.set_time  - start_time).total_seconds() / step_s) + 1)
        if ri >= si:
            continue

        pass_el = el_deg[sat_idx, ri:si]   # (slice_len,)
        pass_az = az_deg[sat_idx, ri:si]

        active = ~np.isnan(pass_el) & (pass_el >= 0.0)

        # Footprint / marginal counts
        if pass_.link_likelihood == "likely":
            n_footprint[ri:si] += active.astype(np.int32)
        else:
            n_marg[ri:si] += active.astype(np.int32)

        # Update best-sat where this pass has higher elevation
        current_best = best_el[ri:si]
        update = active & (np.isnan(current_best) | (pass_el > current_best))
        upd_idxs = np.where(update)[0]          # local indices within slice
        global_idxs = upd_idxs + ri
        best_el[global_idxs]      = pass_el[upd_idxs]
        best_az[global_idxs]      = pass_az[upd_idxs]
        best_sat_idx[global_idxs] = sat_idx
        for gi in global_idxs:
            best_link[gi] = pass_.link_likelihood

    # ── Handoff schedule (pass-level granularity) ────────────────────────
    # Built directly from Pass objects so each slot represents a full pass
    # being the "primary" connection, not a per-minute snapshot.
    handoff_schedule = _build_pass_level_schedule(
        passes, times, step_s, hysteresis_deg,
    )

    # ── Density bins ─────────────────────────────────────────────────────
    density_bins = _build_density_bins(
        passes, best_el, n_footprint, n_marg, times, bin_hours, step_s,
    )

    # ── Summary stats ─────────────────────────────────────────────────────
    covered_min   = int(np.sum(n_footprint > 0))
    any_cov_min   = int(np.sum((n_footprint + n_marg) > 0))
    window_min    = n_times

    gaps = _find_gaps(n_footprint + n_marg, times, step_s)
    longest_gap   = max((e - s).total_seconds() / 60.0 for s, e in gaps) if gaps else 0.0

    pass_durations = [
        (p.set_time - p.rise_time).total_seconds() / 60.0
        for p in passes
    ]
    avg_dur = float(np.mean(pass_durations)) if pass_durations else 0.0

    return AnalysisResult(
        best_el_deg=best_el,
        best_az_deg=best_az,
        best_sat_idx=best_sat_idx,
        best_link=best_link,
        n_in_footprint=n_footprint,
        n_marginal=n_marg,
        handoff_schedule=handoff_schedule,
        density_bins=density_bins,
        window_min=window_min,
        covered_min=covered_min,
        any_coverage_min=any_cov_min,
        coverage_pct=covered_min / window_min * 100.0,
        any_coverage_pct=any_cov_min / window_min * 100.0,
        avg_sats_in_footprint=float(np.mean(n_footprint)),
        peak_simultaneous=int(np.max(n_footprint)) if n_times else 0,
        gaps=gaps,
        longest_gap_min=longest_gap,
        avg_pass_duration_min=avg_dur,
    )


# ---------------------------------------------------------------------------
# Handoff schedule builder (pass-level)
# ---------------------------------------------------------------------------

def _build_pass_level_schedule(
    passes: list,
    times: list[datetime],
    step_s: float,
    hysteresis_deg: float = 5.0,
) -> list[HandoffEntry]:
    """Build a channel-guide style handoff schedule at pass granularity.

    Each entry spans the portion of a pass during which it is the committed
    primary connection.  A handoff occurs only when:
      1. The committed pass has ended (set_time reached), OR
      2. A newly risen pass has peak_el > committed.peak_el + hysteresis_deg

    This avoids the per-minute churn that arises when dozens of Starlink sats
    are simultaneously visible at similar elevations.
    """
    import heapq

    if not passes or not times:
        return []

    window_start = times[0]
    window_end   = times[-1] + timedelta(seconds=step_s)

    # All time boundaries: window edges + every pass rise/set, clamped to window
    boundaries: set[datetime] = {window_start, window_end}
    for p in passes:
        if window_start <= p.rise_time <= window_end:
            boundaries.add(p.rise_time)
        if window_start <= p.set_time <= window_end:
            boundaries.add(p.set_time)
    event_times = sorted(boundaries)

    # Passes sorted by rise_time for the sweep-line
    sorted_passes = sorted(passes, key=lambda p: p.rise_time)
    p_idx = 0

    # Max-heap (negate peak_el) for tracking active passes
    heap: list[tuple[float, int, object]] = []   # (-peak_el, id, pass)

    raw_entries: list[HandoffEntry] = []
    committed = None   # currently committed Pass object

    for ei in range(len(event_times) - 1):
        t_start = event_times[ei]
        t_end   = event_times[ei + 1]
        if t_start >= t_end:
            continue

        # Enqueue newly risen passes
        while p_idx < len(sorted_passes) and sorted_passes[p_idx].rise_time <= t_start:
            p = sorted_passes[p_idx]
            heapq.heappush(heap, (-p.peak_el_deg, id(p), p))
            p_idx += 1

        # Lazy-evict expired passes (set_time <= t_start)
        while heap and heap[0][2].set_time <= t_start:
            heapq.heappop(heap)

        if not heap:
            # Genuine coverage gap
            raw_entries.append(HandoffEntry(
                start_time=t_start,
                end_time=t_end,
                duration_min=(t_end - t_start).total_seconds() / 60.0,
                is_gap=True,
            ))
            committed = None
            continue

        # Best available pass by peak elevation (top of heap after eviction)
        while heap and heap[0][2].set_time <= t_start:
            heapq.heappop(heap)
        if not heap:
            committed = None
            continue
        best_pass = heap[0][2]

        # Decide whether to commit to best_pass or stay with current
        if committed is None or committed.set_time <= t_start:
            committed = best_pass
        elif best_pass.norad_id != committed.norad_id:
            if best_pass.peak_el_deg > committed.peak_el_deg + hysteresis_deg:
                committed = best_pass
            # else: stay with current committed satellite

        # Sanity-check committed is still active
        if committed.set_time <= t_start:
            committed = best_pass

        raw_entries.append(HandoffEntry(
            start_time=t_start,
            end_time=t_end,
            duration_min=(t_end - t_start).total_seconds() / 60.0,
            is_gap=False,
            sat_name=committed.sat_name,
            norad_id=committed.norad_id,
            peak_el_deg=committed.peak_el_deg,
            orbital_alt_km=committed.orbital_alt_km,
            shell=classify_shell(committed.orbital_alt_km),
            link_likelihood=committed.link_likelihood,
        ))

    return _merge_consecutive(raw_entries)


def _merge_consecutive(entries: list[HandoffEntry]) -> list[HandoffEntry]:
    """Merge adjacent entries for the same satellite (or adjacent gaps)."""
    if not entries:
        return []
    result = [entries[0]]
    for e in entries[1:]:
        prev = result[-1]
        same_sat = (not e.is_gap and not prev.is_gap
                    and e.norad_id == prev.norad_id)
        both_gap = e.is_gap and prev.is_gap
        if same_sat or both_gap:
            result[-1] = HandoffEntry(
                start_time=prev.start_time,
                end_time=e.end_time,
                duration_min=(e.end_time - prev.start_time).total_seconds() / 60.0,
                is_gap=prev.is_gap,
                sat_name=prev.sat_name,
                norad_id=prev.norad_id,
                peak_el_deg=max(prev.peak_el_deg, e.peak_el_deg),
                orbital_alt_km=prev.orbital_alt_km,
                shell=prev.shell,
                link_likelihood=prev.link_likelihood,
            )
        else:
            result.append(e)
    return result


# ---------------------------------------------------------------------------
# Density bins
# ---------------------------------------------------------------------------

def _build_density_bins(
    passes: list,
    best_el: np.ndarray,
    n_footprint: np.ndarray,
    n_marginal: np.ndarray,
    times: list[datetime],
    bin_hours: float,
    step_s: float,
) -> list[DensityBin]:
    """Bin passes and per-minute stats into fixed-width time intervals."""
    if not times:
        return []

    bin_s   = bin_hours * 3600.0
    start_t = times[0]
    end_t   = times[-1] + timedelta(seconds=step_s)
    total_s = (end_t - start_t).total_seconds()
    n_bins  = max(1, int(np.ceil(total_s / bin_s)))

    bins: list[DensityBin] = []
    for b in range(n_bins):
        bin_start = start_t + timedelta(seconds=b * bin_s)
        bin_end   = start_t + timedelta(seconds=(b + 1) * bin_s)

        # Timestep indices in this bin
        lo = max(0, int((bin_start - start_t).total_seconds() / step_s))
        hi = min(len(times), int((bin_end   - start_t).total_seconds() / step_s))

        # Passes whose peak falls in this bin
        n_passes = sum(
            1 for p in passes
            if bin_start <= p.peak_time < bin_end
        )

        # Average best elevation over minutes in this bin (ignore NaN gaps)
        slice_el = best_el[lo:hi]
        valid_el = slice_el[~np.isnan(slice_el)]
        avg_el   = float(np.mean(valid_el)) if len(valid_el) else 0.0

        # Coverage fraction
        n_covered = int(np.sum(n_footprint[lo:hi] > 0))
        n_any     = int(np.sum((n_footprint[lo:hi] + n_marginal[lo:hi]) > 0))
        cov_pct   = n_any / max(1, hi - lo) * 100.0

        bins.append(DensityBin(
            start=bin_start,
            end=bin_end,
            n_passes=n_passes,
            avg_best_el_deg=avg_el,
            coverage_pct=cov_pct,
        ))

    return bins


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_gaps(
    combined: np.ndarray,   # n_footprint + n_marginal per step
    times: list[datetime],
    step_s: float,
) -> list[tuple[datetime, datetime]]:
    """Return list of (gap_start, gap_end) datetime pairs."""
    gaps = []
    in_gap = False
    gap_start: datetime | None = None
    n = len(times)
    for i in range(n):
        if combined[i] == 0:
            if not in_gap:
                gap_start = times[i]
                in_gap = True
        else:
            if in_gap:
                gaps.append((gap_start, times[i]))  # type: ignore[arg-type]
                in_gap = False
    if in_gap and gap_start is not None:
        gaps.append((gap_start, times[-1] + timedelta(seconds=step_s)))
    return gaps


def _mode_string(values: list[str]) -> str:
    """Return most-frequent non-empty string from a list."""
    counts: dict[str, int] = {}
    for v in values:
        if v:
            counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get, default="")  # type: ignore[arg-type]


def _hm(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def format_handoff_schedule(result: AnalysisResult, lat: float, lon: float) -> str:
    """Render the handoff schedule as a human-readable table."""
    lines: list[str] = []

    hdr = (
        f"  {'Start (UTC)':<17}  {'End':<8}  {'Dur':>5}  "
        f"{'Satellite':<24}  {'NORAD':>6}  {'PkEl':>5}  "
        f"{'Shell':<22}  Link"
    )
    sep = "─" * len(hdr)
    lines.append(sep)
    lines.append(hdr)
    lines.append(sep)

    prev_date = None
    for i, e in enumerate(result.handoff_schedule, 1):
        date_label = e.start_time.strftime("%Y-%m-%d")
        date_tag   = f"[{date_label}]" if date_label != prev_date else ""
        prev_date  = date_label

        start_str = f"{_hm(e.start_time)} {date_tag:<12}".rstrip()
        end_str   = _hm(e.end_time)
        dur_str   = f"{e.duration_min:.0f}min"

        if e.is_gap:
            lines.append(
                f"  {start_str:<17}  {end_str:<8}  {dur_str:>5}  "
                f"{'·── GAP ──·':<24}  {'':>6}  {'':>5}  "
                f"{'':22}"
            )
        else:
            lines.append(
                f"  {start_str:<17}  {end_str:<8}  {dur_str:>5}  "
                f"{e.sat_name:<24}  {e.norad_id:>6}  "
                f"{e.peak_el_deg:>4.0f}°  "
                f"{e.shell:<22}  {e.link_likelihood}"
            )
    lines.append(sep)
    return "\n".join(lines)


def format_summary(result: AnalysisResult, passes: list) -> str:
    """Render the summary statistics block."""
    w = result.window_min

    def _dur(minutes: float) -> str:
        h, m = divmod(int(minutes), 60)
        return f"{h}h {m:02d}min" if h else f"{m}min"

    lines: list[str] = ["Coverage Summary", "────────────────"]
    lines.append(f"  Window              {_dur(w)}")
    lines.append(
        f"  In-beam (likely)    {_dur(result.covered_min)}  "
        f"({result.coverage_pct:.1f}%)"
    )
    lines.append(
        f"  Covered (incl. marginal)  {_dur(result.any_coverage_min)}  "
        f"({result.any_coverage_pct:.1f}%)"
    )
    lines.append(
        f"  No coverage         {_dur(w - result.any_coverage_min)}  "
        f"({100 - result.any_coverage_pct:.1f}%)"
    )
    lines.append("")
    lines.append(
        f"  Sats in beam        avg {result.avg_sats_in_footprint:.1f}  "
        f"(peak {result.peak_simultaneous})"
    )
    lines.append(f"  Avg pass duration   {result.avg_pass_duration_min:.1f} min")

    if result.gaps:
        longest_gap = max(result.gaps, key=lambda g: (g[1] - g[0]).total_seconds())
        gap_dur = (longest_gap[1] - longest_gap[0]).total_seconds() / 60.0
        lines.append(
            f"  Longest gap         {gap_dur:.0f} min  "
            f"(at {_fmt(longest_gap[0])} UTC)"
        )
    else:
        lines.append("  Longest gap         none")

    lines.append("")
    lines.append("Time-binned Density")
    lines.append("────────────────────")
    lines.append(
        f"  {'Period':<33}  {'Passes':>6}  {'Avg el':>6}  Coverage"
    )
    lines.append(f"  {'─'*33}  {'─'*6}  {'─'*6}  ────────")
    for b in result.density_bins:
        period = f"{_hm(b.start)}–{_hm(b.end)} {b.start.strftime('%d %b')}"
        el_str = f"{b.avg_best_el_deg:.0f}°" if b.avg_best_el_deg else " —"
        lines.append(
            f"  {period:<33}  {b.n_passes:>6}  {el_str:>6}  "
            f"{b.coverage_pct:.0f}%"
        )

    return "\n".join(lines)


def format_analysis_json(
    result: AnalysisResult,
    passes: list,
    include_passes: bool = False,
) -> str:
    """Serialise analysis result to JSON."""
    from link_budget import classify_shell

    schedule = []
    for e in result.handoff_schedule:
        rec: dict = {
            "start": e.start_time.isoformat(),
            "end":   e.end_time.isoformat(),
            "duration_min": round(e.duration_min, 1),
            "is_gap": e.is_gap,
        }
        if not e.is_gap:
            rec.update({
                "satellite": e.sat_name,
                "norad_id": e.norad_id,
                "peak_el_deg": round(e.peak_el_deg, 1),
                "orbital_alt_km": round(e.orbital_alt_km, 1),
                "shell": e.shell,
                "link_likelihood": e.link_likelihood,
            })
        schedule.append(rec)

    summary = {
        "window_min": result.window_min,
        "covered_min": result.covered_min,
        "coverage_pct": round(result.coverage_pct, 1),
        "any_coverage_min": result.any_coverage_min,
        "any_coverage_pct": round(result.any_coverage_pct, 1),
        "avg_sats_in_footprint": round(result.avg_sats_in_footprint, 2),
        "peak_simultaneous_sats": result.peak_simultaneous,
        "longest_gap_min": round(result.longest_gap_min, 1),
        "avg_pass_duration_min": round(result.avg_pass_duration_min, 1),
        "gaps": [
            {"start": s.isoformat(), "end": e.isoformat()}
            for s, e in result.gaps
        ],
    }

    density = [
        {
            "start": b.start.isoformat(),
            "end":   b.end.isoformat(),
            "n_passes": b.n_passes,
            "avg_best_el_deg": round(b.avg_best_el_deg, 1),
            "coverage_pct": round(b.coverage_pct, 1),
        }
        for b in result.density_bins
    ]

    payload: dict = {
        "summary": summary,
        "handoff_schedule": schedule,
        "density_bins": density,
    }

    if include_passes:
        payload["passes"] = [
            {
                "satellite": p.sat_name,
                "norad_id": p.norad_id,
                "rise_time":  p.rise_time.isoformat(),
                "peak_time":  p.peak_time.isoformat(),
                "set_time":   p.set_time.isoformat(),
                "peak_elevation_deg": round(p.peak_el_deg, 2),
                "peak_azimuth_deg":   round(p.peak_az_deg, 2),
                "min_slant_range_km": round(p.min_slant_km, 1),
                "orbital_altitude_km": round(p.orbital_alt_km, 1),
                "shell": classify_shell(p.orbital_alt_km),
                "epoch_age_days": round(p.epoch_age_days, 3),
                "degraded_tle": p.degraded,
                "link_likelihood": p.link_likelihood,
                "link_note": p.link_note,
            }
            for p in passes
        ]

    return json.dumps(payload, indent=2, ensure_ascii=False)
