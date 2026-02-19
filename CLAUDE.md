# starscrape — project notes for Claude

## Architecture

| File | Role |
|---|---|
| `tle_cache.py` | Fetch + cache TLEs from CelesTrak (2-hr TTL, JSON). Recomputes epoch age on load. |
| `propagator.py` | Vectorized SGP4 via `SatrecArray.sgp4` (1-D jd/fr arrays, C extension broadcasts). WGS-84 ECEF → ENU → az/el/slant. |
| `pass_finder.py` | 60-s coarse scan, linear interpolation for rise/set, 3-point parabolic for peak. |
| `link_budget.py` | Nadir-angle geometry → `likely` / `marginal` / `unlikely`. |
| `analyzer.py` | Per-timestep coverage counting, pass-level sweep-line handoff schedule, density bins. |
| `main.py` | argparse CLI; orchestrates pipeline; table + JSON output. |

## Key geometry facts

- `SatrecArray.sgp4(jd, fr)` takes **1-D** arrays of length n_times; the C extension broadcasts across all satellites internally. Do NOT pass 2-D `(n_sats, n_times)` matrices — that raises a shape error.
- Nadir angle formula: `sin(θ_nadir) = Rₑ·cos(ε) / (Rₑ + h)` where ε = elevation, h = altitude.
- At 550 km altitude, nadir = 25° requires elevation ≥ **62.7°** → footprint radius ≈ 259 km.
- Geometric expectation: ~3.9 simultaneous sats in a 25° nadir footprint with 9 500-sat constellation (verified by fix).

## Coverage counting — critical bug history

**Bug (fixed dc27f9d):** `n_footprint` previously counted every above-horizon minute of a "likely" pass (classified by peak elevation) rather than only the minutes where `nadir_ts < BEAM_HALF_ANGLE_DEG` at that instant. A 12-min visible pass with a 75° peak has only ~2 in-footprint minutes; the old code counted 12 → ~14× overcount (53.6 avg simultaneous → 4.8 after fix).

**Fix location:** `analyzer.py`, the `for pass_ in passes` loop. Per-timestep nadir is now computed from `el_deg[sat_idx, ri:si]` and `pass_.orbital_alt_km` before incrementing `n_footprint` / `n_marg`.

If the simultaneous-sat count ever reads above ~10 for a mid-latitude observer with the default 25° nadir setting, the counting is broken again.

## Stress-test baseline (25° nadir, default settings, 6-hr window)

Run with `--hours 6` and check "Sats in beam" and "In-beam" lines.

| Location | Coords | avg in-beam | peak | In-beam % | Longest gap |
|---|---|---|---|---|---|
| San Francisco | 37.77, -122.42 | ~4.8 | ~9 | 100% | none |
| Arizona | 37.7, -110.0 | ~4.9 | ~9 | 100% | none |
| Mid-Pacific (equator) | 0.0, -170.0 | ~2.6 | ~9 | 96% | none |
| Antarctica | -75.0, 0.0 | ~0.7 | ~4 | 47% | ~62 min |
| Arctic | 85.0, 30.0 | 0.0 | 0 | 0% | entire window |

**Expected gradients:**
- Mid-latitude US: avg ~4–5, near-100% coverage.
- Equator: avg ~2–3 (fewer sats reach high elevation at equator vs. mid-lat).
- Antarctica -75°: sparse; only polar-shell sats visible; long gaps expected.
- 85°N: **zero in-beam is correct**. Starlink polar shells are ~97.6° inclination → ground track reaches ±82.4° lat. Observer at 85°N sees them at ≤60° elevation → nadir ≥ 27.5° — outside the 25° cone. `marginal` passes still appear; `likely` never does.

## Stress-test baseline (15° nadir, tighter beam)

| Location | avg in-beam | In-beam % | Longest gap |
|---|---|---|---|
| San Francisco | ~1.5 | 83% | ~1 min |
| Mid-Pacific | ~0.9 | 64% | ~11 min |
| Antarctica -75° | ~0.2 | 17% | ~85 min |
| Arctic 85°N | 0.0 | 0% | entire window |

## Performance targets

Full 24-hr sweep of ~9 500 sats should complete in **< 15 s** on commodity hardware.
Observed: ~5.4 s (Feb 2026 on WSL2/Linux, Intel Core).

## Default behaviour

- Default filter: **likely + marginal** passes only (`--all` to include unlikely).
- Default output: **handoff schedule + summary stats** (`--verbose` appends full pass table).
- Hysteresis default: **5°** — a handoff is only logged when a new sat exceeds current by ≥ 5° peak elevation.
- Handoff schedule is **pass-level** (one row per pass, not per minute) via sweep-line over sorted rise/set events.
