# starscrape

Starlink pass predictor and coverage analyzer for a ground observer.
Given a latitude/longitude, it fetches live TLEs, runs vectorized SGP4 propagation across the full constellation, and outputs a handoff schedule showing which satellite to track minute-by-minute.

## Features

- Fetches Starlink TLEs from CelesTrak with a 2-hour local cache
- Vectorized SGP4 via `SatrecArray` (C extension) — ~9,500 sats × 1,441 steps in ~5 s
- WGS-84 ECEF → topocentric azimuth / elevation / slant range
- Nadir-angle link budget: classifies each pass as `likely`, `marginal`, or `unlikely`
- Pass-level handoff schedule with configurable hysteresis
- Coverage statistics: in-beam %, longest gap, simultaneous sat count, density histogram
- Table and JSON output modes

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and the packages in `requirements.txt` (`sgp4`, `numpy`, `requests`).

## Usage

```bash
# Basic — San Francisco, 24-hour window
python main.py --lat 37.7749 --lon -122.4194

# London, 12-hour window, JSON output
python main.py --lat 51.5074 --lon -0.1278 --hours 12 --format json

# Include all passes (even unlikely), show full pass table
python main.py --lat 40.7128 --lon -74.0060 --all --verbose

# Tighter minimum elevation filter, force TLE refresh
python main.py --lat 37.7749 --lon -122.4194 --min-elevation 20 --refresh
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--lat` | required | Observer latitude, degrees North |
| `--lon` | required | Observer longitude, degrees East |
| `--hours` | 24 | Prediction window length |
| `--min-elevation` | 0 | Minimum peak elevation to report (degrees) |
| `--format` | table | Output format: `table` or `json` |
| `--all` | off | Include `unlikely` passes (default: only `likely`/`marginal`) |
| `--verbose` | off | Append full per-pass table after the summary |
| `--bin-hours` | 1.0 | Width of density histogram bins |
| `--hysteresis` | 5.0 | Min elevation gain (°) required to trigger a handoff |
| `--refresh` | off | Force re-fetch of TLEs, ignoring cache |
| `--chunk-size` | 500 | Satellites per SGP4 batch (tune RAM vs speed) |

## Output

Default output is a **handoff schedule** — one row per recommended satellite, when to switch, and the expected peak elevation — followed by coverage summary statistics.

```
Starlink Coverage  (37.7749°N, -122.4194°E)
Window  : 2026-02-19 04:00:00 → 2026-02-20 04:00:00 UTC
TLEs    : 9548 sats (12 degraded)  | Filter : likely/marginal

Handoff Schedule
  #   Satellite                NORAD   From (UTC)            To (UTC)              PkEl   PkAz
  1   STARLINK-1234            45678   2026-02-19 04:02:00   2026-02-19 04:09:00   74.3°  212°
  2   STARLINK-5678            46789   2026-02-19 04:09:00   2026-02-19 04:17:00   81.1°  188°
  ...

Coverage Summary
  In-beam    : 100.0%   (1440 / 1440 min)
  Longest gap: 0 min
  Avg in-beam sats: 4.8   Peak: 9
  ...
```

With `--verbose`, a full per-pass table is appended showing rise/set/peak times, azimuth, slant range, shell, and link note for every detected pass.

## Link budget

Passes are classified using nadir-angle geometry:

- **likely** — nadir angle < 25° AND elevation ≥ 25° (well inside beam, usable signal)
- **marginal** — nadir < 25° but low elevation, or nadir 25–35° at useful elevation
- **unlikely** — nadir ≥ 35° or elevation too low to be useful

At 550 km altitude, a 25° nadir half-angle corresponds to a ~259 km footprint radius and requires the observer to be at ≥ 62.7° elevation from the satellite's perspective. The expected number of simultaneous in-beam satellites over a mid-latitude US location is ~4–5.

## File structure

| File | Role |
|---|---|
| `tle_cache.py` | CelesTrak fetch, JSON cache (2-hr TTL), epoch-age flagging |
| `propagator.py` | Vectorized SGP4, WGS-84 transforms, az/el/slant computation |
| `pass_finder.py` | Coarse 60-s scan, interpolated rise/set, parabolic peak refinement |
| `link_budget.py` | Nadir-angle link budget, shell classification |
| `analyzer.py` | Coverage counting, handoff schedule, density bins, summary stats |
| `main.py` | CLI entry point |
