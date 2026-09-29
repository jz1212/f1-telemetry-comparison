# download_telemetry.py
# Requires: pip install fastf1 pandas numpy
#
# Writes one CSV per driver into ./csv/ (e.g. csv/RUSSELL.csv) with columns:
#   lap,distance_m,speed_kmh
#
# Distance alignment
# ------------------
# FastF1 has no true "track position" channel. Its Distance is speed integrated
# over time, so every lap ends at a slightly different total (e.g. 5 981 m on
# one lap, 6 027 m on the next). Plotted as-is, the same x value lands on a
# different physical point of the track for each lap and each driver, and the
# error grows towards the end of the lap (braking points visibly shift).
#
# To make x = the same point on track for everyone, each lap is:
#   1. taken from raw car data (real speed samples, not the merged/interpolated
#      telemetry), sliced exactly at the timing line on both ends,
#   2. integrated to distance starting at 0 m on the line,
#   3. scaled so the lap ends exactly at TRACK_LENGTH.
# Lap 1 (standing start from the grid, not the line) and pit in/out laps
# (different path through the pit lane) are skipped by default because they
# can't be mapped to the racing line this way.

import re
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

YEAR = 2026
EVENT = "Azerbaijan Grand Prix"
SESSION = "R"

TRACK_LENGTH = 6003.0      # official Baku circuit length, metres
SKIP_FIRST_LAP = True      # standing start begins behind the line
SKIP_PIT_LAPS = True       # in/out laps run through the pit lane
MIN_SAMPLES = 50           # ignore laps with broken/missing car data
LENGTH_TOLERANCE = 0.08    # reject laps whose integrated length is off by >8%

OUT_DIR = Path("csv")
CACHE_DIR = Path("fastf1_cache")


def safe_name(text: str) -> str:
    text = re.sub(r"[^\w\-]+", "_", str(text).strip(), flags=re.UNICODE)
    return text.strip("_").upper() or "UNKNOWN"


def lap_trace(lap) -> pd.DataFrame | None:
    """Return distance-normalised speed samples for one lap, or None."""
    if pd.isna(lap["LapStartTime"]) or pd.isna(lap["Time"]):
        return None

    try:
        car = lap.get_car_data(interpolate_edges=True)
    except Exception:
        return None
    if car is None or len(car) < MIN_SAMPLES:
        return None

    car = car[["Time", "Speed"]].dropna().sort_values("Time")
    dt = car["Time"].diff().dt.total_seconds().fillna(0.0).to_numpy()
    speed = car["Speed"].to_numpy(dtype=float)

    # Trapezoidal integration: metres = mean speed (m/s) * dt
    ds = np.empty_like(speed)
    ds[0] = 0.0
    ds[1:] = (speed[1:] + speed[:-1]) / 2.0 / 3.6 * dt[1:]
    dist = np.cumsum(ds)

    total = dist[-1]
    if not (TRACK_LENGTH * (1 - LENGTH_TOLERANCE) < total < TRACK_LENGTH * (1 + LENGTH_TOLERANCE)):
        return None  # incomplete lap, red flag, retirement, etc.

    dist = dist * (TRACK_LENGTH / total)

    return pd.DataFrame({
        "lap": int(lap["LapNumber"]),
        "distance_m": np.round(dist, 2),
        "speed_kmh": np.round(speed, 1),
    })


def main():
    CACHE_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    session = fastf1.get_session(YEAR, EVENT, SESSION)
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    used_names = set()
    for drv in session.drivers:
        info = session.get_driver(drv)
        abbr = info.get("Abbreviation") or str(drv)
        name = safe_name(info.get("LastName") or abbr)
        if name in used_names:              # e.g. two drivers share a surname
            name = f"{name}_{safe_name(abbr)}"
        used_names.add(name)

        laps = session.laps.pick_drivers(drv)
        if SKIP_PIT_LAPS:
            laps = laps.pick_wo_box()

        frames, skipped = [], 0
        for _, lap in laps.iterlaps():
            if pd.isna(lap["LapNumber"]):
                continue
            if SKIP_FIRST_LAP and int(lap["LapNumber"]) == 1:
                continue
            tr = lap_trace(lap)
            if tr is None:
                skipped += 1
            else:
                frames.append(tr)

        if not frames:
            print(f"{name:>14}: no usable laps, skipped")
            continue

        out = pd.concat(frames, ignore_index=True)
        path = OUT_DIR / f"{name}.csv"
        out.to_csv(path, index=False)
        print(f"{name:>14}: {len(frames)} laps, {len(out):,} samples"
              f"{f' ({skipped} laps rejected)' if skipped else ''} -> {path}")


if __name__ == "__main__":
    main()
