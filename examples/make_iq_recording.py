"""Build a reproducible example IQ recording with a Doppler-shifted carrier.

Uses examples/request.json (ISS TLE + BEIJING station), finds the first
visibility interval, then synthesises a constant-amplitude carrier whose
instantaneous baseband frequency follows the predicted shift
    shift(t) = (f_tx - f_center) - f_tx * range_rate(t) / c
plus a small fixed offset, with deterministic low-level noise. After the
/api/doppler/correct correction the carrier lands at the fixed offset.

Run:  .venv/bin/python examples/make_iq_recording.py
Out:  examples/iq/iss_beijing.sigmf-meta / .sigmf-data
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.doppler import shift_node_times, shift_nodes
from app.passes import HorizonMask, Site, find_passes
from app.tle import parse_tle

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "iq"

SAMPLE_RATE = 48_000.0
DURATION_S = 4.0
TX_HZ = 145_800_000.0          # matches downlink_frequency_hz in request.json
CENTER_HZ = 145_798_500.0      # recording centre: carrier near +1.5 kHz
CARRIER_OFFSET_HZ = 250.0      # extra fixed offset of the synthetic carrier
AMPLITUDE = 0.5
NOISE_RMS = 1e-3
SEED = 20240101


def main() -> None:
    req = json.loads((ROOT / "examples" / "request.json").read_text())
    sat = req["satellites"][0]
    stn = next(s for s in req["stations"] if s["id"] == "BEIJING")
    tle = parse_tle(sat["tle_line1"], sat["tle_line2"])
    site = Site(station_id=stn["id"], lat_deg=stn["lat_deg"],
                lon_deg=stn["lon_deg"], alt_m=stn["alt_m"],
                mask=HorizonMask(stn["mask"]))
    start = __import__("datetime").datetime.fromisoformat(
        req["window"]["start"].replace("Z", "+00:00"))
    end = __import__("datetime").datetime.fromisoformat(
        req["window"]["end"].replace("Z", "+00:00"))
    intervals = find_passes(tle.satrec, site, start, end)
    if not intervals:
        raise SystemExit("no visibility interval in the example window")
    iv = intervals[0]
    t0 = (iv.start + timedelta(seconds=2.0)).replace(microsecond=0)
    if t0 + timedelta(seconds=DURATION_S) > iv.end:
        raise SystemExit("interval too short for the example recording")

    node_times = shift_node_times(DURATION_S)
    shifts = shift_nodes(tle.satrec, site, t0, node_times, TX_HZ, CENTER_HZ)
    n = int(SAMPLE_RATE * DURATION_S)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    inst = np.interp(t, node_times, shifts) + CARRIER_OFFSET_HZ
    phase = np.concatenate(([0.0], np.cumsum(0.5 * (inst[:-1] + inst[1:])
                                             / SAMPLE_RATE)))
    rng = np.random.default_rng(SEED)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    noise *= NOISE_RMS / np.sqrt(2.0)
    samples = (AMPLITUDE * np.exp(2j * np.pi * phase) + noise)
    samples = samples.astype("<c8")

    OUT_DIR.mkdir(exist_ok=True)
    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": SAMPLE_RATE,
            "core:frequency": CENTER_HZ,
            "core:version": "1.0.0",
            "core:description": "synthetic Doppler-shifted carrier (ISS/BEIJING)",
        },
        "captures": [{
            "core:sample_start": 0,
            "core:datetime": t0.isoformat().replace("+00:00", "Z"),
        }],
        "annotations": [],
    }
    (OUT_DIR / "iss_beijing.sigmf-meta").write_text(
        json.dumps(meta, indent=2))
    (OUT_DIR / "iss_beijing.sigmf-data").write_bytes(samples.tobytes())
    print(f"wrote {n} samples at {SAMPLE_RATE} Hz, start {t0.isoformat()}")
    print(f"shift range: {shifts.min():.1f} .. {shifts.max():.1f} Hz "
          f"(carrier additionally offset by {CARRIER_OFFSET_HZ} Hz)")
    print("submit with:")
    print("  curl -X POST localhost:8000/api/doppler/correct \\")
    print("    -F 'forecast=<examples/request.json' \\")
    print("    -F satellite_id=ISS -F station_id=BEIJING \\")
    print(f"    -F tx_frequency_hz={TX_HZ} \\")
    print("    -F meta=@examples/iq/iss_beijing.sigmf-meta \\")
    print("    -F data=@examples/iq/iss_beijing.sigmf-data \\")
    print("    -o corrected.zip")


if __name__ == "__main__":
    main()
