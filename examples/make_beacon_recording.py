"""Build a reproducible example beacon recording for /api/beacon/identify.

Uses examples/beacon_request.json (ISS + perturbed ISS-DRIFT TLE, BEIJING
station). A constant-amplitude unmodulated carrier is synthesised whose
instantaneous baseband frequency follows the predicted ISS shift
    shift(t) = (f_tx - f_center) - f_tx * range_rate(t) / c
plus a constant LO offset, with deterministic low-level noise. The
recording therefore matches the ISS candidate; ISS-DRIFT scores a
clearly larger residual RMS. The same recording also drives the
ambiguity example (examples/beacon_ambiguous_request.json), where SAT-A
and SAT-B share the ISS orbit and differ only by 30 Hz in downlink
frequency, which the constant LO bias absorbs.

Run:  .venv/bin/python examples/make_beacon_recording.py
Out:  examples/iq/beacon.sigmf-meta / .sigmf-data
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.doppler import shift_node_times, shift_nodes
from app.passes import HorizonMask, Site, find_passes
from app.tle import parse_tle

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "iq"

SAMPLE_RATE = 48_000.0
DURATION_S = 12.0                # accepted windows span ~12 s (>= 5 s)
TX_HZ = 145_800_000.0            # ISS downlink = transmit frequency
CENTER_HZ = 145_798_500.0        # recording centre: carrier near +1.5 kHz
LO_OFFSET_HZ = 180.0             # constant local-oscillator offset
AMPLITUDE = 0.5
NOISE_RMS = 1e-3
SEED = 20240102


def main() -> None:
    req = json.loads((ROOT / "examples" / "beacon_request.json").read_text())
    sat = next(s for s in req["satellites"] if s["id"] == "ISS")
    stn = next(s for s in req["stations"] if s["id"] == "BEIJING")
    tle = parse_tle(sat["tle_line1"], sat["tle_line2"])
    site = Site(station_id=stn["id"], lat_deg=stn["lat_deg"],
                lon_deg=stn["lon_deg"], alt_m=stn["alt_m"],
                mask=HorizonMask(stn["mask"]))
    start = datetime.fromisoformat(req["window"]["start"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(req["window"]["end"].replace("Z", "+00:00"))
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
    inst = np.interp(t, node_times, shifts) + LO_OFFSET_HZ
    phase = np.concatenate(([0.0], np.cumsum(0.5 * (inst[:-1] + inst[1:])
                                             / SAMPLE_RATE)))
    rng = np.random.default_rng(SEED)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    noise *= NOISE_RMS / np.sqrt(2.0)
    samples = (AMPLITUDE * np.exp(2j * np.pi * phase) + noise)
    samples = samples.astype("<c8")

    OUT_DIR.mkdir(exist_ok=True)
    # capture-level core:frequency (the global fallback is not needed)
    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": SAMPLE_RATE,
            "core:version": "1.0.0",
            "core:description": "synthetic unmodulated beacon carrier (ISS/BEIJING)",
        },
        "captures": [{
            "core:sample_start": 0,
            "core:datetime": t0.isoformat().replace("+00:00", "Z"),
            "core:frequency": CENTER_HZ,
        }],
        "annotations": [],
    }
    (OUT_DIR / "beacon.sigmf-meta").write_text(json.dumps(meta, indent=2))
    (OUT_DIR / "beacon.sigmf-data").write_bytes(samples.tobytes())
    print(f"wrote {n} samples at {SAMPLE_RATE} Hz, start {t0.isoformat()}")
    print(f"shift range: {shifts.min():.1f} .. {shifts.max():.1f} Hz "
          f"(carrier additionally offset by LO {LO_OFFSET_HZ} Hz)")
    print("identify with:")
    print("  curl -X POST localhost:8000/api/beacon/identify \\")
    print("    -F 'forecast=<examples/beacon_request.json' \\")
    print("    -F station_id=BEIJING \\")
    print("    -F bias_limit_hz=500 -F rms_limit_hz=50 -F separation_hz=2 \\")
    print("    -F meta=@examples/iq/beacon.sigmf-meta \\")
    print("    -F data=@examples/iq/beacon.sigmf-data")
    print("ambiguity demo: use examples/beacon_ambiguous_request.json and")
    print("  -F bias_limit_hz=500 -F rms_limit_hz=50 -F separation_hz=2")


if __name__ == "__main__":
    main()
