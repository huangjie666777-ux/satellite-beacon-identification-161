"""Build a reproducible beacon-identification recording (deterministic).

Synthesises a single stable unmodulated carrier that follows the predicted
Doppler shift of the ISS downlink in examples/beacon_request.json plus a
constant +250 Hz local-oscillator offset.  The second candidate ISS_ALT
shares the same TLE but a downlink 100 Hz higher, so its debiased residual
is nearly identical: with a separation of 0.5 Hz the identification ends
up ambiguous, demonstrating the tie-handling rule.

Run:  .venv/bin/python examples/make_beacon_recording.py
Out:  examples/iq/beacon.sigmf-meta / beacon.sigmf-data
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
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
DURATION_S = 12.0
TX_HZ = 145_800_000.0          # ISS downlink in beacon_request.json
CENTER_HZ = 145_798_500.0
LO_OFFSET_HZ = 250.0           # constant receiver oscillator offset
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
    start = datetime.fromisoformat(
        req["window"]["start"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(
        req["window"]["end"].replace("Z", "+00:00"))
    iv = find_passes(tle.satrec, site, start, end)[0]
    t0 = (iv.start + timedelta(seconds=2.0)).replace(microsecond=0)
    if t0 + timedelta(seconds=DURATION_S) > iv.end:
        raise SystemExit("interval too short for the beacon recording")

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
    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": SAMPLE_RATE,
            "core:version": "1.0.0",
            "core:description": "synthetic beacon carrier (ISS/BEIJING, "
                                "+250 Hz LO offset)",
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
    print(f"carrier follows ISS Doppler + {LO_OFFSET_HZ} Hz LO offset")
    print("identify (ambiguous with ISS_ALT at 0.5 Hz separation):")
    print("  curl -X POST localhost:8000/api/beacon/identify")
    print("    -F 'forecast=<examples/beacon_request.json'")
    print("    -F station_id=BEIJING -F lo_offset_limit_hz=500")
    print("    -F rms_limit_hz=1.0 -F separation_hz=0.5")
    print("    -F meta=@examples/iq/beacon.sigmf-meta")
    print("    -F data=@examples/iq/beacon.sigmf-data")


if __name__ == "__main__":
    main()
