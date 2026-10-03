"""Build forecast results: JSON summary and per-interval tracking CSVs."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import timedelta

from .coords import SPEED_OF_LIGHT, teme_to_ecef, teme_vel_to_ecef
from .coords import ecef_to_enu_az_el_range
from .passes import PassInterval, Site, _propagate

CSV_HEADER = [
    "time_utc",
    "azimuth_deg",
    "elevation_deg",
    "range_km",
    "range_rate_km_s",
    "doppler_shift_hz",
]


def interval_csv_rows(satrec, site: Site, interval: PassInterval,
                      downlink_hz: float):
    """Per-second tracking rows; Doppler uses ECEF range rate."""
    t = interval.start.replace(microsecond=0)
    if t < interval.start:
        t += timedelta(seconds=1)
    while t <= interval.end:
        r_teme, v_teme = _propagate(satrec, t)
        sat_ecef = teme_to_ecef(r_teme, t)
        vel_ecef = teme_vel_to_ecef(r_teme, v_teme, t)
        az, el, rng = ecef_to_enu_az_el_range(
            sat_ecef, site.ecef, site.lat_deg, site.lon_deg)
        dx = sat_ecef[0] - site.ecef[0]
        dy = sat_ecef[1] - site.ecef[1]
        dz = sat_ecef[2] - site.ecef[2]
        rdot = (dx * vel_ecef[0] + dy * vel_ecef[1] + dz * vel_ecef[2]) / rng
        # range increasing -> negative Doppler shift at the receiver
        doppler_hz = -downlink_hz * (rdot * 1000.0) / SPEED_OF_LIGHT
        yield [
            t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            f"{az:.3f}", f"{el:.3f}", f"{rng:.3f}",
            f"{rdot:.6f}", f"{doppler_hz:.1f}",
        ]
        t += timedelta(seconds=1)


def interval_to_dict(sat_id: str, stn_id: str, iv: PassInterval) -> dict:
    return {
        "satellite_id": sat_id,
        "station_id": stn_id,
        "start": iv.start.isoformat().replace("+00:00", "Z"),
        "end": iv.end.isoformat().replace("+00:00", "Z"),
        "duration_s": round(iv.duration_s, 1),
        "truncated_at_start": iv.truncated_at_start,
        "truncated_at_end": iv.truncated_at_end,
        "max_elevation_deg": round(iv.max_el_deg, 3),
        "max_elevation_time": iv.max_el_time.isoformat().replace("+00:00", "Z"),
    }


def build_zip(summary: dict, csv_files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json",
                    json.dumps(summary, ensure_ascii=False, indent=2))
        for name, content in sorted(csv_files.items()):
            zf.writestr(f"csv/{name}", content)
    return buf.getvalue()


def render_csv(rows) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(CSV_HEADER)
    w.writerows(rows)
    return out.getvalue()

