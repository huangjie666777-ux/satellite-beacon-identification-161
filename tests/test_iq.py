"""Tests for the IQ Doppler-correction endpoint and the playback
empty-target controller fix."""
import copy
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.doppler import shift_node_times, shift_nodes
from app.main import app, playback
from app.passes import HorizonMask, Site, find_passes
from app.sigmf import validate as sigmf_validate, SigmfError
from app.tle import parse_tle

client = TestClient(app)

FS = 48_000.0
DURATION_S = 2.0
TX_HZ = 145_800_000.0
CENTER_HZ = 145_798_500.0
OFFSET_HZ = 250.0


def base_request():
    return json.loads(Path("examples/request.json").read_text())


def _recording():
    """Deterministic in-memory recording matching the example generator."""
    req = base_request()
    sat = req["satellites"][0]
    stn = next(s for s in req["stations"] if s["id"] == "BEIJING")
    tle = parse_tle(sat["tle_line1"], sat["tle_line2"])
    site = Site(station_id="BEIJING", lat_deg=stn["lat_deg"],
                lon_deg=stn["lon_deg"], alt_m=stn["alt_m"],
                mask=HorizonMask(stn["mask"]))
    start = datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 4, 30, tzinfo=timezone.utc)
    iv = find_passes(tle.satrec, site, start, end)[0]
    t0 = (iv.start + timedelta(seconds=2.0)).replace(microsecond=0)
    node_times = shift_node_times(DURATION_S)
    shifts = shift_nodes(tle.satrec, site, t0, node_times, TX_HZ, CENTER_HZ)
    n = int(FS * DURATION_S)
    t = np.arange(n, dtype=np.float64) / FS
    inst = np.interp(t, node_times, shifts) + OFFSET_HZ
    phase = np.concatenate(([0.0], np.cumsum(0.5 * (inst[:-1] + inst[1:]) / FS)))
    rng = np.random.default_rng(7)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 1e-3
    samples = (0.5 * np.exp(2j * np.pi * phase) + noise).astype("<c8")
    meta = {
        "global": {"core:datatype": "cf32_le", "core:sample_rate": FS,
                   "core:frequency": CENTER_HZ, "core:version": "1.0.0"},
        "captures": [{"core:sample_start": 0,
                      "core:datetime": t0.isoformat().replace("+00:00", "Z")}],
        "annotations": [],
    }
    return meta, samples


def post(meta, data_bytes, tx=TX_HZ, sat="ISS", stn="BEIJING", req=None):
    files = {
        "forecast": (None, json.dumps(req or base_request())),
        "satellite_id": (None, sat),
        "station_id": (None, stn),
        "tx_frequency_hz": (None, str(tx)),
        "meta": ("rec.sigmf-meta", json.dumps(meta), "application/json"),
        "data": ("rec.sigmf-data", data_bytes, "application/octet-stream"),
    }
    return client.post("/api/doppler/correct", files=files)


def test_correct_happy_path():
    meta, samples = _recording()
    r = post(meta, samples.tobytes())
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert sorted(zf.namelist()) == [
        "corrected.sigmf-data", "corrected.sigmf-meta", "diagnostics.json"]
    out_meta = json.loads(zf.read("corrected.sigmf-meta"))
    # capture-level centre frequency moved to the transmit frequency,
    # the stale global value is dropped, UTC unchanged
    assert out_meta["captures"][0]["core:frequency"] == TX_HZ
    assert "core:frequency" not in out_meta["global"]
    assert out_meta["captures"][0]["core:datetime"] == \
        meta["captures"][0]["core:datetime"]
    assert out_meta["global"]["core:sample_rate"] == FS
    # original meta untouched
    assert meta["global"]["core:frequency"] == CENTER_HZ
    corrected = np.frombuffer(zf.read("corrected.sigmf-data"), dtype="<c8")
    assert len(corrected) == len(samples)
    diag = json.loads(zf.read("diagnostics.json"))
    assert diag["sample_count"] == len(samples)
    assert diag["window"]["count"] == len(diag["windows"]) > 0
    for w in diag["windows"]:
        # carrier moved from the shifted band to the fixed offset bin
        assert abs(w["peak_freq_hz_after"] - OFFSET_HZ) < FS / 1024
        assert abs(w["peak_freq_hz_before"]) > 1000.0
        assert w["mean_power_after"] == pytest.approx(
            w["mean_power_before"], rel=1e-3)


def test_amplitude_and_count_preserved():
    meta, samples = _recording()
    r = post(meta, samples.tobytes())
    corrected = np.frombuffer(
        zipfile.ZipFile(io.BytesIO(r.content)).read("corrected.sigmf-data"),
        dtype="<c8")
    assert len(corrected) == len(samples)
    assert np.allclose(np.abs(corrected), np.abs(samples), atol=1e-5)


def test_short_recording_no_diagnostics():
    meta, samples = _recording()
    short = samples[:500]
    meta = copy.deepcopy(meta)
    r = post(meta, short.tobytes())
    assert r.status_code == 200, r.text
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    diag = json.loads(zf.read("diagnostics.json"))
    assert diag["windows"] == []
    assert diag["window"]["count"] == 0


def test_reject_empty_and_truncated():
    meta, samples = _recording()
    assert post(meta, b"").status_code == 422
    assert post(meta, samples.tobytes()[:-3]).status_code == 422  # truncated


def test_reject_non_finite_samples():
    meta, samples = _recording()
    bad = samples.copy()
    bad[100] = complex(float("nan"), 0.0)
    assert post(meta, bad.tobytes()).status_code == 422
    bad = samples.copy()
    bad[200] = complex(0.0, float("inf"))
    assert post(meta, bad.tobytes()).status_code == 422


def test_reject_bad_metadata():
    meta, samples = _recording()
    data = samples.tobytes()
    for mutate in (
        lambda m: m["global"].update(**{"core:datatype": "ci16_le"}),
        lambda m: m["global"].update(**{"core:sample_rate": 500.0}),
        lambda m: m["global"].update(**{"core:sample_rate": 500_000.0}),
        lambda m: m["global"].pop("core:frequency"),
        lambda m: m["captures"][0].update(**{"core:sample_start": 1}),
        lambda m: m["captures"].append(dict(m["captures"][0])),
        lambda m: m["captures"][0].pop("core:datetime"),
        lambda m: m["captures"][0].update(
            **{"core:datetime": "2024-01-01 02:29:43"}),
    ):
        m = copy.deepcopy(meta)
        mutate(m)
        assert post(m, data).status_code == 422, mutate
    assert post({"global": {}}, data).status_code == 422


def test_reject_too_long_recording():
    meta, samples = _recording()
    meta["global"]["core:sample_rate"] = 1_000.0
    n = 61_000  # 61 s at 1 kHz
    data = np.zeros(n, dtype="<c8").tobytes()
    r = post(meta, data)
    assert r.status_code == 422
    assert "60" in r.json()["detail"]


def test_reject_shift_beyond_nyquist():
    meta, samples = _recording()
    # centre far from the transmit frequency -> shift exceeds fs/2
    r = post(meta, samples.tobytes(), tx=145_900_000.0)
    assert r.status_code == 422
    assert "half the sample rate" in r.json()["detail"]


def test_reject_outside_visibility():
    meta, samples = _recording()
    meta["captures"][0]["core:datetime"] = "2024-01-01T02:00:00Z"  # no pass
    r = post(meta, samples.tobytes())
    assert r.status_code == 422
    assert "visibility" in r.json()["detail"]


def test_reject_unknown_ids_and_bad_tx():
    meta, samples = _recording()
    data = samples.tobytes()
    assert post(meta, data, sat="NOPE").status_code == 422
    assert post(meta, data, stn="NOPE").status_code == 422
    assert post(meta, data, tx=0.0).status_code == 422
    assert post(meta, data, tx=-5.0).status_code == 422
    assert post(meta, data, tx="NaN").status_code == 422
    assert post(meta, data, tx="abc").status_code == 422


def test_sigmf_validate_unit():
    with pytest.raises(SigmfError):
        sigmf_validate(b"not json", 8)
    with pytest.raises(SigmfError):
        sigmf_validate(b"", 8)


# ---- playback empty-target fix -------------------------------------------

def test_playback_empty_targets_rejected_and_not_running():
    plan = {
        "satellite_id": "ISS", "station_id": "BEIJING", "interval_index": 0,
        "interval_start": "2024-01-01T02:29:41Z",
        "interval_end": "2024-01-01T02:29:44Z",
        "preset_seconds": 1.0, "homing_seconds": 1.0,
        "current_position": {"az_deg": 10.0, "el_deg": 20.0},
        "home_position": {"az_deg": 10.0, "el_deg": 20.0},
        "targets": [],
        "total_az_travel_deg": 0.0, "notes": [],
    }
    r = client.post("/api/playback", json={"plan": plan, "port": 4599})
    assert r.status_code == 422
    st = playback.status()
    assert st["state"] != "running"
