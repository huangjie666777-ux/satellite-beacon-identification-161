import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.tle import TLEError, parse_tle
from app.passes import HorizonMask

L1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9009"
L2 = "2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.49560532    19"

client = TestClient(app)


def base_request():
    return json.loads(Path("examples/request.json").read_text())


def test_tle_ok():
    tle = parse_tle(L1, L2)
    assert tle.catalog_number == 25544
    assert tle.epoch.year == 2024


def test_tle_bad_checksum():
    with pytest.raises(TLEError):
        parse_tle(L1[:68] + "0", L2)


def test_tle_bad_width():
    with pytest.raises(TLEError):
        parse_tle(L1[:-2], L2)


def test_tle_mismatched_numbers():
    bad = "2 99999" + L2[7:]
    bad = bad[:68] + "x"
    total = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in bad[:68])
    bad = bad[:68] + str(total % 10)
    with pytest.raises(TLEError):
        parse_tle(L1, bad)


def test_forecast_ok():
    r = client.post("/api/passes", json=base_request())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["interval_count"] == len(data["intervals"]) > 0
    starts = [i["start"] for i in data["intervals"]]
    assert starts == sorted(starts)
    for iv in data["intervals"]:
        assert iv["duration_s"] > 0
        assert 0 < iv["max_elevation_deg"] <= 90


def test_mask_segments_pass():
    # BEIJING has a high mask; its intervals must be split/shortened vs clear
    r = client.post("/api/passes", json=base_request()).json()
    masked = [i for i in r["intervals"] if i["station_id"] == "BEIJING"]
    clear = [i for i in r["intervals"] if i["station_id"] == "KUNMING"]
    assert masked and clear
    assert sum(i["duration_s"] for i in masked) < sum(
        i["duration_s"] for i in clear)


def test_duplicate_satellite_id():
    req = base_request()
    req["satellites"].append(dict(req["satellites"][0]))
    r = client.post("/api/passes", json=req)
    assert r.status_code == 422


def test_invalid_lat_and_nan():
    req = base_request()
    req["stations"][0]["lat_deg"] = 91.0
    assert client.post("/api/passes", json=req).status_code == 422
    req = base_request()
    req["stations"][0]["lat_deg"] = "NaN"
    assert client.post("/api/passes", json=req).status_code == 422


def test_window_too_long():
    req = base_request()
    req["window"]["end"] = "2024-01-02T13:00:00Z"
    assert client.post("/api/passes", json=req).status_code == 422


def test_stale_tle_rejected():
    req = base_request()
    req["window"] = {"start": "2024-02-01T00:00:00Z",
                     "end": "2024-02-01T01:00:00Z"}
    r = client.post("/api/passes", json=req)
    assert r.status_code == 422
    assert "7 days" in r.json()["detail"]


def test_too_many_satellites():
    req = base_request()
    for i in range(4):
        sat = dict(req["satellites"][0])
        sat["id"] = f"SAT{i}"
        req["satellites"].append(sat)
    assert client.post("/api/passes", json=req).status_code == 422


def test_download_zip():
    import io
    import zipfile
    r = client.post("/api/passes/download", json=base_request())
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert "summary.json" in names
    csvs = [n for n in names if n.endswith(".csv")]
    assert csvs
    summary = json.loads(zf.read("summary.json"))
    assert summary["interval_count"] == len(csvs)
    header = zf.read(csvs[0]).decode().splitlines()[0]
    assert header.split(",") == [
        "time_utc", "azimuth_deg", "elevation_deg", "range_km",
        "range_rate_km_s", "doppler_shift_hz"]


def test_mask_wrap_interpolation():
    m = HorizonMask([(350.0, 10.0), (10.0, 20.0)])
    assert m.min_elevation(0.0) == pytest.approx(15.0)
    assert m.min_elevation(350.0) == pytest.approx(10.0)
    flat = HorizonMask([(45.0, 7.0)])
    assert flat.min_elevation(123.0) == 7.0
    empty = HorizonMask(None)
    assert empty.min_elevation(0.0) == 0.0

