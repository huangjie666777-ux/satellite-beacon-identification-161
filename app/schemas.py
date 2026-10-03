"""Request/response models with strict validation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_SATELLITES = 4
MAX_STATIONS = 4
MAX_WINDOW = timedelta(hours=24)
MAX_EPOCH_AGE = timedelta(days=7)


class Window(BaseModel):
    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must carry a UTC offset")
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _check(self):
        if self.end <= self.start:
            raise ValueError("window end must be after start")
        if self.end - self.start > MAX_WINDOW:
            raise ValueError("window must not exceed 24 hours")
        return self


class SatelliteIn(BaseModel):
    id: str = Field(min_length=1, max_length=32)
    tle_line1: str
    tle_line2: str
    downlink_frequency_hz: float = Field(gt=0, allow_inf_nan=False)


class StationIn(BaseModel):
    id: str = Field(min_length=1, max_length=32)
    lat_deg: float = Field(ge=-90.0, le=90.0, allow_inf_nan=False)
    lon_deg: float = Field(ge=-180.0, le=180.0, allow_inf_nan=False)
    alt_m: float = Field(ge=-500.0, le=10000.0, allow_inf_nan=False)
    mask: list[tuple[float, float]] | None = None

    @field_validator("mask")
    @classmethod
    def _check_mask(cls, v):
        if v is None:
            return v
        for az, el in v:
            if not (0.0 <= az < 360.0) or az != az or az in (float("inf"), float("-inf")):
                raise ValueError("mask azimuth must be finite and in [0, 360)")
            if not (-90.0 <= el <= 90.0):
                raise ValueError("mask elevation must be in [-90, 90]")
        return v


class ForecastRequest(BaseModel):
    window: Window
    satellites: list[SatelliteIn] = Field(min_length=1, max_length=MAX_SATELLITES)
    stations: list[StationIn] = Field(min_length=1, max_length=MAX_STATIONS)

    @model_validator(mode="after")
    def _unique_ids(self):
        for group, items in (("satellite", self.satellites),
                             ("station", self.stations)):
            ids = [x.id for x in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {group} id")
        return self


class IntervalOut(BaseModel):
    satellite_id: str
    station_id: str
    start: datetime
    end: datetime
    duration_s: float
    truncated_at_start: bool
    truncated_at_end: bool
    max_elevation_deg: float
    max_elevation_time: datetime


class ForecastResponse(BaseModel):
    window: Window
    interval_count: int
    intervals: list[IntervalOut]
    notes: list[str]
