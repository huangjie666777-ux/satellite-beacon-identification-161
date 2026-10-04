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


class Position(BaseModel):
    az_deg: float = Field(allow_inf_nan=False)
    el_deg: float = Field(allow_inf_nan=False)


class TrackPlanRequest(BaseModel):
    forecast: ForecastRequest
    interval_index: int = Field(ge=0)
    az_min_deg: float = Field(allow_inf_nan=False)
    az_max_deg: float = Field(allow_inf_nan=False)
    el_min_deg: float = Field(ge=0.0, le=90.0, allow_inf_nan=False)
    el_max_deg: float = Field(ge=0.0, le=90.0, allow_inf_nan=False)
    max_az_rate_dps: float = Field(gt=0.0, allow_inf_nan=False)
    max_el_rate_dps: float = Field(gt=0.0, allow_inf_nan=False)
    current_position: Position
    home_position: Position
    preset_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    homing_seconds: float = Field(gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _check(self):
        span = self.az_max_deg - self.az_min_deg
        if span <= 0.0:
            raise ValueError("az_max_deg must exceed az_min_deg")
        if span > 720.0:
            raise ValueError("azimuth span must not exceed 720 deg")
        if self.el_max_deg <= self.el_min_deg:
            raise ValueError("el_max_deg must exceed el_min_deg")
        for name, pos in (("current_position", self.current_position),
                          ("home_position", self.home_position)):
            if not (self.az_min_deg <= pos.az_deg <= self.az_max_deg):
                raise ValueError(f"{name} azimuth outside mechanical limits")
            if not (self.el_min_deg <= pos.el_deg <= self.el_max_deg):
                raise ValueError(f"{name} elevation outside mechanical limits")
        return self


class MechTargetOut(BaseModel):
    t_rel_s: float
    az_deg: float
    el_deg: float


class TrackPlanResponse(BaseModel):
    satellite_id: str
    station_id: str
    interval_index: int
    interval_start: datetime
    interval_end: datetime
    preset_seconds: float
    homing_seconds: float
    current_position: Position
    home_position: Position
    targets: list[MechTargetOut]
    total_az_travel_deg: float
    notes: list[str]


class PlaybackRequest(BaseModel):
    plan: TrackPlanResponse
    host: str = "127.0.0.1"
    port: int = Field(default=4533, ge=1, le=65535)
    position_tolerance_deg: float = Field(default=2.0, gt=0.0,
                                          allow_inf_nan=False)
    response_timeout_s: float = Field(default=5.0, gt=0.0, allow_inf_nan=False)
