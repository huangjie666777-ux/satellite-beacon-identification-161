"""TLE parsing and validation: width, checksum, catalog-number match, epoch."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sgp4.api import Satrec

TLE_LINE_WIDTH = 69


class TLEError(ValueError):
    """Raised when a TLE fails validation."""


def _checksum_ok(line: str) -> bool:
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10 == int(line[68])


@dataclass
class ParsedTLE:
    line1: str
    line2: str
    satrec: Satrec
    catalog_number: int
    epoch: datetime  # UTC


def parse_tle(line1: str, line2: str) -> ParsedTLE:
    l1 = line1.rstrip("\r\n")
    l2 = line2.rstrip("\r\n")
    for tag, line in (("line1", l1), ("line2", l2)):
        if len(line) != TLE_LINE_WIDTH:
            raise TLEError(f"{tag} must be {TLE_LINE_WIDTH} columns, got {len(line)}")
        if not line[68].isdigit() or not _checksum_ok(line):
            raise TLEError(f"{tag} checksum mismatch")
    if l1[0] != "1" or l2[0] != "2":
        raise TLEError("lines must start with '1' and '2'")

    try:
        num1 = int(l1[2:7])
        num2 = int(l2[2:7])
    except ValueError as exc:
        raise TLEError("invalid catalog number columns") from exc
    if num1 != num2:
        raise TLEError(f"catalog numbers differ: {num1} != {num2}")

    try:
        year2 = int(l1[18:20])
        doy = float(l1[20:32])
    except ValueError as exc:
        raise TLEError("invalid epoch field") from exc
    year = 1900 + year2 if year2 >= 57 else 2000 + year2
    epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1.0)

    satrec = Satrec.twoline2rv(l1, l2)
    if satrec.error != 0:
        raise TLEError(f"SGP4 rejected TLE, error code {satrec.error}")
    return ParsedTLE(line1=l1, line2=l2, satrec=satrec,
                     catalog_number=num1, epoch=epoch)
