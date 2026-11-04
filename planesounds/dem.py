"""Ground elevation models.

ADS-B altitude is above mean sea level, so the observer's ground elevation is
needed to get the true slant distance to an aircraft.

* ``ConstantDEM``  - one elevation everywhere (fine for flat/coastal areas).
* ``SRTMDEM``      - SRTM ``.hgt`` tiles (1" or 3"), read from a folder and
                     optionally auto-downloaded from the public AWS
                     "elevation-tiles-prod" Skadi bucket (no account needed).
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

DEM_DIR = Path.home() / ".planesounds" / "dem"
SKADI_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}/{name}.hgt.gz"


class ConstantDEM:
    def __init__(self, elev_m: float = 0.0):
        self.elev_m = float(elev_m)

    def __call__(self, lat, lon):
        return np.full(np.shape(lat), self.elev_m, dtype=float)

    def describe(self):
        return f"constant {self.elev_m:.0f} m"


def _tile_name(lat_i: int, lon_i: int) -> str:
    ns = "N" if lat_i >= 0 else "S"
    ew = "E" if lon_i >= 0 else "W"
    return f"{ns}{abs(lat_i):02d}{ew}{abs(lon_i):03d}"


class SRTMDEM:
    def __init__(self, folder: str | Path = DEM_DIR, download: bool = True, fallback: float = 0.0):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.download = download
        self.fallback = fallback
        self._tiles: dict[str, np.ndarray | None] = {}

    def describe(self):
        return f"SRTM tiles in {self.folder}" + (" (auto-download)" if self.download else "")

    def _load(self, lat_i, lon_i):
        name = _tile_name(lat_i, lon_i)
        if name in self._tiles:
            return self._tiles[name]
        arr = None
        for cand in (self.folder / f"{name}.hgt", self.folder / f"{name}.hgt.gz"):
            if cand.exists():
                raw = cand.read_bytes()
                arr = self._decode(raw)
                break
        if arr is None and self.download:
            try:
                import requests

                r = requests.get(SKADI_URL.format(ns=name[:3], name=name), timeout=60)
                if r.status_code == 200:
                    (self.folder / f"{name}.hgt.gz").write_bytes(r.content)
                    arr = self._decode(r.content)
            except Exception:
                arr = None
        self._tiles[name] = arr
        return arr

    @staticmethod
    def _decode(raw: bytes):
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        n = int(round(np.sqrt(len(raw) // 2)))
        a = np.frombuffer(raw, dtype=">i2").reshape(n, n).astype(np.float32)
        a[a < -1000] = np.nan  # voids
        return a

    def __call__(self, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        out = np.full(lat.shape, np.nan)
        li, lo = np.floor(lat).astype(int), np.floor(lon).astype(int)
        for a, b in set(zip(li.ravel().tolist(), lo.ravel().tolist())):
            tile = self._load(a, b)
            if tile is None:
                continue
            m = (li == a) & (lo == b)
            n = tile.shape[0] - 1
            # row 0 is the northern edge
            y = (a + 1 - lat[m]) * n
            x = (lon[m] - b) * n
            y0 = np.clip(np.floor(y).astype(int), 0, n - 1)
            x0 = np.clip(np.floor(x).astype(int), 0, n - 1)
            fy, fx = y - y0, x - x0
            v = (tile[y0, x0] * (1 - fx) * (1 - fy) + tile[y0, x0 + 1] * fx * (1 - fy)
                 + tile[y0 + 1, x0] * (1 - fx) * fy + tile[y0 + 1, x0 + 1] * fx * fy)
            out[m] = v
        out[~np.isfinite(out)] = self.fallback
        return out
