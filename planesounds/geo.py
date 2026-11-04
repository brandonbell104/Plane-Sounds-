"""Geodesy helpers: WGS84 <-> ECEF, distances and lat/lon grids."""
from __future__ import annotations

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
M_PER_DEG_LAT = 111_320.0


def lla_to_ecef(lat, lon, alt):
    """Geodetic (deg, deg, metres MSL~ellipsoid) -> ECEF metres. Vectorised."""
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    alt = np.asarray(alt, dtype=float)
    s = np.sin(lat)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * s * s)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - WGS84_E2) + alt) * s
    return np.stack([x, y, z], axis=-1)


def ecef_to_lla(xyz):
    """ECEF -> geodetic (Bowring, good to mm for our purposes)."""
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    b = WGS84_A * (1 - WGS84_F)
    ep2 = (WGS84_A**2 - b**2) / b**2
    p = np.hypot(x, y)
    th = np.arctan2(z * WGS84_A, p * b)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - WGS84_E2 * WGS84_A * np.cos(th) ** 3)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), alt


def enu_basis(lat0, lon0):
    """Rows: east, north, up unit vectors (ECEF) at a reference point."""
    la, lo = np.radians(lat0), np.radians(lon0)
    e = np.array([-np.sin(lo), np.cos(lo), 0.0])
    n = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    u = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return np.stack([e, n, u])


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def offset_latlon(lat, lon, de_m, dn_m):
    """Small offset (metres east/north) -> new lat/lon."""
    lat = np.asarray(lat, dtype=float)
    dlat = np.asarray(dn_m) / M_PER_DEG_LAT
    dlon = np.asarray(de_m) / (M_PER_DEG_LAT * np.cos(np.radians(lat)))
    return lat + dlat, np.asarray(lon) + dlon


class Grid:
    """Global quasi-regular grid with ~``res_m`` spacing.

    Cell (i, j): lat = i * dlat, lon = j * dlon(i) where dlon depends on the
    row latitude, so the east-west spacing stays ~res_m everywhere.
    Integer cell ids make deduplication across flights trivial.
    """

    def __init__(self, res_m: float):
        self.res_m = float(res_m)
        self.dlat = self.res_m / M_PER_DEG_LAT

    def row_dlon(self, i):
        lat = np.asarray(i) * self.dlat
        return self.res_m / (M_PER_DEG_LAT * np.maximum(np.cos(np.radians(lat)), 1e-6))

    def cell_of(self, lat, lon):
        i = np.round(np.asarray(lat) / self.dlat).astype(np.int64)
        j = np.round(np.asarray(lon) / self.row_dlon(i)).astype(np.int64)
        return i, j

    def center(self, i, j):
        i = np.asarray(i)
        return i * self.dlat, np.asarray(j) * self.row_dlon(i)

    @staticmethod
    def key(i, j):
        return (np.asarray(i, dtype=np.int64) << 32) ^ (np.asarray(j, dtype=np.int64) & 0xFFFFFFFF)

    @staticmethod
    def unkey(k):
        k = np.asarray(k, dtype=np.int64)
        j = (k & 0xFFFFFFFF).astype(np.int64)
        j = np.where(j >= 2**31, j - 2**32, j)
        i = (k - (j & 0xFFFFFFFF)) >> 32
        # undo xor: i<<32 ^ (j & mask) == (i<<32) + (j & mask) since low bits of i<<32 are 0
        return i, j

    def cells_near(self, lat, lon, radius_m):
        """All cell keys within ``radius_m`` (horizontal) of any of the given points."""
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        r_cells = int(np.ceil(radius_m / self.res_m))
        offs = np.arange(-r_cells, r_cells + 1)
        di, dj = np.meshgrid(offs, offs, indexing="ij")
        mask = (di**2 + dj**2) * self.res_m**2 <= (radius_m + self.res_m) ** 2
        di, dj = di[mask], dj[mask]
        i0, j0 = self.cell_of(lat, lon)
        keys = []
        chunk = max(1, 2_000_000 // max(len(di), 1))
        for s in range(0, len(i0), chunk):
            ii = (i0[s:s + chunk, None] + di[None, :]).ravel()
            # recompute j within each row so spacing is right at the new row latitude
            lon_pts = np.repeat(lon[s:s + chunk], len(di))
            jj = np.round(lon_pts / self.row_dlon(ii)).astype(np.int64) + np.tile(dj, min(chunk, len(i0) - s))
            keys.append(np.unique(self.key(ii, jj)))
        if not keys:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(keys))

    def bbox_cells(self, lat_min, lat_max, lon_min, lon_max):
        i_lo, i_hi = int(np.floor(lat_min / self.dlat)), int(np.ceil(lat_max / self.dlat))
        keys = []
        for i in range(i_lo, i_hi + 1):
            d = float(self.row_dlon(i))
            j = np.arange(int(np.floor(lon_min / d)), int(np.ceil(lon_max / d)) + 1)
            keys.append(self.key(np.full_like(j, i), j))
        return np.concatenate(keys) if keys else np.empty(0, dtype=np.int64)
