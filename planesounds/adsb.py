"""Historical ADS-B data: loading, normalising and downloading.

Everything is normalised to one table (one row per position report):

    flight_id, icao24, callsign, typecode, time (unix s), lat, lon,
    alt_m (metres above mean sea level), gs_ms, on_ground

Supported sources
-----------------
* readsb / tar1090 ``trace_full_*.json`` files (adsb.lol, ADSBExchange
  globe_history, airplanes.live dumps, your own receiver). Plain or gzipped.
* adsb.lol daily ``globe_history`` releases on GitHub (auto-download,
  streamed and filtered to your bounding box so only a small parquet is kept).
* OpenSky Network (Trino via the optional ``pyopensky`` package, needs a
  free account with historical access) or OpenSky CSV/parquet exports.
* FlightRadar24 per-flight CSV exports.
* Any CSV/parquet with recognisable column names.
"""
from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import os
import re
import tarfile
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .aircraft_types import classify

FT = 0.3048
KT = 0.514444
COLUMNS = ["flight_id", "icao24", "callsign", "typecode", "time", "lat", "lon", "alt_m", "gs_ms",
           "on_ground"]
DATA_DIR = Path.home() / ".planesounds" / "adsb"


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _find(cols, *names):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def normalize(df: pd.DataFrame, alt_unit: str | None = None, speed_unit: str | None = None) -> pd.DataFrame:
    """Map an arbitrary position table onto the standard columns."""
    cols = list(df.columns)
    out = pd.DataFrame()
    # FlightRadar24 export: Timestamp, UTC, Callsign, Position("lat,lon"), Altitude(ft), Speed(kt), Direction
    if _find(cols, "position") and not _find(cols, "lat", "latitude"):
        pos = df[_find(cols, "position")].astype(str).str.split(",", expand=True)
        df = df.assign(lat=pd.to_numeric(pos[0], errors="coerce"), lon=pd.to_numeric(pos[1], errors="coerce"))
        cols = list(df.columns)
        alt_unit = alt_unit or "ft"
        speed_unit = speed_unit or "kt"

    tcol = _find(cols, "time", "timestamp", "ts", "unix", "seen_time", "datetime", "utc")
    if tcol is None:
        raise ValueError(f"no time column found in {cols}")
    t = df[tcol]
    if np.issubdtype(t.dtype, np.number):
        tt = t.astype(float)
        if tt.median() > 1e12:  # ms
            tt = tt / 1000.0
    else:
        tt = pd.to_datetime(t, utc=True, errors="coerce").astype("int64") / 1e9
    out["time"] = tt.values

    out["lat"] = pd.to_numeric(df[_find(cols, "lat", "latitude")], errors="coerce").values
    out["lon"] = pd.to_numeric(df[_find(cols, "lon", "lng", "longitude", "long")], errors="coerce").values

    # altitude: prefer geometric (GNSS) over barometric
    acol = _find(cols, "alt_m", "geoaltitude", "geo_altitude", "alt_geom", "geom_alt", "altitude_m",
                 "baroaltitude", "baro_altitude", "alt_baro", "altitude", "alt")
    a = pd.to_numeric(df[acol], errors="coerce") if acol else pd.Series(np.nan, index=df.index)
    unit = alt_unit
    if unit is None:
        unit = "m" if acol and acol.lower() in ("alt_m", "geoaltitude", "baroaltitude", "altitude_m",
                                                "geo_altitude", "baro_altitude") else "ft"
    out["alt_m"] = (a * (FT if unit == "ft" else 1.0)).values
    # fill missing geo with baro if both exist
    bcol = _find(cols, "baroaltitude", "baro_altitude")
    if acol and bcol and acol != bcol:
        out["alt_m"] = out["alt_m"].fillna(pd.to_numeric(df[bcol], errors="coerce"))

    scol = _find(cols, "gs_ms", "velocity", "gs", "groundspeed", "ground_speed", "speed", "spd")
    s = pd.to_numeric(df[scol], errors="coerce") if scol else pd.Series(np.nan, index=df.index)
    su = speed_unit or ("ms" if scol and scol.lower() in ("gs_ms", "velocity") else "kt")
    out["gs_ms"] = (s * (KT if su == "kt" else 1.0)).values

    icol = _find(cols, "icao24", "icao", "hex", "icao_hex", "modes")
    out["icao24"] = df[icol].astype(str).str.lower().str.strip().values if icol else "unknown"
    ccol = _find(cols, "callsign", "flight", "call")
    out["callsign"] = df[ccol].astype(str).str.strip().values if ccol else ""
    tycol = _find(cols, "typecode", "type", "t", "aircraft_type", "icaotype", "model")
    out["typecode"] = df[tycol].astype(str).str.upper().str.strip().values if tycol else ""
    gcol = _find(cols, "on_ground", "onground", "ground")
    out["on_ground"] = df[gcol].astype(bool).values if gcol else False
    if "flight_id" in df.columns:
        out["flight_id"] = df["flight_id"].astype(str).values
    return finalize(out)


def finalize(df: pd.DataFrame, gap_s: float = 600.0) -> pd.DataFrame:
    df = df.dropna(subset=["time", "lat", "lon"]).copy()
    df["alt_m"] = df["alt_m"].astype(float)
    df = df.sort_values(["icao24", "time"]).reset_index(drop=True)
    if "flight_id" not in df.columns:
        df["flight_id"] = np.nan
    miss = df["flight_id"].isna() | (df["flight_id"].astype(str).isin(["", "nan", "None"]))
    if miss.any():
        gap = df.groupby("icao24")["time"].diff().fillna(0) > gap_s
        seg = gap.groupby(df["icao24"]).cumsum()
        auto = df["icao24"].astype(str) + "_" + seg.astype(int).astype(str)
        df["flight_id"] = df["flight_id"].astype(object).where(~miss, auto)
    df["flight_id"] = df["flight_id"].astype(str)
    # propagate the most common typecode / callsign within a flight
    for c in ("typecode", "callsign"):
        df[c] = df[c].replace({"NAN": "", "nan": "", "None": "", "NONE": ""}).fillna("")
        mode = df[df[c] != ""].groupby("flight_id")[c].agg(lambda s: s.value_counts().index[0])
        df[c] = df["flight_id"].map(mode).fillna("")
    df["propulsion"] = df["typecode"].map(classify)
    return df[COLUMNS + ["propulsion"]]


def load_table(path: str | Path, **kw) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        raw = pd.read_parquet(path)
        if set(COLUMNS) <= set(raw.columns):
            return finalize(raw)
    elif path.suffix.lower() in (".json", ".gz") or path.name.startswith("trace_"):
        return load_readsb_traces([path])
    else:
        raw = pd.read_csv(path, low_memory=False)
    return normalize(raw, **kw)


def save(df: pd.DataFrame, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# readsb trace_full JSON
# ---------------------------------------------------------------------------

def _read_maybe_gz(b: bytes) -> dict:
    if b[:2] == b"\x1f\x8b":
        b = gzip.decompress(b)
    return json.loads(b)


def parse_trace(obj: dict, bbox=None, t_window=None) -> pd.DataFrame | None:
    """Parse one readsb ``trace_full`` object.

    trace rows: [dt, lat, lon, alt_baro_ft|"ground"|None, gs_kt, track, flags, vrate,
                 aircraft{...}|None, source, alt_geom_ft, geom_rate, ias, roll]
    """
    tr = obj.get("trace") or []
    if not tr:
        return None
    t0 = float(obj.get("timestamp", 0))
    n = len(tr)
    tt = np.empty(n)
    la = np.empty(n)
    lo = np.empty(n)
    ab = np.full(n, np.nan)
    ag = np.full(n, np.nan)
    gs = np.full(n, np.nan)
    gnd = np.zeros(n, bool)
    cs = [""] * n
    for k, row in enumerate(tr):
        tt[k] = t0 + row[0]
        la[k] = row[1]
        lo[k] = row[2]
        a = row[3] if len(row) > 3 else None
        if a == "ground":
            gnd[k] = True
        elif a is not None:
            ab[k] = a
        if len(row) > 4 and row[4] is not None:
            gs[k] = row[4]
        if len(row) > 10 and row[10] is not None:
            ag[k] = row[10]
        if len(row) > 8 and isinstance(row[8], dict):
            cs[k] = (row[8].get("flight") or "").strip()
    keep = np.ones(n, bool)
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        keep &= (la >= lat_min) & (la <= lat_max) & (lo >= lon_min) & (lo <= lon_max)
    if t_window is not None:
        keep &= (tt >= t_window[0]) & (tt <= t_window[1])
    if not keep.any():
        return None
    alt_ft = np.where(np.isfinite(ag), ag, ab)
    df = pd.DataFrame({
        "icao24": str(obj.get("icao", "")).lower().lstrip("~"),
        "callsign": pd.Series(cs).replace("", np.nan).ffill().bfill().fillna("").values,
        "typecode": (obj.get("t") or "").upper(),
        "time": tt, "lat": la, "lon": lo, "alt_m": alt_ft * FT, "gs_ms": gs * KT, "on_ground": gnd,
    })
    return df[keep]


def load_readsb_traces(paths: Iterable[str | Path], bbox=None, t_window=None,
                       progress: Callable[[float], None] | None = None) -> pd.DataFrame:
    """Load trace files, directories of trace files or ``.tar`` archives."""
    files: list[Path] = []
    tars: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += [q for q in p.rglob("trace_full_*") if q.is_file()]
            tars += sorted(p.glob("*.tar"))
        elif p.suffix == ".tar" or ".tar." in p.name:
            tars.append(p)
        else:
            files.append(p)
    parts = []
    for k, f in enumerate(files):
        try:
            d = parse_trace(_read_maybe_gz(f.read_bytes()), bbox, t_window)
            if d is not None:
                parts.append(d)
        except Exception:
            pass
        if progress and k % 200 == 0:
            progress(k / max(len(files), 1))
    for tp in tars:
        with open(tp, "rb") as fh:
            parts += list(_iter_tar_traces(fh, bbox, t_window))
    if not parts:
        return pd.DataFrame(columns=COLUMNS + ["propulsion"])
    return finalize(pd.concat(parts, ignore_index=True))


def _iter_tar_traces(fileobj, bbox, t_window, progress=None, total_bytes=None):
    with tarfile.open(fileobj=fileobj, mode="r|*") as tf:
        for k, m in enumerate(tf):
            if not m.isfile() or "trace_full" not in m.name:
                continue
            try:
                d = parse_trace(_read_maybe_gz(tf.extractfile(m).read()), bbox, t_window)
            except Exception:
                continue
            if d is not None:
                yield d
            if progress and total_bytes and k % 500 == 0:
                try:
                    progress(min(fileobj.tell() / total_bytes, 1.0))
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# adsb.lol globe_history daily releases (GitHub)
# ---------------------------------------------------------------------------

class _ChainedHTTP(io.RawIOBase):
    """Read several HTTP responses back to back as one stream (split tars)."""

    def __init__(self, urls, session):
        self.urls = list(urls)
        self.session = session
        self.resp = None
        self.pos = 0

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def _next(self):
        if self.resp is not None:
            self.resp.close()
        if not self.urls:
            self.resp = None
            return False
        self.resp = self.session.get(self.urls.pop(0), stream=True, timeout=60)
        self.resp.raise_for_status()
        self.resp.raw.decode_content = True
        return True

    def readinto(self, b):
        while True:
            if self.resp is None and not self._next():
                return 0
            n = self.resp.raw.readinto(b)
            if n:
                self.pos += n
                return n
            self.resp.close()
            self.resp = None


def adsblol_assets(day: dt.date, token: str | None = None) -> list[dict]:
    """List the release assets for one day of adsb.lol globe_history."""
    import requests

    token = token or os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    tag_prefix = f"v{day:%Y.%m.%d}-planes-readsb-prod"
    out = []
    for repo in (f"adsblol/globe_history_{day.year}", "adsblol/globe_history"):
        page = 1
        while page < 40:
            r = requests.get(f"https://api.github.com/repos/{repo}/releases",
                             params={"per_page": 100, "page": page}, headers=headers, timeout=30)
            if r.status_code != 200:
                break
            rels = r.json()
            if not rels:
                break
            for rel in rels:
                if rel.get("tag_name", "").startswith(tag_prefix):
                    for a in rel.get("assets", []):
                        if ".tar" in a["name"]:
                            out.append({"name": a["name"], "url": a["browser_download_url"],
                                        "size": a["size"], "tag": rel["tag_name"]})
            if out:
                break
            page += 1
        if out:
            break
    # prefer one tag (the -0 prod), keep split parts in order
    if out:
        tags = sorted({a["tag"] for a in out})
        out = sorted([a for a in out if a["tag"] == tags[0]], key=lambda a: a["name"])
    return out


def download_adsblol_day(day: dt.date, bbox, t_window=None, out_path: str | Path | None = None,
                         progress: Callable[[float, str], None] | None = None,
                         token: str | None = None) -> pd.DataFrame:
    """Stream one day of adsb.lol history and keep only points in ``bbox``.

    The full day is several GB; it is filtered on the fly, never stored.
    """
    import requests

    assets = adsblol_assets(day, token)
    if not assets:
        raise RuntimeError(f"no adsb.lol release found for {day}. Check "
                           f"https://github.com/adsblol/globe_history_{day.year}/releases")
    total = sum(a["size"] for a in assets)
    if progress:
        progress(0.0, f"streaming {len(assets)} part(s), {total / 1e9:.1f} GB")
    stream = io.BufferedReader(_ChainedHTTP([a["url"] for a in assets], requests.Session()),
                               buffer_size=1 << 20)
    parts = []

    def prog(x):
        if progress:
            progress(x, f"{x * total / 1e9:.2f} / {total / 1e9:.1f} GB, {len(parts)} aircraft in area")

    for d in _iter_tar_traces(stream, bbox, t_window, progress=prog, total_bytes=total):
        parts.append(d)
    df = finalize(pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame(columns=COLUMNS)
    if out_path:
        save(df, out_path)
    return df


# ---------------------------------------------------------------------------
# OpenSky (optional)
# ---------------------------------------------------------------------------

def opensky_history(t_start: float, t_end: float, bbox) -> pd.DataFrame:
    """Query OpenSky's historical database via Trino (``pip install pyopensky``,
    requires an OpenSky account with historical data access)."""
    try:
        from pyopensky.trino import Trino
    except ImportError as e:
        raise RuntimeError("pip install pyopensky  (and configure your OpenSky credentials)") from e
    lat_min, lat_max, lon_min, lon_max = bbox
    df = Trino().history(
        pd.Timestamp(t_start, unit="s", tz="UTC"), pd.Timestamp(t_end, unit="s", tz="UTC"),
        bounds=(lon_min, lat_min, lon_max, lat_max),
    )
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=COLUMNS)
    return normalize(df.reset_index(drop=True))


def merge_typecodes(df: pd.DataFrame, aircraft_db_csv: str | Path) -> pd.DataFrame:
    """Fill missing type codes from an aircraft database CSV (e.g. OpenSky's
    aircraftDatabase.csv, or a tar1090-db export) with icao24 + typecode columns."""
    db = pd.read_csv(aircraft_db_csv, low_memory=False, dtype=str)
    icol = _find(db.columns, "icao24", "icao", "hex")
    tcol = _find(db.columns, "typecode", "icaotype", "type", "t")
    m = dict(zip(db[icol].str.lower(), db[tcol].fillna("").str.upper()))
    df = df.copy()
    miss = df["typecode"].fillna("") == ""
    df.loc[miss, "typecode"] = df.loc[miss, "icao24"].map(m).fillna("")
    df["propulsion"] = df["typecode"].map(classify)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("flight_id")
    return pd.DataFrame({
        "icao24": g["icao24"].first(), "callsign": g["callsign"].first(),
        "typecode": g["typecode"].first(), "propulsion": g["propulsion"].first(),
        "start": pd.to_datetime(g["time"].min(), unit="s", utc=True),
        "end": pd.to_datetime(g["time"].max(), unit="s", utc=True),
        "points": g.size(), "median_alt_m": g["alt_m"].median(), "median_gs_ms": g["gs_ms"].median(),
    }).reset_index()
