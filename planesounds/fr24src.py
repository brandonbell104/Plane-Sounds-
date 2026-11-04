"""Historical flight tracks from Flightradar24 via the open-source ``fr24`` package.

This is the "scrape it with a Python package" route from the reference video:

1. **Discover** which aircraft were in the area: request FR24's *historical*
   live-feed playback (a snapshot of every aircraft inside a bounding box at a
   past moment) every few minutes across the time window. Dense areas are
   split into tiles automatically (FR24 caps a snapshot at 1500 aircraft).
2. **Download full tracks**: for every aircraft found, fetch its flight
   playback - the entire recorded path of that flight (lat, lon, altitude,
   speed every few seconds).

Everything is cached on disk under ``~/.planesounds/fr24`` so an interrupted
download resumes where it stopped and re-runs are free.

Notes
-----
* This is historical data, but how far back you can go depends on your FR24
  account: anonymous/free access only covers recent days; paid plans go back
  months to years. A login can be given (username/password) to extend it.
* ``fr24`` is unofficial. Respect Flightradar24's terms of use, keep the
  request rate low and use it for personal projects only.
"""
from __future__ import annotations

import asyncio
import json
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import adsb
from .aircraft_types import classify

CACHE = Path.home() / ".planesounds" / "fr24"
FT = 0.3048
KT = 0.514444


class FR24Error(RuntimeError):
    pass


@dataclass
class ScrapeStats:
    snapshots: int = 0
    tiles_split: int = 0
    flights_seen: int = 0
    flights_skipped_jets: int = 0
    tracks_cached: int = 0
    tracks_downloaded: int = 0
    tracks_failed: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.snapshots} snapshots, {self.flights_seen} aircraft found "
                f"({self.flights_skipped_jets} jets skipped), tracks: {self.tracks_downloaded} downloaded, "
                f"{self.tracks_cached} from cache, {self.tracks_failed} failed")


def available() -> bool:
    try:
        import fr24  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# low-level calls (kept small so tests can replace them)
# ---------------------------------------------------------------------------

def _check_http(res, what: str):
    code = getattr(getattr(res, "response", None), "status_code", 200)
    if code in (401, 402, 403):
        raise FR24Error(f"{what}: HTTP {code} - access refused. Historical data this old probably needs an "
                        "FR24 login/subscription (or the request was blocked).")
    if code == 429:
        raise FR24Error("rate limited")
    if code >= 400:
        raise FR24Error(f"{what}: HTTP {code}")


def snapshot_records(res) -> list[dict]:
    """Flatten a live-feed playback result into plain dicts."""
    from fr24.grpc import live_feed_flightdata_dict

    return [live_feed_flightdata_dict(f) for f in res.to_proto().live_feed_response.flights_list]


async def _snapshot(fr, bbox, ts: int, limit: int) -> list[dict]:
    from fr24 import BoundingBox

    s, n, w, e = bbox
    res = await fr.live_feed_playback.fetch(BoundingBox(s, n, w, e), timestamp=int(ts), duration=7, limit=limit)
    _check_http(res, "live feed playback")
    return snapshot_records(res)


async def _playback(fr, flight_id: int, ts_hint: int | None) -> dict:
    res = await fr.playback.fetch(flight_id=flight_id, timestamp=ts_hint)
    _check_http(res, "flight playback")
    return res.to_dict()


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def _split(b):
    s, n, w, e = b
    mn, me = (s + n) / 2, (w + e) / 2
    return [(s, mn, w, me), (s, mn, me, e), (mn, n, w, me), (mn, n, me, e)]


async def _with_retry(coro_fn, tries=3, base=2.0):
    for k in range(tries):
        try:
            return await coro_fn()
        except FR24Error as ex:
            if "rate limited" not in str(ex) or k == tries - 1:
                raise
        except Exception:
            if k == tries - 1:
                raise
        await asyncio.sleep(base * (2 ** k))


async def discover(fr, bbox, t_lo: float, t_hi: float, step_s: float = 300, limit: int = 1500,
                   delay_s: float = 0.6, progress=None, stats: ScrapeStats | None = None) -> dict[int, dict]:
    """Every aircraft seen in ``bbox`` at any snapshot between t_lo and t_hi."""
    stats = stats or ScrapeStats()
    seen: dict[int, dict] = {}
    times = np.arange(int(t_lo), int(t_hi) + 1, int(step_s))
    snap_dir = CACHE / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for k, ts in enumerate(times):
        queue = [(tuple(round(v, 4) for v in bbox), 0)]
        while queue:
            b, depth = queue.pop()
            fp = snap_dir / f"{int(ts)}_{b[0]}_{b[1]}_{b[2]}_{b[3]}.json"
            if fp.exists():
                recs = json.loads(fp.read_text())
            else:
                try:
                    recs = await _with_retry(lambda: _snapshot(fr, b, int(ts), limit))
                except Exception as ex:
                    stats.errors.append(f"snapshot {int(ts)}: {ex}")
                    if isinstance(ex, FR24Error) and "access refused" in str(ex):
                        raise
                    if stats.snapshots == 0 and len(stats.errors) >= 3:   # nothing works: stop early
                        raise FR24Error(f"could not reach Flightradar24 - {stats.errors[0]}")
                    continue
                fp.write_text(json.dumps(recs, default=str))
                await asyncio.sleep(delay_s)
            stats.snapshots += 1
            if len(recs) >= 0.95 * limit and depth < 5:
                stats.tiles_split += 1
                queue += [(sb, depth + 1) for sb in _split(b)]
                continue
            for r in recs:
                fid = int(r["flightid"])
                info = seen.setdefault(fid, {"flightid": fid, "callsign": r.get("callsign") or "",
                                             "typecode": (r.get("typecode") or "").upper(),
                                             "registration": r.get("registration") or "",
                                             "first_seen": int(r.get("timestamp", ts * 1000)) // 1000})
                info["last_seen"] = int(r.get("timestamp", ts * 1000)) // 1000
        if progress:
            progress((k + 1) / len(times), f"snapshot {k + 1}/{len(times)}: {len(seen)} aircraft so far")
    stats.flights_seen = len(seen)
    if stats.snapshots == 0 and stats.errors:
        raise FR24Error(f"could not get any snapshot from Flightradar24 - first error: {stats.errors[0]}")
    return seen


def _track_rows(d: dict, fid: int, fallback: dict) -> pd.DataFrame | None:
    try:
        flight = d["result"]["response"]["data"]["flight"]
    except (KeyError, TypeError):
        return None
    track = flight.get("track") or []
    if not track:
        return None
    ac = flight.get("aircraft") or {}
    ident = flight.get("identification") or {}
    model = ((ac.get("model") or {}).get("code") or fallback.get("typecode") or "").upper()
    hexid = ((ac.get("identification") or {}).get("modes") or "").lower()
    cs = ident.get("callsign") or fallback.get("callsign") or ""
    rows = {
        "time": [p["timestamp"] for p in track],
        "lat": [p["latitude"] for p in track],
        "lon": [p["longitude"] for p in track],
        "alt_m": [(p.get("altitude") or {}).get("feet", np.nan) * FT for p in track],
        "gs_ms": [(p.get("speed") or {}).get("kts", np.nan) * KT for p in track],
    }
    df = pd.DataFrame(rows)
    df["icao24"] = hexid or f"fr24{fid:x}"
    df["callsign"] = cs
    df["typecode"] = model
    df["flight_id"] = f"fr24_{fid:x}"
    # FR24 reports 0 ft on the ground
    df["on_ground"] = (df["alt_m"] <= 0) & (df["gs_ms"] < 30)
    return df


async def download_tracks(fr, seen: dict[int, dict], skip_jets: bool = False, delay_s: float = 0.6,
                          progress=None, stats: ScrapeStats | None = None) -> list[pd.DataFrame]:
    stats = stats or ScrapeStats()
    tdir = CACHE / "tracks"
    tdir.mkdir(parents=True, exist_ok=True)
    out = []
    items = list(seen.values())
    for k, info in enumerate(items):
        fid = info["flightid"]
        if skip_jets and classify(info["typecode"]) == "jet":
            stats.flights_skipped_jets += 1
            continue
        fp = tdir / f"{fid:x}.json"
        d = None
        if fp.exists():
            d = json.loads(fp.read_text())
            stats.tracks_cached += 1
        else:
            try:
                d = await _with_retry(lambda: _playback(fr, fid, None))
                if _track_rows(d, fid, info) is None:   # retry with the time we saw it
                    d = await _with_retry(lambda: _playback(fr, fid, info["first_seen"]))
                fp.write_text(json.dumps(d, default=str))
                stats.tracks_downloaded += 1
                await asyncio.sleep(delay_s)
            except Exception as ex:
                stats.tracks_failed += 1
                stats.errors.append(f"track {fid:x} ({info['callsign']}): {ex}")
                if isinstance(ex, FR24Error) and "access refused" in str(ex) and stats.tracks_downloaded == 0:
                    raise
        if d is not None:
            df = _track_rows(d, fid, info)
            if df is not None:
                out.append(df)
        if progress:
            progress((k + 1) / max(len(items), 1), f"track {k + 1}/{len(items)} ({info['callsign'] or fid})")
    return out


async def _scrape_async(bbox, t_lo, t_hi, step_s, skip_jets, username, password, delay_s,
                        progress, stats, clip_margin_deg, client_factory, limit):
    if client_factory is None:
        from fr24 import FR24

        client_factory = FR24
    async with client_factory() as fr:
        if username and password and hasattr(fr, "login"):
            await fr.login({"username": username, "password": password})
        seen = await discover(fr, bbox, t_lo, t_hi, step_s, limit=limit, delay_s=delay_s, stats=stats,
                              progress=(lambda x, m: progress(0.3 * x, m)) if progress else None)
        parts = await download_tracks(fr, seen, skip_jets, delay_s, stats=stats,
                                      progress=(lambda x, m: progress(0.3 + 0.7 * x, m)) if progress else None)
    if not parts:
        return pd.DataFrame(columns=adsb.COLUMNS + ["propulsion"])
    df = pd.concat(parts, ignore_index=True)
    # keep the data small: only the part of each path near the area & time window
    s, n, w, e = bbox
    m = clip_margin_deg
    keep = ((df.lat >= s - m) & (df.lat <= n + m) & (df.lon >= w - m) & (df.lon <= e + m)
            & (df.time >= t_lo - 7200) & (df.time <= t_hi + 7200))
    return adsb.finalize(df[keep])


def scrape(bbox, t_lo: float, t_hi: float, step_s: float = 300, skip_jets: bool = False,
           username: str | None = None, password: str | None = None, delay_s: float = 0.6,
           progress: Callable[[float, str], None] | None = None, out_path: str | Path | None = None,
           clip_margin_deg: float = 0.5, client_factory=None, limit: int = 1500) -> tuple[pd.DataFrame, ScrapeStats]:
    """Download historical flight tracks for an area and time window.

    ``bbox`` = (lat_min, lat_max, lon_min, lon_max); times are unix seconds (UTC).
    Returns the normalised position table and statistics.
    """
    stats = ScrapeStats()
    t0 = _time.time()
    df = asyncio.run(_scrape_async((bbox[0], bbox[1], bbox[2], bbox[3]), t_lo, t_hi, step_s, skip_jets,
                                   username, password, delay_s, progress, stats, clip_margin_deg,
                                   client_factory, limit))
    if out_path is not None and len(df):
        adsb.save(df, out_path)
    stats.errors = stats.errors[-50:]
    if progress:
        progress(1.0, f"done in {_time.time() - t0:.0f} s - {stats.summary()}")
    return df, stats
