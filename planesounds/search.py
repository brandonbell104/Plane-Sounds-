"""Geolocation search: where on Earth would a microphone have heard these
Doppler curves at these relative times (and nothing else in between)?

Stages (each narrows the candidate set before the next, more expensive one):

1. Flight pre-filter      - area, time window, propulsion type (drop jets).
2. Speed filter           - only track samples whose 3-D speed matches each
                            event's measured speed range.
3. Slant-range candidates - grid cells whose CPA slant distance to a
                            speed-matching flight falls in the measured range.
4. Time-offset filter     - delayed-CPA times must reproduce the observed gap
                            between events (+/- tolerance), and fall inside the
                            allowed recording-start window.
5. Silence filter         - reject cells that would have heard *other*
                            aircraft during parts of the recording where no
                            aircraft is audible.
6. Doppler scoring        - refine to a finer grid, model the exact received
                            curve from the real trajectory (retarded time), and
                            score it against the spectrogram (event 1 time and
                            all f0's swept; later events time-locked).
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d
from scipy.spatial import cKDTree

from .doppler import resample_track, track_received_curve
from .cuts import segment_bounds, segment_of
from .geo import Grid, lla_to_ecef, offset_latlon
from .spectro import Spectrogram


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """One measured Doppler pass-by in the recording (audio clock, seconds)."""
    name: str
    t_cpa: float                       # audio time of the CPA (curve midpoint)
    f0: float                          # emitted (rest) frequency of the traced line
    v_range: tuple[float, float]       # aircraft speed bounds (m/s)
    d_range: tuple[float, float]       # CPA slant distance bounds (m)
    t_window: tuple[float, float]      # audio span used for scoring
    f_band: tuple[float, float]        # frequency band of the curve
    harmonics: list[float] = field(default_factory=lambda: [1.0])
    trace_t: list[float] | None = None
    trace_f: list[float] | None = None
    score_mode: str = "spectrogram"   # or "trace" (for very faint curves)
    allow_jets: bool = False

    def to_dict(self):
        d = dict(self.__dict__)
        for k in ("trace_t", "trace_f"):
            if d[k] is not None:
                d[k] = [float(x) for x in d[k]]
        return d


@dataclass
class SearchConfig:
    bbox: tuple[float, float, float, float]        # lat_min, lat_max, lon_min, lon_max
    t0_range: tuple[float, float]                  # unix time range of audio t=0
    duration: float                                # recording length (s)
    grid_m: float = 200.0
    refine_m: float = 100.0
    c: float = 343.0
    offset_tol_s: float = 6.0
    exclude_jets: bool = True
    silence: bool = True
    silence_radius_m: float = 2500.0               # props / unknown / helis
    silence_radius_jet_m: float = 4500.0
    clean_spans: list[tuple[float, float]] | None = None  # audio spans with no audible aircraft
    silence_pad_s: float = 15.0
    max_violations: int = 0
    max_score_points: int = 40000
    dt_sweep_s: float = 4.0
    f0_sweep_frac: float = 0.03
    smooth_s: float = 4.0
    alt_offset_m: float = 0.0                      # add to all ADS-B altitudes
    max_gap_s: float = 90.0
    cuts: list[float] = field(default_factory=list)   # audio times of edits/cuts
    # time removed at each cut (s): (min, max). (0, None) = segments are in chronological
    # order, unknown amount removed. None = segments may be in any order.
    cut_removed_s: tuple[float, float | None] | None = (0.0, None)


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

@dataclass
class Track:
    fid: str
    icao24: str
    callsign: str
    typecode: str
    propulsion: str
    t: np.ndarray
    xyz: np.ndarray
    vel: np.ndarray
    speed: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    valid: np.ndarray
    _tree: cKDTree | None = None

    @property
    def tree(self):
        if self._tree is None:
            self._tree = cKDTree(self.xyz)
        return self._tree


def prepare_tracks(df: pd.DataFrame, t_lo: float, t_hi: float, bbox=None, margin_deg: float = 0.15,
                   alt_offset_m: float = 0.0, smooth_s: float = 4.0, max_gap_s: float = 90.0,
                   progress=None) -> dict[str, Track]:
    d = df[(df.time >= t_lo) & (df.time <= t_hi)]
    d = d[~d.on_ground.astype(bool) & d.alt_m.notna()]
    if bbox is not None:
        la0, la1, lo0, lo1 = bbox
        d = d[(d.lat >= la0 - margin_deg) & (d.lat <= la1 + margin_deg)
              & (d.lon >= lo0 - margin_deg) & (d.lon <= lo1 + margin_deg)]
    tracks = {}
    groups = list(d.groupby("flight_id", sort=False))
    for k, (fid, g) in enumerate(groups):
        if len(g) < 3:
            continue
        g = g.sort_values("time")
        raw_t = g.time.values.astype(float)
        xyz = lla_to_ecef(g.lat.values, g.lon.values, g.alt_m.values + alt_offset_m)
        t_u, x_u, v_u = resample_track(raw_t, xyz, 1.0, smooth_s)
        if len(t_u) < 5:
            continue
        j = np.clip(np.searchsorted(raw_t, t_u), 1, len(raw_t) - 1)
        gap = np.maximum(t_u - raw_t[j - 1], 0) + np.maximum(raw_t[j] - t_u, 0)
        valid = gap <= max_gap_s
        lat_u = np.interp(t_u, raw_t, g.lat.values)
        lon_u = np.interp(t_u, raw_t, g.lon.values)
        alt_u = np.interp(t_u, raw_t, g.alt_m.values)
        r0 = g.iloc[0]
        tracks[fid] = Track(str(fid), str(r0.icao24), str(r0.callsign), str(r0.typecode),
                            str(r0.get("propulsion", "unknown")), t_u, x_u, v_u,
                            np.linalg.norm(v_u, axis=1), lat_u, lon_u, alt_u, valid)
        if progress and k % 50 == 0:
            progress(k / max(len(groups), 1))
    return tracks


# ---------------------------------------------------------------------------
# Stage 2-4: candidates
# ---------------------------------------------------------------------------

def _cpa_query(track: Track, obs_xyz: np.ndarray, c: float):
    """Vectorised CPA of many observers to one track (KD-tree + projection)."""
    dist, idx = track.tree.query(obs_xyz)
    rel = obs_xyz - track.xyz[idx]
    v = track.vel[idx]
    vv = np.einsum("ij,ij->i", v, v)
    off = np.where(vv > 1, np.einsum("ij,ij->i", rel, v) / np.maximum(vv, 1), 0.0)
    off = np.clip(off, -1.0, 1.0)
    tau = track.t[idx] + off
    d = np.linalg.norm(obs_xyz - (track.xyz[idx] + v * off[:, None]), axis=1)
    edge = (idx == 0) | (idx == len(track.t) - 1)
    return tau, d, idx, edge


def event_candidates(ev: Event, tracks: dict[str, Track], cfg: SearchConfig, grid: Grid, dem,
                     cell_cache: dict, progress=None) -> pd.DataFrame:
    vlo, vhi = ev.v_range
    dlo, dhi = ev.d_range
    t0_lo, t0_hi = cfg.t0_range
    tau_lo = t0_lo + ev.t_cpa - dhi / cfg.c - cfg.offset_tol_s - 5
    tau_hi = t0_hi + ev.t_cpa + cfg.offset_tol_s + 5
    rows = []
    items = list(tracks.values())
    for n, tr in enumerate(items):
        if progress and n % 20 == 0:
            progress(n / max(len(items), 1))
        if not ev.allow_jets and cfg.exclude_jets and tr.propulsion == "jet":
            continue
        m = (tr.valid & (tr.speed >= vlo) & (tr.speed <= vhi) & (tr.t >= tau_lo) & (tr.t <= tau_hi))
        if not m.any():
            continue
        sel = np.flatnonzero(m)
        step = max(1, int(cfg.grid_m / 2 / max(np.median(tr.speed[sel]), 1)))
        sel = sel[::step]
        if cfg.bbox is not None:
            la0, la1, lo0, lo1 = cfg.bbox
            pad = dhi / 111_000
            inb = ((tr.lat[sel] >= la0 - pad) & (tr.lat[sel] <= la1 + pad)
                   & (tr.lon[sel] >= lo0 - pad * 1.5) & (tr.lon[sel] <= lo1 + pad * 1.5))
            sel = sel[inb]
            if not len(sel):
                continue
        keys = grid.cells_near(tr.lat[sel], tr.lon[sel], dhi)
        lat, lon = grid.center(*grid.unkey(keys))
        if cfg.bbox is not None:
            la0, la1, lo0, lo1 = cfg.bbox
            inb = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
            keys, lat, lon = keys[inb], lat[inb], lon[inb]
        if not len(keys):
            continue
        obs = _cell_xyz(keys, lat, lon, dem, cell_cache)
        tau, d, idx, edge = _cpa_query(tr, obs, cfg.c)
        spd = tr.speed[idx]
        heard = tau + d / cfg.c
        T0 = heard - ev.t_cpa
        ok = (~edge & tr.valid[idx] & (d >= dlo) & (d <= dhi) & (spd >= vlo) & (spd <= vhi)
              & (T0 >= t0_lo - cfg.offset_tol_s) & (T0 <= t0_hi + cfg.offset_tol_s))
        if ok.any():
            rows.append(pd.DataFrame({"key": keys[ok], "fid": tr.fid, "heard": heard[ok], "d": d[ok],
                                      "speed": spd[ok], "T0": T0[ok]}))
    if not rows:
        return pd.DataFrame(columns=["key", "fid", "heard", "d", "speed", "T0"])
    return pd.concat(rows, ignore_index=True)


def _cell_xyz(keys, lat, lon, dem, cache):
    elev = np.empty(len(keys))
    miss = np.array([k not in cache for k in keys.tolist()], dtype=bool) if len(cache) else np.ones(len(keys), bool)
    if miss.any():
        e = dem(lat[miss], lon[miss])
        for k, v in zip(keys[miss].tolist(), e.tolist()):
            cache[k] = v
    elev[:] = [cache[k] for k in keys.tolist()]
    return lla_to_ecef(lat, lon, elev)


def event_segments(events: list[Event], cuts) -> list[int]:
    """Segment index (between cuts) of every event."""
    return [segment_of(e.t_cpa, cuts or []) for e in events]


def combine_events(cands: list[pd.DataFrame], events: list[Event], tol: float, segs=None,
                   cut_removed=(0.0, None)) -> pd.DataFrame:
    """Join per-event candidate tables on grid cell.

    Events in the *same* uncut segment must reproduce their observed time
    offset (within ``tol``). Events in different segments (separated by a cut)
    only have to agree on the location: each segment gets its own start time.
    Adds ``T0_s{seg}`` columns (unix time of audio t=0 implied by that segment).
    """
    segs = list(segs) if segs is not None else [0] * len(events)
    first_in_seg: dict[int, int] = {}
    base = cands[0].rename(columns={c: f"{c}_0" for c in ("fid", "heard", "d", "speed", "T0")})
    first_in_seg[segs[0]] = 0
    for k in range(1, len(cands)):
        ck = cands[k].rename(columns={c: f"{c}_{k}" for c in ("fid", "heard", "d", "speed", "T0")})
        base = base.merge(ck, on="key", how="inner")
        j = first_in_seg.setdefault(segs[k], k)
        if j != k:
            base = base[np.abs(base[f"T0_{k}"] - base[f"T0_{j}"]) <= tol]
        elif cut_removed is not None and segs[k - 1] != segs[k]:
            # new segment after a cut: T0 jumps by the amount of time removed
            p = first_in_seg[segs[k - 1]]
            removed = base[f"T0_{k}"] - base[f"T0_{p}"]
            lo, hi = cut_removed
            ok = removed >= (lo or 0.0) - tol
            if hi is not None:
                ok &= removed <= hi + tol
            base = base[ok]
        for j in range(k):
            base = base[base[f"fid_{k}"] != base[f"fid_{j}"]]
        if base.empty:
            break
    base = base.copy()
    for sg in sorted(set(segs)):
        cols = [f"T0_{k}" for k in range(len(events)) if segs[k] == sg and f"T0_{k}" in base]
        base[f"T0_s{sg}"] = base[cols].mean(axis=1) if len(base) and cols else np.nan
    base["T0"] = base[f"T0_s{segs[0]}"] if len(base) else []
    return base.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 5: silence
# ---------------------------------------------------------------------------

def default_clean_spans(events: list[Event], duration: float, pad: float, cuts=()) -> list[tuple[float, float]]:
    """Quiet spans = everything except the event windows, never crossing a cut
    (a small margin is left around every cut). Only segments that contain at
    least one event are returned - elsewhere the start time is unknown."""
    bounds = segment_bounds(cuts or [], duration)
    ev_segs = {segment_of(e.t_cpa, cuts or []) for e in events}
    spans = []
    for si, (sa, sb) in enumerate(bounds):
        if si not in ev_segs:
            continue
        sa2, sb2 = (sa + 1.0 if si > 0 else sa), (sb - 1.0 if si < len(bounds) - 1 else sb)
        busy = sorted((max(sa2, e.t_window[0] - pad), min(sb2, e.t_window[1] + pad))
                      for e in events if segment_of(e.t_cpa, cuts or []) == si)
        cur = sa2
        for a, b in busy:
            if a > cur:
                spans.append((cur, a))
            cur = max(cur, b)
        if cur < sb2:
            spans.append((cur, sb2))
    return [sp for sp in spans if sp[1] - sp[0] > 5]


def silence_violations(surv: pd.DataFrame, n_events: int, tracks: dict[str, Track], cfg: SearchConfig,
                       spans, grid: Grid, dem, cell_cache, progress=None) -> np.ndarray:
    """Number of non-event aircraft each surviving row would have clearly heard
    during the 'clean' (aircraft-free) spans of the recording."""
    if surv.empty or not spans:
        return np.zeros(len(surv), dtype=int)
    ukeys, inv = np.unique(surv.key.values, return_inverse=True)
    lat, lon = grid.center(*grid.unkey(ukeys))
    obs = _cell_xyz(ukeys, lat, lon, dem, cell_cache)
    tree = cKDTree(obs)
    spans = np.asarray(spans, dtype=float)
    seg_ids = sorted(int(c[4:]) for c in surv.columns if c.startswith("T0_s"))
    span_seg = np.array([segment_of(0.5 * (a + b), cfg.cuts or []) for a, b in spans])
    keep = np.isin(span_seg, seg_ids)          # spans in segments without events: start time unknown
    spans, span_seg = spans[keep], span_seg[keep]
    if not len(spans):
        return np.zeros(len(surv), dtype=int)
    T0_by_seg = {sg: surv[f"T0_s{sg}"].values for sg in seg_ids}
    T0_span = np.stack([T0_by_seg[sg] for sg in span_seg], 1)   # (rows, spans)
    T0 = surv.T0.values
    event_fids = np.stack([surv[f"fid_{k}"].astype(str).values for k in range(n_events)], 1)
    viol = np.zeros(len(surv), dtype=int)
    # row lookup by unique key
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(ukeys)))
    ends = np.searchsorted(inv[order], np.arange(len(ukeys)), side="right")
    t_lo, t_hi = np.nanmin(T0_span) + spans[:, 0].min() - 60, np.nanmax(T0_span) + spans[:, 1].max() + 60
    items = list(tracks.values())
    for n, tr in enumerate(items):
        if progress and n % 50 == 0:
            progress(n / max(len(items), 1))
        rad = cfg.silence_radius_jet_m if tr.propulsion == "jet" else cfg.silence_radius_m
        m = tr.valid & (tr.t >= t_lo - rad / cfg.c) & (tr.t <= t_hi)
        sel = np.flatnonzero(m)[::2]
        if not len(sel):
            continue
        hits = tree.query_ball_point(tr.xyz[sel], rad)
        si = np.repeat(sel, [len(h) for h in hits])
        if not len(si):
            continue
        ki = np.concatenate([np.asarray(h, dtype=int) for h in hits])
        dist = np.linalg.norm(tr.xyz[si] - obs[ki], axis=1)
        heard = tr.t[si] + dist / cfg.c
        # expand to rows sharing that cell
        hit_rows = set()
        for kk in np.unique(ki):
            rws = order[starts[kk]:ends[kk]]
            hk = heard[ki == kk]
            for r in rws:
                if (event_fids[r] == tr.fid).any():
                    continue
                a = hk[:, None] - T0_span[r][None, :]
                if np.any((a >= spans[None, :, 0]) & (a <= spans[None, :, 1])):
                    hit_rows.add(r)
        if hit_rows:
            viol[list(hit_rows)] += 1
    return viol


# ---------------------------------------------------------------------------
# Stage 6: Doppler scoring
# ---------------------------------------------------------------------------

class Scorer:
    def __init__(self, spec: Spectrogram, events: list[Event], c: float, dt_sweep: float, f0_frac: float,
                 cuts=()):
        self.spec = spec
        self.segs = event_segments(events, list(cuts))
        # widen ridges by +/-1 bin so tiny model errors still land on the curve
        z = maximum_filter1d(spec.z, size=3, axis=0)
        self.spec_wide = Spectrogram(spec.t, spec.f, spec.db, z)
        self.events = events
        self.c = c
        self.dts = np.arange(-dt_sweep, dt_sweep + 1e-9, 0.5)
        self.f0s = [ev.f0 * np.linspace(1 - f0_frac, 1 + f0_frac, 13) for ev in events]
        step = max(spec.dt, 0.25)
        self.grids = [np.arange(ev.t_window[0], ev.t_window[1] + 1e-9, step) for ev in events]

    def _curve(self, tr: Track, obs_xyz, t_center_emit, half=150.0):
        m = (tr.t >= t_center_emit - half) & (tr.t <= t_center_emit + half)
        t_recv, f_rel, _ = track_received_curve(obs_xyz, tr.t[m], tr.xyz[m], tr.vel[m], 1.0, self.c)
        return t_recv, f_rel

    def _event_scores(self, k, a_grid_unix_shift, t_recv, f_rel):
        """Score event k for many time shifts at once.

        a_grid_unix_shift: array (n_shift,) of T0 values (unix time of audio t=0).
        Returns array (n_shift,) of best-over-f0 scores and the argmax f0."""
        ev = self.events[k]
        ag = self.grids[k]
        # model relative frequency at each audio grid time, for each T0
        tt = ag[None, :] + a_grid_unix_shift[:, None]
        g = np.interp(tt, t_recv, f_rel, left=np.nan, right=np.nan)
        best = np.full(len(a_grid_unix_shift), -np.inf)
        bestf = np.zeros(len(a_grid_unix_shift))
        if ev.score_mode == "trace" and ev.trace_t is not None and len(ev.trace_t) > 3:
            tt2 = np.asarray(ev.trace_t)[None, :] + a_grid_unix_shift[:, None]
            g2 = np.interp(tt2, t_recv, f_rel, left=np.nan, right=np.nan)
            tf = np.asarray(ev.trace_f)[None, :]
            for f0 in self.f0s[k]:
                err = np.nanmean(np.abs(f0 * g2 - tf), axis=1)
                cov = np.mean(np.isfinite(g2), axis=1)
                sc = np.where(cov > 0.5, 10.0 / (1.0 + err), 0.0)
                upd = sc > best
                best[upd], bestf[upd] = sc[upd], f0
            return best, bestf
        tq = np.broadcast_to(ag[None, :], g.shape)
        for f0 in self.f0s[k]:
            acc = np.zeros(g.shape)
            for h in ev.harmonics:
                v = self.spec_wide.sample(tq, f0 * h * g)
                acc += np.nan_to_num(v)
            sc = acc.mean(axis=1) / len(ev.harmonics)
            upd = sc > best
            best[upd], bestf[upd] = sc[upd], f0
        return best, bestf

    def score(self, tracks_per_event: list[Track], obs_xyz: np.ndarray):
        """Joint score.

        Within each uncut segment the first event is swept in time and the
        others are locked to it by flight geometry. Different segments (split
        by a cut) are swept independently - their start times are unrelated.
        """
        curves, heard = [], []
        for tr in tracks_per_event:
            r = np.linalg.norm(tr.xyz - obs_xyz[None, :], axis=1)
            i = int(np.argmin(r))
            heard.append(tr.t[i] + r[i] / self.c)
            curves.append(self._curve(tr, obs_xyz, tr.t[i]))
        n = len(curves)
        ev_scores, ev_f0, ev_T0 = [0.0] * n, [0.0] * n, [0.0] * n
        total, seg_T0, seg_dt = 0.0, {}, {}
        for sg in sorted(set(self.segs)):
            ks = [k for k in range(n) if self.segs[k] == sg]
            ref = ks[0]
            T0s = heard[ref] - (self.events[ref].t_cpa + self.dts)
            tot = np.zeros(len(T0s))
            per = {}
            for k in ks:
                sc, f = self._event_scores(k, T0s, *curves[k])
                per[k] = (sc, f)
                tot += sc
            j = int(np.argmax(tot))
            total += float(tot[j])
            seg_T0[sg], seg_dt[sg] = float(T0s[j]), float(self.dts[j])
            for k in ks:
                ev_scores[k], ev_f0[k], ev_T0[k] = float(per[k][0][j]), float(per[k][1][j]), float(T0s[j])
        s0 = self.segs[0]
        return {"score": total / n, "T0": seg_T0[s0], "dt": seg_dt[s0], "T0_segs": seg_T0,
                "T0_events": ev_T0, "event_scores": ev_scores, "f0_best": ev_f0}

    def model_curves(self, tracks_per_event, obs_xyz, T0, f0s):
        """``T0`` may be one value or one per event (events in different segments)."""
        T0s = np.broadcast_to(np.asarray(T0, dtype=float), (len(tracks_per_event),))
        out = []
        for k, tr in enumerate(tracks_per_event):
            r = np.linalg.norm(tr.xyz - obs_xyz[None, :], axis=1)
            t_recv, f_rel = self._curve(tr, obs_xyz, tr.t[int(np.argmin(r))], half=300)
            out.append((t_recv - T0s[k], f_rel * f0s[k]))
        return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    results: pd.DataFrame
    best_per_combo: pd.DataFrame
    stages: list[dict]
    candidates: list[pd.DataFrame]
    survivors: pd.DataFrame
    tracks: dict
    log: list[str]
    sites: pd.DataFrame | None = None   # distinct local maxima (incl. ambiguous alternatives)


def distinct_sites(res: pd.DataFrame, combo_cols, min_sep_m: float = 750.0, limit: int = 300) -> pd.DataFrame:
    """Greedy non-maximum suppression: best points that are at least
    ``min_sep_m`` apart for the same flight combination. Shows ambiguous
    alternatives (e.g. mirror positions when events are split by cuts)."""
    from .geo import haversine_m

    keep, kept_by_combo = [], {}
    for i, r in enumerate(res.itertuples(index=False)):
        key = tuple(getattr(r, c) for c in combo_cols)
        prev = kept_by_combo.setdefault(key, [])
        if all(haversine_m(r.lat, r.lon, a, b) >= min_sep_m for a, b in prev):
            prev.append((r.lat, r.lon))
            keep.append(i)
            if len(keep) >= limit:
                break
    out = res.iloc[keep].reset_index(drop=True)
    if len(out):
        out["rel_score"] = out["score"] / out["score"].iloc[0]
    return out


def run_search(flights: pd.DataFrame, events: list[Event], spec: Spectrogram, cfg: SearchConfig, dem,
               progress: Callable[[float, str], None] | None = None) -> SearchResult:
    log: list[str] = []
    stages: list[dict] = []

    def P(x, msg):
        if progress:
            progress(float(np.clip(x, 0, 1)), msg)

    def L(msg):
        log.append(msg)

    t_start = _time.time()
    pad = max(e.d_range[1] for e in events) / cfg.c + 600
    t_lo = cfg.t0_range[0] - pad
    t_hi = cfg.t0_range[1] + cfg.duration + pad
    P(0.0, "preparing tracks")
    tracks = prepare_tracks(flights, t_lo, t_hi, cfg.bbox, alt_offset_m=cfg.alt_offset_m,
                            smooth_s=cfg.smooth_s, max_gap_s=cfg.max_gap_s,
                            progress=lambda x: P(0.1 * x, "preparing tracks"))
    n_jets = sum(t.propulsion == "jet" for t in tracks.values())
    L(f"{len(tracks)} flight tracks in area/time window ({n_jets} known jets)")
    stages.append({"stage": "flights in window", "count": len(tracks)})

    grid = Grid(cfg.grid_m)
    cache: dict = {}
    cands = []
    for k, ev in enumerate(events):
        c = event_candidates(ev, tracks, cfg, grid, dem, cache,
                             progress=lambda x, k=k: P(0.1 + 0.3 * (k + x) / len(events),
                                                       f"candidates for {ev.name}"))
        L(f"{ev.name}: {len(c)} (cell, flight) candidates from {c.fid.nunique() if len(c) else 0} flights "
          f"matching speed {ev.v_range[0]:.0f}-{ev.v_range[1]:.0f} m/s and slant range "
          f"{ev.d_range[0]:.0f}-{ev.d_range[1]:.0f} m")
        stages.append({"stage": f"{ev.name}: speed + slant range", "count": int(c.key.nunique()) if len(c) else 0})
        cands.append(c)
        if c.empty:
            break

    if any(c.empty for c in cands) or len(cands) < len(events):
        bad = next((ev.name for ev, c in zip(events, cands) if c.empty), None)
        L(f"no candidates for '{bad}': no flight in the data matches its speed/distance/time. If it is a false "
          "detection, delete that event; otherwise widen its ranges, the time window or the area")
        empty = pd.DataFrame()
        return SearchResult(empty, empty, stages, cands, empty, tracks, log)

    segs = event_segments(events, cfg.cuts)
    if len(set(segs)) > 1:
        L(f"events span {len(set(segs))} uncut segments {segs}: time offsets are only enforced "
          "within a segment; each segment gets its own start time")
    surv = combine_events(cands, events, cfg.offset_tol_s, segs, cfg.cut_removed_s)
    stages.append({"stage": "time-offset filter" if len(events) > 1 else "time window", "count": int(surv.key.nunique()) if len(surv) else 0})
    L(f"{len(surv)} rows / {surv.key.nunique() if len(surv) else 0} cells after time-offset filter")

    if cfg.silence and len(surv):
        spans = cfg.clean_spans if cfg.clean_spans is not None else default_clean_spans(events, cfg.duration, cfg.silence_pad_s, cfg.cuts)
        L(f"silence spans (audio s): {[(round(a), round(b)) for a, b in spans]}")
        v = silence_violations(surv, len(events), tracks, cfg, spans, grid, dem, cache,
                               progress=lambda x: P(0.4 + 0.15 * x, "silence filter"))
        surv = surv.assign(violations=v)
        surv = surv[surv.violations <= cfg.max_violations].reset_index(drop=True)
        stages.append({"stage": "silence filter", "count": int(surv.key.nunique()) if len(surv) else 0})
        L(f"{len(surv)} rows / {surv.key.nunique() if len(surv) else 0} cells after silence filter")
    if surv.empty:
        empty = pd.DataFrame()
        return SearchResult(empty, empty, stages, cands, surv, tracks, log)

    # prioritise if too many: closeness of each event's d/speed to the range centre
    prio = np.zeros(len(surv))
    for k, ev in enumerate(events):
        dm, dw = np.mean(ev.d_range), (ev.d_range[1] - ev.d_range[0]) / 2
        vm, vw = np.mean(ev.v_range), (ev.v_range[1] - ev.v_range[0]) / 2
        prio += ((surv[f"d_{k}"] - dm) / dw) ** 2 + ((surv[f"speed_{k}"] - vm) / vw) ** 2
    surv = surv.assign(prio=prio).sort_values("prio").reset_index(drop=True)
    n_sub = int(round(cfg.grid_m / cfg.refine_m)) ** 2
    max_rows = max(1, cfg.max_score_points // max(n_sub, 1))
    if len(surv) > max_rows:
        L(f"scoring the {max_rows} most plausible of {len(surv)} rows (raise max_score_points to score all)")
        surv_s = surv.iloc[:max_rows]
    else:
        surv_s = surv

    scorer = Scorer(spec, events, cfg.c, cfg.dt_sweep_s, cfg.f0_sweep_frac, cfg.cuts)
    nr = int(round(cfg.grid_m / cfg.refine_m))
    offs = (np.arange(nr) - (nr - 1) / 2) * cfg.refine_m
    oe, on = [a.ravel() for a in np.meshgrid(offs, offs)]
    out = []
    lat_c, lon_c = grid.center(*grid.unkey(surv_s.key.values))
    for n, (row, la, lo) in enumerate(zip(surv_s.itertuples(index=False), lat_c, lon_c)):
        if n % 25 == 0:
            P(0.55 + 0.45 * n / max(len(surv_s), 1), f"Doppler scoring {n}/{len(surv_s)}")
        trs = [tracks[getattr(row, f"fid_{k}")] for k in range(len(events))]
        sl, so = offset_latlon(np.full(len(oe), la), np.full(len(oe), lo), oe, on)
        el = dem(sl, so)
        xyzs = lla_to_ecef(sl, so, el)
        for p in range(len(oe)):
            r = scorer.score(trs, xyzs[p])
            rec = {"lat": float(sl[p]), "lon": float(so[p]), "elev": float(el[p]), "score": r["score"],
                   "T0": r["T0"], "dt": r["dt"], "cell": int(row.key)}
            for sg, tv in r["T0_segs"].items():
                rec[f"T0_s{sg}"] = tv
            for k in range(len(events)):
                rec[f"score_{k}"] = r["event_scores"][k]
                rec[f"T0_{k}"] = r["T0_events"][k]
                rec[f"f0_{k}"] = r["f0_best"][k]
                rec[f"fid_{k}"] = trs[k].fid
                rec[f"callsign_{k}"] = trs[k].callsign
                rec[f"type_{k}"] = trs[k].typecode
                rec[f"d_{k}"] = float(getattr(row, f"d_{k}"))
                rec[f"speed_{k}"] = float(getattr(row, f"speed_{k}"))
            out.append(rec)
    res = pd.DataFrame(out).sort_values("score", ascending=False).reset_index(drop=True)
    res["start_utc"] = pd.to_datetime(res["T0"], unit="s", utc=True)
    sg_cols = sorted(int(c[4:]) for c in res.columns if c.startswith("T0_s"))
    for a, b in zip(sg_cols[:-1], sg_cols[1:]):
        res[f"removed_s{a}_s{b}"] = res[f"T0_s{b}"] - res[f"T0_s{a}"]   # time cut out between segments
    combo = [f"fid_{k}" for k in range(len(events))]
    best = res.drop_duplicates(subset=combo, keep="first").reset_index(drop=True)
    sites = distinct_sites(res, combo)
    stages.append({"stage": "Doppler-scored points", "count": len(res)})
    L(f"scored {len(res)} points across {len(best)} flight combinations in {_time.time() - t_start:.0f} s")
    P(1.0, "done")
    return SearchResult(res, best, stages, cands, surv, tracks, log, sites)


def focus_heatmap(tracks_per_event: list[Track], events: list[Event], spec: Spectrogram, center, radius_m,
                  step_m, dem, c=343.0, dt_sweep=4.0, f0_frac=0.03, progress=None, cuts=()) -> pd.DataFrame:
    """Score a fine grid around one location for a fixed flight combination."""
    scorer = Scorer(spec, events, c, dt_sweep, f0_frac, cuts)
    offs = np.arange(-radius_m, radius_m + 1e-9, step_m)
    oe, on = [a.ravel() for a in np.meshgrid(offs, offs)]
    keep = oe**2 + on**2 <= radius_m**2
    oe, on = oe[keep], on[keep]
    la, lo = offset_latlon(np.full(len(oe), center[0]), np.full(len(oe), center[1]), oe, on)
    el = dem(la, lo)
    xyz = lla_to_ecef(la, lo, el)
    rows = []
    for p in range(len(oe)):
        if progress and p % 50 == 0:
            progress(p / len(oe))
        r = scorer.score(tracks_per_event, xyz[p])
        rows.append({"lat": la[p], "lon": lo[p], "score": r["score"], "T0": r["T0"],
                     **{f"score_{k}": s for k, s in enumerate(r["event_scores"])}})
    return pd.DataFrame(rows)
