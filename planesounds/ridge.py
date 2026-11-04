"""Tonal ridge tracing (Viterbi / dynamic programming) and automatic event detection."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .spectro import Spectrogram, enhance_lines


def trace_ridge(s: Spectrogram, t0: float, t1: float, f_lo: float, f_hi: float,
                max_slope_hz_s: float = 3.0, jump_penalty: float = 0.6,
                smooth_frames: int = 5, monotonic: bool = True):
    """Find the brightest continuous path through a time/frequency box.

    ``monotonic`` forbids upward frequency steps (a single pass-by Doppler
    curve only ever decreases), which makes tracing very faint curves robust.
    Returns (t, f, strength) arrays.
    """
    sub = s.crop(t0, t1, f_lo, f_hi)
    if sub.z.shape[0] < 3 or sub.z.shape[1] < 3:
        raise ValueError("selection too small")
    img = enhance_lines(sub.z, smooth_frames)
    nf, nt = img.shape
    max_jump = max(1, int(np.ceil(max_slope_hz_s * sub.dt / sub.df)))
    jumps = np.arange(-max_jump, (0 if monotonic else max_jump) + 1)
    score = img[:, 0].copy()
    back = np.zeros((nf, nt), dtype=np.int32)
    idx = np.arange(nf)
    for k in range(1, nt):
        best = np.full(nf, -np.inf)
        arg = np.zeros(nf, dtype=np.int32)
        for j in jumps:
            # coming to bin i from bin i - j  (j<0 means frequency went down)
            src = idx - j
            valid = (src >= 0) & (src < nf)
            cand = np.full(nf, -np.inf)
            cand[valid] = score[src[valid]] - jump_penalty * abs(j)
            upd = cand > best
            best[upd] = cand[upd]
            arg[upd] = src[upd]
        score = best + img[:, k]
        back[:, k] = arg
    path = np.zeros(nt, dtype=int)
    path[-1] = int(np.argmax(score))
    for k in range(nt - 1, 0, -1):
        path[k - 1] = back[path[k], k]
    strength = sub.z[path, np.arange(nt)]
    # sub-bin refinement by parabolic interpolation on the raw image
    fr = sub.f[path].astype(float)
    for k in range(nt):
        i = path[k]
        if 0 < i < nf - 1:
            a, b, c = img[i - 1, k], img[i, k], img[i + 1, k]
            den = a - 2 * b + c
            if den < 0:
                fr[k] += 0.5 * (a - c) / den * sub.df
    return sub.t.copy(), fr, strength


def clean_trace(t, f, strength, min_strength: float = 1.5, smooth: int = 3):
    """Drop weak frames (where the ridge is lost in noise)."""
    s = uniform_filter1d(strength.astype(float), max(1, smooth))
    keep = s >= min_strength
    return t[keep], f[keep], strength[keep]


def _segments(t_start, t_end, cuts):
    edges = [t_start] + sorted(c for c in cuts if t_start < c < t_end) + [t_end]
    return list(zip(edges[:-1], edges[1:]))


def detect_candidates(s: Spectrogram, cuts=(), f_lo=40.0, f_hi=600.0, win_s=120.0, step_s=20.0,
                      min_drop_frac=0.04, min_strength=1.5, lines_per_window=3, cut_margin_s=1.0,
                      top=40, group_s=30.0, progress=None):
    """Scan the whole recording for descending tonal ridges (aircraft pass-bys).

    * Never traces across a cut: each uncut segment is scanned on its own.
    * Several candidate lines per window (fundamental + harmonics, or two
      aircraft) are traced.
    * Lines with the same CPA time are grouped as one aircraft; the strongest
      line is kept and the others are listed as its extra lines/harmonics.

    Returns dicts (sorted by score) including the trace itself, so a detection
    can be saved as an event without re-tracing. Always review them visually.
    """
    from .doppler import fit_linear

    raw = []
    segs = [(a + cut_margin_s, b - cut_margin_s) for a, b in _segments(float(s.t[0]), float(s.t[-1]), cuts)]
    segs = [(a, b) for a, b in segs if b - a >= 25]
    jobs = []
    for si, (a, b) in enumerate(segs):
        w = min(win_s, b - a)
        t0 = a
        while True:
            jobs.append((si, t0, min(t0 + w, b)))
            if t0 + w >= b:
                break
            t0 += step_s
    for n, (si, t0, t1) in enumerate(jobs):
        if progress and n % 5 == 0:
            progress(n / max(len(jobs), 1))
        sub = s.crop(t0, t1, f_lo, f_hi)
        if sub.z.shape[0] < 5 or sub.z.shape[1] < 10:
            continue
        prof = np.percentile(enhance_lines(sub.z, 9), 75, axis=1)
        pk, pr = find_peaks(prof, prominence=0.3)
        if not len(pk):
            continue
        order = np.argsort(-pr["prominences"])[:lines_per_window]
        for i in pk[order]:
            fc = float(sub.f[i])
            # the profile peak may be the approach plateau, the recede plateau or the middle:
            # try each hypothesis and keep the best fit (a fast plane drops ~1.5-1.9x in pitch)
            best = None
            for lo_k, hi_k in ((1 / 1.9, 1.08), (0.92, 1.9), (1 / 1.4, 1.4)):
                band = (max(f_lo, fc * lo_k), min(f_hi, fc * hi_k))
                try:
                    tt, ff, st = trace_ridge(s, t0, t1, *band)
                    tt, ff, st = clean_trace(tt, ff, st, min_strength)
                    if len(tt) < 0.3 * (t1 - t0) / s.dt or len(tt) < 20:
                        continue
                    drop = np.percentile(ff, 95) / max(np.percentile(ff, 5), 1e-6) - 1
                    if drop < min_drop_frac:
                        continue
                    fit = fit_linear(tt, ff, weights=st)
                    margin = 0.15 * (t1 - t0)
                    if not (t0 + margin <= fit.t_cpa <= t1 - margin) or fit.rms_hz > 0.01 * fit.f0:
                        continue
                    cov = len(tt) * s.dt / (t1 - t0)
                    q = float(np.median(st) * cov * min(drop, 0.6) * 100 / (1 + 300 * fit.rms_hz / fit.f0))
                    if best is None or q > best["score"]:
                        best = {"segment": si, "t0": t0, "t1": t1, "f_lo": band[0], "f_hi": band[1],
                                "t_cpa": fit.t_cpa, "f0": fit.f0, "v": fit.v, "d": fit.d, "rms": fit.rms_hz,
                                "strength": float(np.median(st)), "coverage": cov, "score": q,
                                "trace_t": tt, "trace_f": ff, "trace_s": st}
                except Exception:
                    continue
            if best is not None:
                raw.append(best)
    # group by aircraft (same CPA time), keep the best line, remember the others
    raw.sort(key=lambda d: -d["score"])
    groups: list[dict] = []
    for c in raw:
        g = next((g for g in groups if g["segment"] == c["segment"]
                  and abs(g["t_cpa"] - c["t_cpa"]) < group_s), None)
        if g is None:
            c["lines"] = [round(c["f0"], 1)]
            groups.append(c)
            continue
        # another line of the same pass-by: keep it only if it is a clean harmonic
        # (integer ratio to the best line, or the best line is a harmonic of it)
        r = c["f0"] / g["f0"]
        k = max(round(r), 1) if r >= 1 else 1 / max(round(1 / r), 1)
        if abs(r / k - 1) < 0.03 and abs(c["t_cpa"] - g["t_cpa"]) < 6 \
                and all(abs(c["f0"] / f - 1) > 0.03 for f in g["lines"]):
            g["lines"].append(round(c["f0"], 1))
    for g in groups:
        # quality flag used by the UI to pre-select good detections
        g["good"] = bool(g["rms"] < 0.004 * g["f0"] and g["v"] >= 30 and g["strength"] >= 2.0
                         and g["coverage"] >= 0.4)
        base = g["f0"]
        g["harmonics_hint"] = sorted({round(f / base, 2) for f in g["lines"]})
        g["n_lines"] = len(g["lines"])
    groups.sort(key=lambda d: d["t_cpa"])
    return sorted(groups, key=lambda d: -d["score"])[:top]
