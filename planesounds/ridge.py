"""Tonal ridge tracing (Viterbi / dynamic programming) and automatic event detection."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

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


def detect_candidates(s: Spectrogram, f_lo=40.0, f_hi=600.0, win_s=90.0, step_s=20.0,
                      min_drop_frac=0.08, min_strength=2.0, top=15):
    """Scan the recording for descending tonal ridges that look like a pass-by.

    Returns a list of dicts with a suggested selection box, sorted by score.
    This is a *helper* - always review suggestions on the spectrogram.
    """
    from .doppler import fit_linear

    out = []
    t_start, t_end = s.t[0], s.t[-1]
    t0 = t_start
    while t0 + 10 < t_end:
        t1 = min(t0 + win_s, t_end)
        sub = s.crop(t0, t1, f_lo, f_hi)
        if sub.z.size == 0:
            break
        # find strongest persistent line in this window
        prof = enhance_lines(sub.z, 9).mean(axis=1)
        fc = float(sub.f[int(np.argmax(prof))])
        band = (max(f_lo, fc * 0.75), min(f_hi, fc * 1.3))
        try:
            tt, ff, st = trace_ridge(s, t0, t1, *band)
            tt, ff, st = clean_trace(tt, ff, st, min_strength)
            if len(tt) > 20:
                drop = (np.percentile(ff, 95) - np.percentile(ff, 5)) / np.median(ff)
                if drop >= min_drop_frac:
                    fit = fit_linear(tt, ff)
                    if t0 <= fit.t_cpa <= t1 and fit.rms_hz < 0.03 * fit.f0:
                        out.append({"t0": t0, "t1": t1, "f_lo": band[0], "f_hi": band[1],
                                    "t_cpa": fit.t_cpa, "f0": fit.f0, "v": fit.v, "d": fit.d,
                                    "strength": float(np.median(st)), "rms": fit.rms_hz,
                                    "score": float(np.median(st) * len(tt) * drop)})
        except Exception:
            pass
        t0 += step_s
    # de-duplicate by CPA time
    out.sort(key=lambda d: -d["score"])
    kept = []
    for c in out:
        if all(abs(c["t_cpa"] - k["t_cpa"]) > 20 for k in kept):
            kept.append(c)
    return kept[:top]
