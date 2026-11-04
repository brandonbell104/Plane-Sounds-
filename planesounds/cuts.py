"""Edit / cut detection.

YouTube videos are often edited. Across a cut the time offset between two
Doppler curves is meaningless, so the search must treat each uncut segment
as having its own unknown start time.

Two detectors:

* **Audio** - a cut almost always changes the *background*: the noise floor's
  level and spectral shape jump (different place, mic gain, wind, room).
  We compare a low-percentile ("noise floor", ignores speech/bursts) spectrum
  in a window just before vs. just after each instant. Digital-silence gaps
  and hard sample discontinuities are flagged too.
* **Video** - ffmpeg scene-change detection on a low-res copy of the video.
  A visual cut does *not* always mean an audio cut (b-roll over continuous
  sound), so each video cut also gets the audio novelty at that instant.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import percentile_filter


@dataclass
class Cut:
    t: float                  # audio time (s)
    source: str               # "audio", "silence", "video", "manual"
    strength: float           # detector score (z-like; higher = more certain)
    audio_novelty: float = float("nan")
    note: str = ""

    def to_dict(self):
        return asdict(self)


def _band_energies(x, sr, hop_s=0.05, win_s=0.1, n_bands=32, f_lo=40.0):
    nper = int(win_s * sr)
    hop = int(hop_s * sr)
    f, t, Z = signal.stft(x, fs=sr, nperseg=nper, noverlap=nper - hop, boundary=None, padded=False)
    p = np.abs(Z) ** 2
    edges = np.geomspace(f_lo, 0.48 * sr, n_bands + 1)
    E = np.stack([p[(f >= a) & (f < b)].mean(axis=0) if ((f >= a) & (f < b)).any() else np.zeros(p.shape[1])
                  for a, b in zip(edges[:-1], edges[1:])])
    return t, 10 * np.log10(E + 1e-12)


def audio_novelty(x: np.ndarray, sr: int, ctx_s: float = 2.0, hop_s: float = 0.05, floor_pct: float = 20):
    """Noise-floor change score per frame. Returns (t, novelty_db, level_db)."""
    t, E = _band_energies(x, sr, hop_s)
    W = max(3, int(ctx_s / hop_s))
    # rolling low percentile over W frames (centred), then shift to get before/after windows
    F = percentile_filter(E, floor_pct, size=(1, W), mode="nearest")
    half = W // 2
    before = np.roll(F, half + 1, axis=1)
    after = np.roll(F, -half, axis=1)
    d = after - before
    # spectral-shape change + overall level change (both in dB)
    shape = np.median(np.abs(d - np.median(d, axis=0, keepdims=True)), axis=0)
    level = np.abs(np.median(d, axis=0))
    nov = shape + 0.5 * level
    nov[: W] = 0
    nov[-W:] = 0
    return t, nov, E.mean(axis=0)


def _refine_changepoint(E, p, W):
    """Best split point near frame p: maximise the difference between the
    per-band medians on either side (robust to speech bursts)."""
    lo, hi = max(0, p - 2 * W), min(E.shape[1], p + 2 * W)
    best, arg = -1.0, p
    margin = max(2, W // 4)
    for s in range(lo + margin, hi - margin):
        a = np.median(E[:, max(lo, s - W):s], axis=1)
        b = np.median(E[:, s:min(hi, s + W)], axis=1)
        v = float(np.sum((a - b) ** 2))
        if v > best:
            best, arg = v, s
    return arg


def detect_audio_cuts(x: np.ndarray, sr: int, sensitivity: float = 1.0, ctx_s: float = 2.0,
                      min_gap_s: float = 3.0, min_db: float = 2.0) -> list[Cut]:
    """Find likely edit points from the audio alone.

    ``sensitivity`` > 1 finds more (weaker) cuts, < 1 fewer.
    """
    hop = 0.05
    t, nov, lvl = audio_novelty(x, sr, ctx_s, hop)
    cuts: list[Cut] = []
    if len(nov) > 10:
        med = np.median(nov)
        mad = np.median(np.abs(nov - med)) * 1.4826 + 1e-6
        z = (nov - med) / mad
        thr = 8.0 / max(sensitivity, 1e-3)
        peaks, props = signal.find_peaks(z, height=thr, distance=max(1, int(min_gap_s / hop)))
        _, E = _band_energies(x, sr, hop)
        W = max(3, int(ctx_s / hop))
        for p in peaks:
            if nov[p] < min_db / max(sensitivity, 1e-3):
                continue
            q = _refine_changepoint(E, p, W)
            cuts.append(Cut(float(t[q]), "audio", float(z[p]), float(nov[p])))

    # digital silence / hard mutes (common at edit points with fades)
    frame = int(0.02 * sr)
    if frame > 0 and len(x) > frame:
        n = len(x) // frame
        rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-20)
        db = 20 * np.log10(rms)
        quiet = db < -75
        if quiet.any():
            edges = np.flatnonzero(np.diff(np.concatenate([[0], quiet.astype(int), [0]])))
            for a, b in zip(edges[::2], edges[1::2]):
                if (b - a) * 0.02 >= 0.06 and a > 0 and b < n:
                    tc = (a + b) / 2 * 0.02
                    if all(abs(tc - c.t) > min_gap_s for c in cuts):
                        cuts.append(Cut(float(tc), "silence", 99.0, note=f"{(b - a) * 20:.0f} ms of digital silence"))

    # hard sample discontinuity (spliced without crossfade): huge 1-sample jump vs local scale
    dx = np.abs(np.diff(x))
    if len(dx) > sr:
        scale = np.convolve(dx, np.ones(sr // 10) / (sr // 10), mode="same") + 1e-6
        r = dx / scale
        idx = np.flatnonzero(r > 40)
        for i in idx[:: max(1, len(idx) // 50)] if len(idx) else []:
            tc = i / sr
            if all(abs(tc - c.t) > min_gap_s for c in cuts):
                cuts.append(Cut(float(tc), "audio", float(r[i] / 4), note="sample discontinuity"))
    cuts.sort(key=lambda c: c.t)
    return cuts


def detect_video_cuts(video_path: str | Path, threshold: float = 0.35, t_offset: float = 0.0) -> list[Cut]:
    """Scene changes via ffmpeg's scene score. ``t_offset`` is subtracted so
    times line up with a trimmed audio clip."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(video_path), "-an", "-vf",
           f"scale=160:-2,select='gt(scene,{threshold})',metadata=print", "-f", "null", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if out.returncode != 0 and "pts_time" not in out.stderr:
        raise RuntimeError(f"ffmpeg scene detection failed: {out.stderr[-400:]}")
    cuts, t_cur = [], None
    for ln in out.stderr.splitlines():
        m = re.search(r"pts_time:\s*([0-9.]+)", ln)
        if m:
            t_cur = float(m.group(1))
            continue
        m = re.search(r"lavfi\.scene_score=([0-9.]+)", ln)
        if m and t_cur is not None:
            cuts.append(Cut(t_cur - t_offset, "video", float(m.group(1)) * 10))
            t_cur = None
    return [c for c in cuts if c.t >= 0]


def annotate_with_audio(cuts: list[Cut], x: np.ndarray, sr: int, ctx_s: float = 2.0) -> list[Cut]:
    """Attach the audio novelty around each (e.g. video) cut so the user can
    tell 'picture changed but sound continues' from a real audio cut."""
    if not cuts:
        return cuts
    t, nov, _ = audio_novelty(x, sr, ctx_s)
    for c in cuts:
        m = np.abs(t - c.t) <= 0.75
        c.audio_novelty = float(nov[m].max()) if m.any() else float("nan")
        if c.source == "video":
            c.note = "audio changes too" if c.audio_novelty >= 2.0 else "audio may be continuous (b-roll?)"
    return cuts


def merge_cuts(*lists: list[Cut], tol: float = 1.0) -> list[Cut]:
    allc = sorted([c for lst in lists for c in lst], key=lambda c: c.t)
    out: list[Cut] = []
    for c in allc:
        if out and abs(c.t - out[-1].t) <= tol:
            if c.source not in out[-1].source:
                out[-1].source += "+" + c.source
            out[-1].strength = max(out[-1].strength, c.strength)
            continue
        out.append(Cut(**c.to_dict()))
    return out


# ---------------------------------------------------------------------------
# segments
# ---------------------------------------------------------------------------

def segment_bounds(cut_times, duration):
    edges = [0.0] + sorted(float(c) for c in cut_times if 0 < c < duration) + [float(duration)]
    return list(zip(edges[:-1], edges[1:]))


def segment_of(t, cut_times) -> int:
    return int(np.searchsorted(np.sort(np.asarray(list(cut_times), dtype=float)), t, side="right"))
