"""Audio acquisition: YouTube (yt-dlp) or local files, decoded with ffmpeg."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

CACHE_DIR = Path.home() / ".planesounds" / "cache"


@dataclass
class Audio:
    samples: np.ndarray  # float32 mono, [-1, 1]
    sr: int
    source: str = ""
    title: str = ""
    upload_date: str = ""  # YYYYMMDD from YouTube metadata (if any)
    meta: dict | None = None
    start_offset: float = 0.0   # seconds trimmed from the start of the original file
    media_path: str = ""        # local file the audio was decoded from

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr

    def segment(self, t0: float, t1: float) -> np.ndarray:
        a, b = int(max(t0, 0) * self.sr), int(min(t1, self.duration) * self.sr)
        return self.samples[a:b]


def _require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install it (https://ffmpeg.org/download.html, "
                           "or `winget install ffmpeg` on Windows) and restart.")


def decode_file(path: str | Path, sr: int = 8000, start: float | None = None,
                end: float | None = None) -> Audio:
    """Decode any audio/video file to mono float32 at ``sr`` Hz using ffmpeg.

    No loudness normalisation, no noise reduction: faint tones must survive.
    """
    _require_ffmpeg()
    path = Path(path)
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if end is not None:
        cmd += ["-t", str(end - (start or 0))]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {out.stderr.decode(errors='ignore')[:500]}")
    x = np.frombuffer(out.stdout, dtype=np.float32).copy()
    return Audio(x, sr, source=str(path), title=path.stem, start_offset=float(start or 0.0),
                 media_path=str(path))


def download_youtube(url: str, cache_dir: Path = CACHE_DIR, progress=None) -> tuple[Path, dict]:
    """Download the best audio stream of a YouTube (or any yt-dlp supported)
    URL. Returns (file path, info dict). Cached by URL."""
    try:
        import yt_dlp
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("yt-dlp is not installed: pip install yt-dlp") from e
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(url.strip().encode()).hexdigest()[:16]
    meta_path = cache_dir / f"{key}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        p = Path(meta.get("_file", ""))
        if p.exists():
            return p, meta

    def hook(d):
        if progress and d.get("status") == "downloading":
            tot = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if tot:
                progress(min(d.get("downloaded_bytes", 0) / tot, 1.0))

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(cache_dir / f"{key}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        fname = Path(ydl.prepare_filename(info))
    if not fname.exists():  # extension can change after post-processing
        cands = sorted(cache_dir.glob(f"{key}.*"))
        cands = [c for c in cands if c.suffix != ".json"]
        if not cands:
            raise RuntimeError("download finished but file not found")
        fname = cands[0]
    keep = {k: info.get(k) for k in ("id", "title", "uploader", "upload_date", "timestamp",
                                      "release_timestamp", "duration", "description",
                                      "webpage_url", "location", "is_live", "was_live")}
    keep["_file"] = str(fname)
    meta_path.write_text(json.dumps(keep, default=str))
    return fname, keep


def download_video_lowres(url: str, cache_dir: Path = CACHE_DIR, progress=None) -> Path:
    """Low-resolution, video-only copy (for scene-cut detection). Cached."""
    import yt_dlp

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(url.strip().encode()).hexdigest()[:16] + "_video"
    have = [c for c in cache_dir.glob(f"{key}.*") if c.suffix not in (".json", ".part")]
    if have:
        return have[0]

    def hook(d):
        if progress and d.get("status") == "downloading":
            tot = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if tot:
                progress(min(d.get("downloaded_bytes", 0) / tot, 1.0))

    opts = {"format": "worstvideo[height>=144]/bv*[height<=360]/worst", "outtmpl": str(cache_dir / f"{key}.%(ext)s"),
            "quiet": True, "no_warnings": True, "noplaylist": True, "progress_hooks": [hook]}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        f = Path(ydl.prepare_filename(info))
    if not f.exists():
        f = next(c for c in cache_dir.glob(f"{key}.*") if c.suffix not in (".json", ".part"))
    return f


def load_source(src: str, sr: int = 8000, start=None, end=None, progress=None) -> Audio:
    """``src`` may be a URL or a local path."""
    if src.startswith(("http://", "https://", "www.", "youtu")):
        path, meta = download_youtube(src if "://" in src else "https://" + src, progress=progress)
        a = decode_file(path, sr, start, end)
        a.source, a.title, a.upload_date, a.meta = src, meta.get("title") or "", meta.get("upload_date") or "", meta
        return a
    return decode_file(src, sr, start, end)


def bandpass(x: np.ndarray, sr: int, f_lo: float, f_hi: float, order: int = 6) -> np.ndarray:
    f_hi = min(f_hi, 0.49 * sr)
    sos = signal.butter(order, [max(f_lo, 1.0), f_hi], btype="band", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def make_listenable(x: np.ndarray, sr: int, f_lo: float, f_hi: float, gain_db: float = 30.0,
                    pitch_shift: float = 1.0) -> tuple[np.ndarray, int]:
    """Band-pass + boost a clip so a faint Doppler tone becomes audible.
    ``pitch_shift`` > 1 plays it back faster/higher (helps with sub-100 Hz tones)."""
    y = bandpass(x, sr, f_lo, f_hi)
    y = y * (10 ** (gain_db / 20))
    peak = np.max(np.abs(y)) or 1.0
    if peak > 0.99:
        y = np.tanh(y / peak * 2.0) * 0.95  # soft limit
    return y.astype(np.float32), int(sr * pitch_shift)


def write_wav(path, x, sr):
    wavfile.write(str(path), sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))
