"""High-resolution spectrograms tuned for faint, low-frequency tonal curves."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter


@dataclass
class Spectrogram:
    t: np.ndarray       # frame centre times (s, audio clock)
    f: np.ndarray       # bin frequencies (Hz)
    db: np.ndarray      # raw power in dB, shape (len(f), len(t))
    z: np.ndarray       # whitened / normalised "brightness" (>=0), same shape

    @property
    def dt(self):
        return float(self.t[1] - self.t[0]) if len(self.t) > 1 else 1.0

    @property
    def df(self):
        return float(self.f[1] - self.f[0]) if len(self.f) > 1 else 1.0

    def crop(self, t0=None, t1=None, f0=None, f1=None) -> "Spectrogram":
        ti = np.ones(len(self.t), bool) if t0 is None else (self.t >= t0) & (self.t <= t1)
        fi = np.ones(len(self.f), bool) if f0 is None else (self.f >= f0) & (self.f <= f1)
        return Spectrogram(self.t[ti], self.f[fi], self.db[np.ix_(fi, ti)], self.z[np.ix_(fi, ti)])

    def sample(self, t, f, field="z"):
        """Bilinear sample of the image at arbitrary (t, f) points (NaN outside)."""
        img = getattr(self, field)
        t = np.asarray(t, dtype=float)
        f = np.asarray(f, dtype=float)
        ti = (t - self.t[0]) / self.dt
        fi = (f - self.f[0]) / self.df
        ok = (ti >= 0) & (ti <= len(self.t) - 1) & (fi >= 0) & (fi <= len(self.f) - 1)
        out = np.full(t.shape, np.nan)
        if not ok.any():
            return out
        ti, fi = ti[ok], fi[ok]
        t0 = np.minimum(np.floor(ti).astype(int), len(self.t) - 2)
        f0 = np.minimum(np.floor(fi).astype(int), len(self.f) - 2)
        at, af = ti - t0, fi - f0
        v = (img[f0, t0] * (1 - at) * (1 - af) + img[f0, t0 + 1] * at * (1 - af)
             + img[f0 + 1, t0] * (1 - at) * af + img[f0 + 1, t0 + 1] * at * af)
        out[ok] = v
        return out

    def max_near(self, t, f, df_hz=1.0, field="z"):
        """Max brightness within +/- df_hz of each (t, f) - tolerant of small model error."""
        offs = np.linspace(-df_hz, df_hz, 5)
        vals = np.stack([self.sample(t, np.asarray(f) + o, field) for o in offs])
        with np.errstate(all="ignore"):
            return np.nanmax(vals, axis=0)


def compute(x: np.ndarray, sr: int, f_max: float = 1000.0, f_min: float = 0.0,
            win_s: float = 1.0, hop_s: float = 0.25, whiten: str = "freq",
            t_offset: float = 0.0, oversample: int = 2) -> Spectrogram:
    """STFT power spectrogram.

    * ``win_s`` controls frequency resolution (1 s -> 1 Hz bins). Doppler tones
      from propellers are narrow, so long windows dramatically improve SNR.
    * ``whiten='freq'`` subtracts each frequency's median level (kills constant
      hum, mains lines and the steep low-frequency noise slope); ``'both'``
      additionally removes broadband clicks/transients per frame.
    """
    nper = max(16, int(round(win_s * sr)))
    hop = max(1, int(round(hop_s * sr)))
    nfft = int(2 ** np.ceil(np.log2(nper * oversample)))
    f, t, Z = signal.stft(x, fs=sr, window="hann", nperseg=nper, noverlap=nper - hop,
                          nfft=nfft, boundary=None, padded=False)
    band = (f >= f_min) & (f <= f_max)
    p = np.abs(Z[band]) ** 2
    db = 10 * np.log10(p + 1e-14)
    z = whiten_db(db, whiten)
    return Spectrogram(t + t_offset, f[band], db.astype(np.float32), z.astype(np.float32))


def whiten_db(db: np.ndarray, mode: str = "freq") -> np.ndarray:
    z = db.copy()
    if mode in ("freq", "both"):
        z -= np.median(z, axis=1, keepdims=True)
    if mode in ("time", "both"):
        # remove broadband per-frame level using a local spectral median
        z -= median_filter(z, size=(31, 1), mode="nearest")
    # robust scale -> roughly "sigmas above the noise floor"
    mad = np.median(np.abs(z - np.median(z))) * 1.4826 + 1e-9
    z = (z - np.median(z)) / mad
    return np.clip(z, 0, None)


def enhance_lines(z: np.ndarray, t_len: int = 5) -> np.ndarray:
    """Smooth along time to favour slowly-varying tones over noise speckle."""
    return uniform_filter(z, size=(1, t_len), mode="nearest")


def downsample_for_display(s: Spectrogram, max_t=1500, max_f=600, field="z"):
    img = getattr(s, field)
    ft = max(1, int(np.ceil(img.shape[1] / max_t)))
    ff = max(1, int(np.ceil(img.shape[0] / max_f)))
    nt, nf = img.shape[1] // ft, img.shape[0] // ff
    if nt == 0 or nf == 0:
        return s.t, s.f, img
    im = img[: nf * ff, : nt * ft].reshape(nf, ff, nt, ft).max(axis=(1, 3))
    return s.t[: nt * ft : ft], s.f[: nf * ff : ff], im
