"""Doppler physics for an airborne source and a static ground observer.

Key subtlety (the "retarded time" problem): the frequency an observer hears at
reception time ``t`` depends on the source's radial velocity at the *emission*
time ``tau`` where ``t = tau + r(tau)/c``. Naively using the velocity at ``t``
produces a curve with the wrong shape, not merely a delayed one.

Frequency heard:  f(t) = f0 * c / (c + v_r(tau)),   v_r = dr/dtau  (> 0 receding)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares


def speed_of_sound(temp_c: float = 15.0) -> float:
    """Speed of sound in dry air (m/s)."""
    return 331.3 * np.sqrt(1.0 + temp_c / 273.15)


# ---------------------------------------------------------------------------
# Straight-line, constant-velocity model (used to *measure* events)
# ---------------------------------------------------------------------------

def linear_model(t, f0, v, d, t_cpa, c=343.0):
    """Received frequency for a source passing at constant speed ``v`` (m/s)
    with closest slant distance ``d`` (m). ``t_cpa`` is the *reception* time of
    the CPA sound (i.e. emission CPA time + d/c), in the same clock as ``t``.

    Solved analytically: with u = tau - tau_c and T = t - tau_c,
    (c^2 - v^2) u^2 - 2 c^2 T u + (c^2 T^2 - d^2) = 0.
    """
    t = np.asarray(t, dtype=float)
    v = min(abs(v), 0.95 * c)
    d = max(abs(d), 1.0)
    tau_c = t_cpa - d / c
    T = t - tau_c
    a = c * c - v * v
    disc = v * v * c * c * T * T + a * d * d
    u = (c * c * T - np.sqrt(disc)) / a
    r = c * (T - u)
    vr = v * v * u / np.maximum(r, 1e-6)
    return f0 * c / (c + vr)


@dataclass
class LinearFit:
    f0: float
    v: float
    d: float
    t_cpa: float
    c: float
    rms_hz: float
    n: int
    sigma: dict = field(default_factory=dict)

    def ranges(self, k: float = 3.0, min_rel_v: float = 0.12, min_rel_d: float = 0.35,
               min_t: float = 3.0) -> dict:
        """Generous measurement ranges (a search should never cut the truth out)."""
        sv = max(k * self.sigma.get("v", 0.0), min_rel_v * self.v)
        sd = max(k * self.sigma.get("d", 0.0), min_rel_d * self.d)
        st = max(k * self.sigma.get("t_cpa", 0.0), min_t)
        return {
            "v": (max(self.v - sv, 5.0), self.v + sv),
            "d": (max(self.d - sd, 30.0), self.d + sd),
            "t_cpa": (self.t_cpa - st, self.t_cpa + st),
        }

    def to_dict(self):
        return asdict(self)


def fit_linear(t, f, c=343.0, weights=None, x0=None, bounds_v=(10, 320), bounds_d=(30, 20000)) -> LinearFit:
    """Robust least-squares fit of the straight-line Doppler model to traced
    (time, frequency) points. Several starting points are tried."""
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    w = np.ones_like(t) if weights is None else np.asarray(weights, dtype=float)[ok]
    w = np.sqrt(np.clip(w, 1e-6, None))
    if len(t) < 5:
        raise ValueError("need at least 5 traced points to fit a Doppler curve")

    f_hi, f_lo = np.percentile(f, 95), np.percentile(f, 5)
    f_mid = 0.5 * (f_hi + f_lo)
    # f_hi/f_lo ~ (c+v)/(c-v)  ->  v estimate
    ratio = max(f_hi / max(f_lo, 1e-6), 1.0001)
    v_guess = float(np.clip(c * (ratio - 1) / (ratio + 1), 15, 300))
    # time where curve crosses its midpoint
    t_guess = float(t[np.argmin(np.abs(f - f_mid))])

    def resid(p):
        return w * (linear_model(t, p[0], p[1], p[2], p[3], c) - f)

    lo = [f.min() * 0.7, bounds_v[0], bounds_d[0], t.min() - 120]
    hi = [f.max() * 1.3, bounds_v[1], bounds_d[1], t.max() + 120]
    starts = [x0] if x0 is not None else []
    for d0 in (300, 800, 2000, 5000):
        for v0 in (v_guess, 0.7 * v_guess, 1.3 * v_guess):
            starts.append([f_mid, np.clip(v0, *bounds_v), d0, t_guess])
    best = None
    for s in starts:
        s = np.clip(np.asarray(s, dtype=float), np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        try:
            r = least_squares(resid, s, bounds=(lo, hi), loss="soft_l1",
                              f_scale=max(0.5, 0.01 * f_mid))
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    res = linear_model(t, *p, c) - f
    rms = float(np.sqrt(np.mean(res**2)))
    sigma = {}
    try:
        J = best.jac
        cov = np.linalg.pinv(J.T @ J) * (np.sum(best.fun**2) / max(len(t) - 4, 1))
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        sigma = {"f0": float(sd[0]), "v": float(sd[1]), "d": float(sd[2]), "t_cpa": float(sd[3])}
    except Exception:
        pass
    return LinearFit(float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(c), rms, int(len(t)), sigma)


# ---------------------------------------------------------------------------
# Full trajectory model (used to *score* candidate locations)
# ---------------------------------------------------------------------------

def resample_track(t, xyz, dt=1.0, smooth_s=4.0):
    """Resample an ECEF track onto a uniform clock and smooth it so finite
    difference velocities are not step-like. Returns (t_u, xyz_u, vel_u)."""
    t = np.asarray(t, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    order = np.argsort(t)
    t, xyz = t[order], xyz[order]
    keep = np.concatenate([[True], np.diff(t) > 1e-3])
    t, xyz = t[keep], xyz[keep]
    if len(t) < 2:
        return t, xyz, np.zeros_like(xyz)
    tu = np.arange(t[0], t[-1] + 1e-9, dt)
    xu = np.stack([np.interp(tu, t, xyz[:, k]) for k in range(3)], axis=1)
    if smooth_s > 0 and len(tu) > 5:
        xu = gaussian_filter1d(xu, sigma=smooth_s / dt, axis=0, mode="nearest")
    vel = np.gradient(xu, dt, axis=0)
    return tu, xu, vel


def track_received_curve(obs_xyz, t_u, xyz_u, vel_u, f0=1.0, c=343.0):
    """Model received frequency vs *reception* time for a real trajectory.

    Returns (t_recv, f_recv, r) sampled at every emission sample. ``t_recv`` is
    monotonic because |v_r| < c.
    """
    dvec = xyz_u - obs_xyz[None, :]
    r = np.linalg.norm(dvec, axis=1)
    vr = np.einsum("ij,ij->i", dvec, vel_u) / np.maximum(r, 1e-6)
    t_recv = t_u + r / c
    f = f0 * c / (c + vr)
    return t_recv, f, r


def cpa_info(obs_xyz, t_u, xyz_u, vel_u, c=343.0):
    """Closest point of approach: emission time, slant distance, speed, and
    the *delayed* CPA time (when the CPA sound reaches the observer)."""
    r = np.linalg.norm(xyz_u - obs_xyz[None, :], axis=1)
    k = int(np.argmin(r))
    # parabolic refinement
    tk, rk = t_u[k], r[k]
    if 0 < k < len(r) - 1:
        r0, r1, r2 = r[k - 1], r[k], r[k + 1]
        den = r0 - 2 * r1 + r2
        if den > 0:
            off = 0.5 * (r0 - r2) / den
            tk = t_u[k] + off * (t_u[1] - t_u[0])
            rk = r1 - 0.25 * (r0 - r2) * off
    spd = float(np.linalg.norm(vel_u[k]))
    return {"t_emit": float(tk), "d": float(rk), "speed": spd, "t_heard": float(tk + rk / c),
            "edge": k == 0 or k == len(r) - 1}
