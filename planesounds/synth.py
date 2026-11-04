"""Synthetic scenario generator: fake flights + physically-correct audio.

Used by the test-suite and by the app's "demo" mode so the whole pipeline can
be validated end-to-end without real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import lla_to_ecef, offset_latlon

FT = 0.3048


def straight_flight(lat0, lon0, t_mid, heading_deg, speed_ms, alt_msl_m, half_len_s=900,
                    dt=5.0, icao="abc123", callsign="TEST1", typecode="C172", turn_deg_s=0.0,
                    accel=0.0):
    """Flight passing through (lat0, lon0) at time ``t_mid``. Optional constant
    turn-rate / acceleration to make the Doppler curve non-ideal."""
    ts = np.arange(-half_len_s, half_len_s + 1e-9, 0.5)
    hdg = np.radians(heading_deg + turn_deg_s * ts)
    spd = np.maximum(speed_ms + accel * ts, 10)
    # integrate path relative to the mid point
    de = np.cumsum(spd * np.sin(hdg)) * 0.5
    dn = np.cumsum(spd * np.cos(hdg)) * 0.5
    mid = len(ts) // 2
    de -= de[mid]
    dn -= dn[mid]
    lat, lon = offset_latlon(np.full_like(de, lat0), np.full_like(de, lon0), de, dn)
    sel = np.arange(0, len(ts), int(dt / 0.5))
    return pd.DataFrame({
        "icao24": icao, "callsign": callsign, "typecode": typecode,
        "time": t_mid + ts[sel], "lat": lat[sel], "lon": lon[sel],
        "alt_m": np.full(len(sel), alt_msl_m), "gs_ms": spd[sel], "on_ground": False,
    })


def render_audio(flights: list[pd.DataFrame], obs_lat, obs_lon, obs_elev, rec_start, duration,
                 f0s, sr=8000, c=343.0, harmonics=(1.0, 0.5, 0.3), noise=0.05, ref_amp=0.02,
                 ref_dist=1000.0, seed=0):
    """Render what a microphone at the observer would record."""
    rng = np.random.default_rng(seed)
    t = rec_start + np.arange(int(duration * sr)) / sr
    obs = lla_to_ecef(obs_lat, obs_lon, obs_elev)
    y = np.zeros_like(t)
    for fl, f0 in zip(flights, f0s):
        tau = np.arange(fl.time.min(), fl.time.max(), 0.05)
        xyz = np.stack([np.interp(tau, fl.time, k) for k in lla_to_ecef(fl.lat, fl.lon, fl.alt_m).T], 1)
        r = np.linalg.norm(xyz - obs, axis=1)
        t_recv = tau + r / c
        ok = (t >= t_recv[0]) & (t <= t_recv[-1])
        tau_t = np.interp(t[ok], t_recv, tau)
        r_t = np.interp(t[ok], t_recv, r)
        amp = ref_amp * ref_dist / np.maximum(r_t, 50)
        sig = np.zeros(ok.sum())
        for k, h in enumerate(harmonics, start=1):
            sig += h * np.sin(2 * np.pi * f0 * k * tau_t + rng.uniform(0, 6.28))
        y[ok] += amp * sig
    # brown-ish background noise (lots of LF energy, like real outdoor audio)
    w = rng.standard_normal(len(t))
    b = np.cumsum(w)
    b -= np.convolve(b, np.ones(sr) / sr, mode="same")
    b /= np.std(b) + 1e-9
    y += noise * (0.5 * w + b)
    return y.astype(np.float32)


def demo_scenario(seed=0):
    """Two prop flights heard 920 s apart + some distractor traffic."""
    rng = np.random.default_rng(seed)
    obs = (40.8735, -73.4140, 30.0)  # observer lat, lon, ground elev (m)
    t_rec = 1_757_366_000.0          # recording start (unix)
    flights = []
    # event 1: passes ~1.2 km horizontal at 600 m MSL, 95 m/s, gentle turn
    olat, olon = offset_latlon(obs[0], obs[1], 900, -800)
    flights.append(straight_flight(float(olat), float(olon), t_rec + 150, 70, 95, 1000 * FT * 2.2,
                                   icao="a1b2c3", callsign="N123AB", typecode="PA28", turn_deg_s=0.03))
    # event 2: 920 s later
    olat, olon = offset_latlon(obs[0], obs[1], -400, 1500)
    flights.append(straight_flight(float(olat), float(olon), t_rec + 150 + 920, 200, 75, 457,
                                   icao="d4e5f6", callsign="N456CD", typecode="C172", accel=0.02))
    f0s = [102.0, 88.0]
    distract = []
    for k in range(25):
        dlat, dlon = rng.uniform(-0.35, 0.35), rng.uniform(-0.45, 0.45)
        ty = rng.choice(["B738", "A320", "C172", "SR22", "PC12", "E75L"])
        jet = ty in ("B738", "A320", "E75L")
        distract.append(straight_flight(obs[0] + dlat, obs[1] + dlon, t_rec + rng.uniform(-1800, 3000),
                                        rng.uniform(0, 360), 210 if jet else rng.uniform(55, 110),
                                        8000 if jet else rng.uniform(400, 2500),
                                        icao=f"ff{k:04x}", callsign=f"X{k}", typecode=str(ty)))
    # drop distractors that would pass close to the observer during the recording
    from .geo import haversine_m
    keep = []
    for d in distract:
        dist = haversine_m(obs[0], obs[1], d.lat.values, d.lon.values)
        near = (dist < 6000) & (d.time.values > t_rec - 300) & (d.time.values < t_rec + 1500)
        if not near.any():
            keep.append(d)
    from .adsb import finalize
    all_fl = finalize(pd.concat(flights + keep, ignore_index=True))
    duration = 1300.0
    audio = render_audio(flights, obs[0], obs[1], obs[2], t_rec, duration, f0s)
    return {"obs": obs, "rec_start": t_rec, "duration": duration, "flights": all_fl,
            "audio": audio, "sr": 8000, "f0s": f0s, "event_flights": ["a1b2c3", "d4e5f6"]}
