"""End-to-end test on a synthetic scenario: the search must find the observer."""
import numpy as np
import pandas as pd

from planesounds import adsb, doppler, ridge, search, spectro, synth
from planesounds.dem import ConstantDEM
from planesounds.geo import Grid, haversine_m


def _measure(spec, t0, t1, flo, fhi, name):
    t, f, st = ridge.trace_ridge(spec, t0, t1, flo, fhi)
    t, f, st = ridge.clean_trace(t, f, st)
    fit = doppler.fit_linear(t, f)
    rg = fit.ranges()
    return search.Event(name=name, t_cpa=fit.t_cpa, f0=fit.f0, v_range=rg["v"], d_range=rg["d"],
                        t_window=(fit.t_cpa - 60, fit.t_cpa + 60), f_band=(flo, fhi),
                        trace_t=list(t), trace_f=list(f))


def test_linear_model_matches_track_model():
    c = 343.0
    t = np.arange(0, 120, 0.5)
    tu = np.arange(-300, 300, 1.0)
    xyz = np.stack([80 * tu, np.full_like(tu, 540.0), np.full_like(tu, 720.0)], 1)
    vel = np.gradient(xyz, 1.0, axis=0)
    tr, fr, _ = doppler.track_received_curve(np.zeros(3), tu, xyz, vel, 110, c)
    lin = doppler.linear_model(t, 110, 80, 900, 900 / c, c)
    assert np.max(np.abs(np.interp(t, tr, fr) - lin)) < 0.1


def test_fit_recovers_parameters():
    t = np.arange(0, 120, 0.5)
    f = doppler.linear_model(t, 110, 80, 900, 60) + np.random.default_rng(1).normal(0, 0.3, len(t))
    fit = doppler.fit_linear(t, f)
    assert abs(fit.v - 80) < 3 and abs(fit.d - 900) < 60 and abs(fit.t_cpa - 60) < 1


def test_grid_keys_roundtrip():
    g = Grid(200)
    i = np.array([-5, 3, 22000]); j = np.array([-7, 9, -31000])
    ii, jj = g.unkey(g.key(i, j))
    assert (ii == i).all() and (jj == j).all()


def test_normalize_opensky_like():
    raw = pd.DataFrame({"time": [1, 2, 3], "icao24": ["ABC"] * 3, "lat": [1, 1.1, 1.2], "lon": [2, 2, 2],
                        "geoaltitude": [500, 510, 520], "velocity": [50, 50, 50], "callsign": ["X1 "] * 3,
                        "onground": [False] * 3})
    df = adsb.normalize(raw)
    assert df.alt_m.iloc[0] == 500 and df.icao24.iloc[0] == "abc" and df.flight_id.nunique() == 1


def test_end_to_end_search_finds_observer():
    sc = synth.demo_scenario()
    spec = spectro.compute(sc["audio"], sc["sr"], f_max=400, win_s=1.0, hop_s=0.25)
    e1 = _measure(spec, 60, 260, 60, 140, "event 1")
    e2 = _measure(spec, 980, 1180, 50, 120, "event 2")
    obs_lat, obs_lon, obs_elev = sc["obs"]
    cfg = search.SearchConfig(
        bbox=(obs_lat - 0.15, obs_lat + 0.15, obs_lon - 0.2, obs_lon + 0.2),
        t0_range=(sc["rec_start"] - 3 * 3600, sc["rec_start"] + 3 * 3600),
        duration=sc["duration"], max_score_points=6000,
    )
    res = search.run_search(sc["flights"], [e1, e2], spec, cfg, ConstantDEM(obs_elev))
    assert not res.results.empty, res.log
    top = res.results.iloc[0]
    err = haversine_m(obs_lat, obs_lon, top.lat, top.lon)
    assert err < 400, (err, res.log)
    assert abs(top.T0 - sc["rec_start"]) < 3
    assert top.fid_0.startswith("a1b2c3") and top.fid_1.startswith("d4e5f6")


def test_readsb_trace_parsing(tmp_path):
    import gzip, json, tarfile
    obj = {"icao": "a1b2c3", "t": "C172", "r": "N123AB", "timestamp": 1757366000.0,
           "trace": [[0, 40.0, -73.0, 2000, 100, 90, 0, 0, {"flight": "N123AB  "}, "adsb_icao", 2100],
                     [5, 40.001, -73.0, 2010, 100, 90, 0, 0, None, "adsb_icao", None],
                     [10, 40.002, -73.0, "ground", 0, 90, 0, 0, None, "adsb_icao", None],
                     [15, 45.0, -73.0, 2000, 100, 90, 0, 0, None, "adsb_icao", None]]}
    d = tmp_path / "traces" / "c3"
    d.mkdir(parents=True)
    (d / "trace_full_a1b2c3.json").write_bytes(gzip.compress(json.dumps(obj).encode()))
    df = adsb.load_readsb_traces([tmp_path], bbox=(39, 41, -74, -72))
    assert len(df) == 3 and df.typecode.iloc[0] == "C172" and df.callsign.iloc[0] == "N123AB"
    assert abs(df.alt_m.iloc[0] - 2100 * 0.3048) < 1e-6 and abs(df.alt_m.iloc[1] - 2010 * 0.3048) < 1e-6
    assert df.on_ground.iloc[2] and df.propulsion.iloc[0] == "prop"
    tar = tmp_path / "day.tar"
    with tarfile.open(tar, "w") as tf:
        tf.add(d / "trace_full_a1b2c3.json", arcname="./traces/c3/trace_full_a1b2c3.json")
    df2 = adsb.load_readsb_traces([tar], bbox=(39, 41, -74, -72))
    assert len(df2) == 3
