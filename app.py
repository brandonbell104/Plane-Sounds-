"""PlaneSounds - Doppler geolocation workbench (run: streamlit run app.py)."""
from __future__ import annotations

import datetime as dt
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from planesounds import adsb, audio as au, doppler, ridge, search, spectro, synth
from planesounds.dem import ConstantDEM, SRTMDEM
from planesounds.geo import haversine_m, lla_to_ecef

st.set_page_config(page_title="PlaneSounds", page_icon="✈️", layout="wide")
pio.templates.default = "plotly_dark"
S = st.session_state
for k, v in {"audio": None, "events": [], "sel": None, "trace": None, "fit": None, "flights": None,
             "result": None, "spec_cache": {}, "area": None, "demo": None, "heat": None}.items():
    S.setdefault(k, v)

UTC = dt.timezone.utc
# explicit rgb() stops: the Streamlit frontend re-maps some named/hex scale colours
SPEC_CMAP = [[i / 8, f"rgb({r},{g},{b})"] for i, (r, g, b) in enumerate([
    (2, 2, 8), (20, 14, 54), (59, 15, 112), (100, 26, 128), (140, 41, 129),
    (222, 73, 104), (254, 159, 109), (254, 207, 146), (252, 253, 191)])]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_spec(win_s, hop_s, f_max, whiten, t0=None, t1=None, f_min=0.0):
    a = S.audio
    key = (id(a), round(win_s, 3), round(hop_s, 3), round(f_max), whiten,
           None if t0 is None else round(t0, 1), None if t1 is None else round(t1, 1), round(f_min))
    cache = S.spec_cache
    if key not in cache:
        if len(cache) > 12:
            cache.clear()
        if t0 is None:
            x, off = a.samples, 0.0
        else:
            pad = win_s
            lo = max(0.0, t0 - pad)
            x, off = a.segment(lo, t1 + pad), lo
        cache[key] = spectro.compute(x, a.sr, f_max=f_max, f_min=f_min, win_s=win_s, hop_s=hop_s,
                                     whiten=whiten, t_offset=off)
    return cache[key]


def spec_figure(s: spectro.Spectrogram, title="", zmax=None, overlays=(), height=460, field="z",
                selectable=False):
    tt, ff, img = spectro.downsample_for_display(s, 1400, 500, field)
    if zmax is None:
        zmax = float(np.percentile(img, 99.5)) if img.size else 1
    fig = go.Figure(go.Heatmap(x=tt, y=ff, z=img, colorscale=SPEC_CMAP, zmin=0 if field == "z" else None,
                               zmax=zmax, colorbar=dict(title="σ" if field == "z" else "dB"),
                               hovertemplate="t=%{x:.2f}s<br>f=%{y:.2f}Hz<br>%{z:.2f}<extra></extra>"))
    for ov in overlays:
        fig.add_trace(ov)
    fig.update_layout(title=title, height=height, margin=dict(l=50, r=10, t=40, b=40),
                      xaxis_title="audio time (s)", yaxis_title="frequency (Hz)",
                      dragmode="select" if selectable else "zoom", showlegend=bool(overlays))
    return fig


def line(x, y, name, color, width=2, dash=None, mode="lines"):
    return go.Scatter(x=x, y=y, name=name, mode=mode, line=dict(color=color, width=width, dash=dash),
                      marker=dict(size=4, color=color))


def unix(d: dt.date, t: dt.time):
    return dt.datetime.combine(d, t, tzinfo=UTC).timestamp()


def fmt_utc(x):
    return dt.datetime.fromtimestamp(float(x), UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def make_dem():
    if S.get("dem_mode", "Constant") == "Constant":
        return ConstantDEM(S.get("dem_const", 0.0))
    return SRTMDEM(download=True, fallback=S.get("dem_const", 0.0))


def c_sound():
    return doppler.speed_of_sound(S.get("temp_c", 15.0))


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("✈️ PlaneSounds")
    st.caption("Find where a recording was made from the Doppler curves of aircraft flying over.")
    st.number_input("Air temperature (°C)", -40.0, 50.0, 15.0, 1.0, key="temp_c",
                    help="Sets the speed of sound. Use the local temperature at recording time if known.")
    st.caption(f"speed of sound ≈ {c_sound():.1f} m/s")
    st.divider()
    st.subheader("Session")
    if S.events:
        st.download_button("💾 Save events (JSON)", json.dumps({"events": S.events}, indent=1),
                           "planesounds_events.json", "application/json")
    up = st.file_uploader("Load events JSON", type=["json"], key="ev_upload")
    if up is not None and st.button("Load events"):
        S.events = json.load(up).get("events", [])
        st.success(f"loaded {len(S.events)} events")
    st.divider()
    if S.audio is not None:
        a = S.audio
        st.markdown(f"**Audio:** {a.title or a.source}  \n{a.duration:.0f} s @ {a.sr} Hz")
        if a.upload_date:
            st.markdown(f"YouTube upload date: **{a.upload_date}**")
    st.markdown(f"**Events:** {len(S.events)}")
    if S.flights is not None:
        st.markdown(f"**Flight data:** {S.flights.flight_id.nunique()} flights, {len(S.flights):,} points")

tabs = st.tabs(["1 · Audio", "2 · Spectrogram", "3 · Measure events", "4 · Area, time & flight data",
                "5 · Search", "6 · Inspect results", "ℹ️ How it works"])

# ---------------------------------------------------------------------------
# 1. audio
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Load audio")
    c1, c2 = st.columns([3, 1])
    url = c1.text_input("YouTube (or any yt-dlp supported) URL, or a local file path",
                        placeholder="https://www.youtube.com/watch?v=...")
    sr = c2.selectbox("Sample rate", [4000, 8000, 16000], index=1,
                      help="8 kHz keeps everything up to 4 kHz - plenty for propeller/engine tones.")
    c3, c4 = st.columns(2)
    t_start = c3.number_input("Start at (s)", 0.0, value=0.0, step=10.0)
    t_end = c4.number_input("End at (s, 0 = whole file)", 0.0, value=0.0, step=10.0)
    up = st.file_uploader("...or upload an audio/video file", type=None)
    b1, b2 = st.columns(2)
    if b1.button("⬇️ Load audio", type="primary", disabled=not (url or up)):
        try:
            bar = st.progress(0.0, "downloading")
            if up is not None:
                tmp = Path(tempfile.mkdtemp()) / up.name
                tmp.write_bytes(up.getvalue())
                src = str(tmp)
            else:
                src = url.strip()
            a = au.load_source(src, sr, t_start or None, t_end or None,
                               progress=lambda x: bar.progress(x, "downloading"))
            bar.progress(1.0, "decoded")
            S.audio, S.spec_cache, S.trace, S.fit, S.sel, S.demo = a, {}, None, None, None, None
            st.success(f"Loaded {a.duration:.1f} s of audio")
        except Exception as e:
            st.error(f"Could not load audio: {e}")
    if b2.button("🧪 Load synthetic demo scenario",
                 help="Two simulated propeller flyovers 920 s apart + background traffic, with matching "
                      "fake ADS-B data. Useful to learn the workflow and verify the pipeline."):
        sc = synth.demo_scenario()
        S.audio = au.Audio(sc["audio"], sc["sr"], source="synthetic demo", title="Synthetic demo")
        S.flights = sc["flights"]
        S.demo = {k: sc[k] for k in ("obs", "rec_start", "duration")}
        S.spec_cache, S.trace, S.fit, S.sel, S.events = {}, None, None, None, []
        lat, lon, _ = sc["obs"]
        S.area = {"lat_min": lat - 0.3, "lat_max": lat + 0.3, "lon_min": lon - 0.4, "lon_max": lon + 0.4}
        st.success("Demo loaded. Truth (hidden from the search): "
                   f"{lat:.5f}, {lon:.5f}; recording started {fmt_utc(sc['rec_start'])}. "
                   "Events are at ~148 s (≈100 Hz) and ~1090 s (≈88 Hz).")
    if S.audio is not None:
        a = S.audio
        st.markdown(f"**{a.title or a.source}** - {a.duration:.1f} s")
        if a.meta:
            with st.expander("Video metadata (clues for date/time and area!)"):
                st.json({k: v for k, v in a.meta.items() if k != "_file"})
            st.info("The *upload* date is only an upper bound on the *recording* date. "
                    "Look for dates in the title/description, timestamps on screen, weather, sun angle, etc.")
        if a.duration < 1200:
            st.audio(a.samples, sample_rate=a.sr)

# ---------------------------------------------------------------------------
# 2. spectrogram
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Spectrogram explorer")
    if S.audio is None:
        st.info("Load audio first.")
    else:
        a = S.audio
        with st.expander("Overview (whole recording)", expanded=True):
            oc = st.columns(4)
            ov_fmax = oc[0].number_input("Max freq (Hz)", 50.0, a.sr / 2, min(1000.0, a.sr / 2), 50.0, key="ov_fmax")
            ov_win = oc[1].select_slider("Window (s)", [0.25, 0.5, 1.0, 2.0], 0.5, key="ov_win")
            ov_hop = oc[2].select_slider("Hop (s)", [0.25, 0.5, 1.0, 2.0], 1.0, key="ov_hop")
            ov_wh = oc[3].selectbox("Whitening", ["freq", "both", "none"], 0, key="ov_wh",
                                    help="'freq' removes constant hum per frequency; 'both' also removes broadband bursts")
            so = get_spec(ov_win, ov_hop, ov_fmax, ov_wh)
            st.plotly_chart(spec_figure(so, "Whole recording - look for faint descending curves",
                                        height=380), width="stretch", theme=None)
            if st.button("🔎 Auto-detect Doppler candidates"):
                sd = get_spec(1.0, 0.25, min(800.0, a.sr / 2), "freq")
                with st.spinner("scanning..."):
                    S.detected = ridge.detect_candidates(sd, f_lo=30, f_hi=min(700, a.sr / 2 - 10))
            if S.get("detected"):
                df = pd.DataFrame(S.detected)[["t_cpa", "f0", "v", "d", "strength", "rms", "t0", "t1", "f_lo", "f_hi"]]
                st.dataframe(df.round(2), width="stretch")
                pick = st.selectbox("Use detection as zoom selection", range(len(df)),
                                    format_func=lambda i: f"#{i}: CPA {df.t_cpa[i]:.1f}s, {df.f0[i]:.1f} Hz")
                if st.button("Zoom to it"):
                    d = S.detected[pick]
                    S.sel = {"t0": max(0, d["t_cpa"] - 90), "t1": min(a.duration, d["t_cpa"] + 90),
                             "f0": d["f_lo"], "f1": d["f_hi"]}
                    st.rerun()

        st.subheader("Zoom")
        sel = S.sel or {"t0": 0.0, "t1": min(a.duration, 240.0), "f0": 40.0, "f1": min(400.0, a.sr / 2)}
        zc = st.columns(6)
        z_t0 = zc[0].number_input("t from", 0.0, a.duration, float(sel["t0"]), 1.0)
        z_t1 = zc[1].number_input("t to", 0.0, a.duration, float(sel["t1"]), 1.0)
        z_f0 = zc[2].number_input("f from", 0.0, a.sr / 2, float(sel["f0"]), 1.0)
        z_f1 = zc[3].number_input("f to", 0.0, a.sr / 2, float(sel["f1"]), 1.0)
        z_win = zc[4].select_slider("Window (s)", [0.5, 1.0, 2.0, 3.0, 4.0], 1.0,
                                    help="Longer = sharper frequency, blurrier time. 1-2 s works well for props.")
        z_c = zc[5].slider("Contrast", 0.2, 3.0, 1.0, 0.1)
        S.sel = {"t0": z_t0, "t1": max(z_t1, z_t0 + 5), "f0": z_f0, "f1": max(z_f1, z_f0 + 5)}
        sz = get_spec(z_win, z_win / 4, S.sel["f1"] + 5, "freq", S.sel["t0"], S.sel["t1"], max(0, S.sel["f0"] - 5))
        sz = sz.crop(S.sel["t0"], S.sel["t1"], S.sel["f0"], S.sel["f1"])
        zmax = float(np.percentile(sz.z, 99.5) / z_c) if sz.z.size else 1
        ov = []
        if S.trace is not None:
            ov.append(line(S.trace[0], S.trace[1], "trace", "#00e5ff", mode="markers"))
        fig = spec_figure(sz, "Drag a box around a Doppler curve to select it", zmax=zmax, overlays=ov,
                          selectable=True, height=520)
        ev = st.plotly_chart(fig, width="stretch", theme=None, on_select="rerun", selection_mode=("box",),
                             key="zoom_chart")
        try:
            box = ev["selection"]["box"][0] if ev and ev["selection"].get("box") else None
        except Exception:
            box = None
        if box:
            x0, x1 = sorted(box["x"])
            y0, y1 = sorted(box["y"])
            st.success(f"Box: {x0:.1f}-{x1:.1f} s, {y0:.1f}-{y1:.1f} Hz")
            if st.button("Use this box as the selection (zoom in)"):
                S.sel = {"t0": x0, "t1": x1, "f0": y0, "f1": y1}
                st.rerun()

        st.subheader("Listen")
        lc = st.columns(4)
        gain = lc[0].slider("Gain (dB)", 0, 60, 30)
        pitch = lc[1].select_slider("Playback speed", [1.0, 1.5, 2.0, 3.0, 4.0], 1.0,
                                    help="Speeding up raises low tones into a more audible range")
        lc[2].markdown(" ")
        if lc[2].button("▶️ Render band-passed clip"):
            x = a.segment(S.sel["t0"], S.sel["t1"])
            y, sr2 = au.make_listenable(x, a.sr, max(S.sel["f0"], 15), S.sel["f1"], gain, pitch)
            st.audio(y, sample_rate=sr2)

# ---------------------------------------------------------------------------
# 3. events
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Measure Doppler events")
    if S.audio is None:
        st.info("Load audio first.")
    else:
        a = S.audio
        sel = S.sel or {"t0": 0.0, "t1": min(a.duration, 240.0), "f0": 40.0, "f1": 400.0}
        st.caption(f"Current selection: {sel['t0']:.1f}-{sel['t1']:.1f} s, {sel['f0']:.1f}-{sel['f1']:.1f} Hz "
                   "(set it in the Spectrogram tab). Select one harmonic line of one pass-by, ideally the clearest.")
        cc = st.columns(5)
        win = cc[0].select_slider("Window (s)", [0.5, 1.0, 2.0, 3.0], 1.0, key="ev_win")
        slope = cc[1].number_input("Max slope (Hz/s)", 0.2, 30.0, 3.0, 0.5,
                                   help="Max frequency change per second along the trace")
        mono = cc[2].checkbox("Monotonic (only falling)", True)
        minst = cc[3].number_input("Min strength (σ)", 0.0, 10.0, 1.5, 0.25)
        cc[4].markdown(" ")
        if cc[4].button("✏️ Trace + fit", type="primary"):
            s_ev = get_spec(win, 0.25, sel["f1"] + 10, "freq", sel["t0"], sel["t1"], max(0, sel["f0"] - 10))
            try:
                t, f, stn = ridge.trace_ridge(s_ev, sel["t0"], sel["t1"], sel["f0"], sel["f1"],
                                              max_slope_hz_s=slope, monotonic=mono)
                t, f, stn = ridge.clean_trace(t, f, stn, minst)
                S.trace = (t, f, stn)
                S.fit = doppler.fit_linear(t, f, c_sound(), weights=stn)
            except Exception as e:
                st.error(f"Trace/fit failed: {e}")

        fit = S.fit
        if fit is not None:
            st.markdown(f"**Straight-line fit:** f₀ = {fit.f0:.2f} Hz · speed = {fit.v:.1f} m/s "
                        f"({fit.v * 1.944:.0f} kt) · CPA slant distance = {fit.d:.0f} m · "
                        f"CPA at {fit.t_cpa:.2f} s · RMS = {fit.rms_hz:.2f} Hz over {fit.n} points")
            st.caption("Tweak the curve by hand like in the reference video: the speed sets how *big* the "
                       "frequency drop is, the distance sets how *quickly* it happens.")
            mc = st.columns(4)
            m_f0 = mc[0].slider("f₀ (Hz)", float(fit.f0 * 0.8), float(fit.f0 * 1.2), float(fit.f0), 0.05)
            m_v = mc[1].slider("speed (m/s)", 10.0, 320.0, float(np.clip(fit.v, 10, 320)), 0.5)
            m_d = mc[2].slider("CPA distance (m)", 30.0, 15000.0, float(np.clip(fit.d, 30, 15000)), 10.0)
            m_t = mc[3].slider("CPA time (s)", float(fit.t_cpa - 30), float(fit.t_cpa + 30), float(fit.t_cpa), 0.05)
            s_ev = get_spec(win, 0.25, sel["f1"] + 10, "freq", sel["t0"], sel["t1"], max(0, sel["f0"] - 10))
            s_ev = s_ev.crop(sel["t0"], sel["t1"], sel["f0"], sel["f1"])
            tt = np.linspace(sel["t0"], sel["t1"], 600)
            ovs = [line(S.trace[0], S.trace[1], "trace", "#00e5ff", mode="markers"),
                   line(tt, doppler.linear_model(tt, fit.f0, fit.v, fit.d, fit.t_cpa, c_sound()), "fit", "#76ff03"),
                   line(tt, doppler.linear_model(tt, m_f0, m_v, m_d, m_t, c_sound()), "manual", "#ffea00", dash="dot")]
            show_c = st.checkbox("Hide overlays (see the raw curve)", False)
            st.plotly_chart(spec_figure(s_ev, "Traced ridge + model", overlays=[] if show_c else ovs, height=500),
                            width="stretch", theme=None)
            resid = S.trace[1] - doppler.linear_model(S.trace[0], fit.f0, fit.v, fit.d, fit.t_cpa, c_sound())
            with st.expander("Fit residuals (systematic wiggles = turns / speed or altitude changes - "
                             "that's exactly what the trajectory scoring exploits)"):
                st.plotly_chart(go.Figure(line(S.trace[0], resid, "residual", "#ff4081", mode="markers"))
                                .update_layout(height=250, yaxis_title="Hz", xaxis_title="s"),
                                width="stretch", theme=None)

            st.subheader("Save as event")
            base = fit.ranges()
            use_manual = st.checkbox("Centre ranges on the manual slider values", False)
            v_c, d_c, t_c, f_c = (m_v, m_d, m_t, m_f0) if use_manual else (fit.v, fit.d, fit.t_cpa, fit.f0)
            rc = st.columns(4)
            vw = rc[0].slider("speed ± (m/s)", 2.0, 80.0, float(max((base["v"][1] - base["v"][0]) / 2, 2.0)), 1.0)
            dw = rc[1].slider("distance ± (m)", 20.0, 5000.0, float(max((base["d"][1] - base["d"][0]) / 2, 20)), 10.0)
            harm = rc[2].text_input("Harmonics to score (× f₀)", "1",
                                    help="e.g. '1, 2, 3' if the traced line is the fundamental and harmonics are visible")
            mode = rc[3].selectbox("Score mode", ["spectrogram", "trace"],
                                   help="'trace' scores by distance to the traced line - best for extremely faint curves")
            wc = st.columns(3)
            name = wc[0].text_input("Name", f"event {len(S.events) + 1}")
            twin = wc[1].slider("Scoring window ± (s) around CPA", 10.0, 180.0, 60.0, 5.0)
            jets = wc[2].checkbox("Could be a jet", False, help="Leave off for propeller sounds")
            if st.button("➕ Save event", type="primary"):
                try:
                    hs = [float(h) for h in harm.replace(";", ",").split(",") if h.strip()]
                except ValueError:
                    hs = [1.0]
                e = search.Event(name=name, t_cpa=float(t_c), f0=float(f_c),
                                 v_range=(max(5.0, v_c - vw), v_c + vw), d_range=(max(30.0, d_c - dw), d_c + dw),
                                 t_window=(max(0.0, t_c - twin), min(a.duration, t_c + twin)),
                                 f_band=(sel["f0"], sel["f1"]), harmonics=hs or [1.0],
                                 trace_t=list(map(float, S.trace[0])), trace_f=list(map(float, S.trace[1])),
                                 score_mode=mode, allow_jets=jets)
                S.events.append(e.to_dict())
                S.events.sort(key=lambda d: d["t_cpa"])
                st.success(f"saved {name}")

        if S.events:
            st.subheader("Saved events")
            df = pd.DataFrame([{k: v for k, v in e.items() if k not in ("trace_t", "trace_f")} for e in S.events])
            st.dataframe(df, width="stretch")
            if len(S.events) > 1:
                gaps = np.diff([e["t_cpa"] for e in S.events])
                st.caption("Offsets between events: " + ", ".join(f"{g:.1f} s" for g in gaps))
            dc = st.columns([2, 1])
            rm = dc[0].selectbox("Remove event", range(len(S.events)), format_func=lambda i: S.events[i]["name"])
            if dc[1].button("🗑️ Remove"):
                S.events.pop(rm)
                st.rerun()

# ---------------------------------------------------------------------------
# 4. area, time & data
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Search area & time")
    area = S.area or {"lat_min": 40.5, "lat_max": 41.2, "lon_min": -74.3, "lon_max": -73.2}
    am = st.radio("Define area by", ["Centre + radius", "Bounding box", "Continental US"], horizontal=True)
    if am == "Centre + radius":
        c = st.columns(3)
        clat = c[0].number_input("Centre lat", -90.0, 90.0, (area["lat_min"] + area["lat_max"]) / 2, 0.01, format="%.4f")
        clon = c[1].number_input("Centre lon", -180.0, 180.0, (area["lon_min"] + area["lon_max"]) / 2, 0.01, format="%.4f")
        rad = c[2].number_input("Radius (km)", 1.0, 2000.0, 30.0, 5.0)
        dlat = rad / 111.32
        dlon = rad / (111.32 * np.cos(np.radians(clat)))
        S.area = {"lat_min": clat - dlat, "lat_max": clat + dlat, "lon_min": clon - dlon, "lon_max": clon + dlon}
    elif am == "Bounding box":
        c = st.columns(4)
        S.area = {"lat_min": c[0].number_input("lat min", -90.0, 90.0, area["lat_min"], 0.01, format="%.4f"),
                  "lat_max": c[1].number_input("lat max", -90.0, 90.0, area["lat_max"], 0.01, format="%.4f"),
                  "lon_min": c[2].number_input("lon min", -180.0, 180.0, area["lon_min"], 0.01, format="%.4f"),
                  "lon_max": c[3].number_input("lon max", -180.0, 180.0, area["lon_max"], 0.01, format="%.4f")}
    else:
        S.area = {"lat_min": 24.4, "lat_max": 49.4, "lon_min": -125.0, "lon_max": -66.9}
        st.warning("Whole-country searches need a full day of ADS-B (tens of GB) and lots of RAM/CPU. "
                   "Narrow the area whenever you have any other clue.")
    ar = S.area
    figm = go.Figure(go.Scattermap(lat=[ar["lat_min"], ar["lat_min"], ar["lat_max"], ar["lat_max"], ar["lat_min"]],
                                   lon=[ar["lon_min"], ar["lon_max"], ar["lon_max"], ar["lon_min"], ar["lon_min"]],
                                   mode="lines", fill="toself", name="area"))
    figm.update_layout(map=dict(style="open-street-map", center=dict(lat=(ar["lat_min"] + ar["lat_max"]) / 2,
                                                                    lon=(ar["lon_min"] + ar["lon_max"]) / 2),
                                zoom=max(1, 8 - np.log2(max(ar["lat_max"] - ar["lat_min"], 0.01) * 8))),
                       height=300, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(figm, width="stretch", theme=None)

    st.subheader("When was it recorded? (time of audio t = 0)")
    default_day = dt.date.today() - dt.timedelta(days=2)
    if S.demo:
        default_day = dt.datetime.fromtimestamp(S.demo["rec_start"], UTC).date()
    elif S.audio is not None and S.audio.upload_date:
        try:
            default_day = dt.datetime.strptime(S.audio.upload_date, "%Y%m%d").date()
        except ValueError:
            pass
    tm = st.radio("Time knowledge", ["Day known (UTC)", "Exact start time known", "Custom range"], horizontal=True)
    if tm == "Day known (UTC)":
        d = st.date_input("Recording day (UTC)", default_day)
        c = st.columns(2)
        h0 = c[0].time_input("earliest start (UTC)", dt.time(0, 0))
        h1 = c[1].time_input("latest start (UTC)", dt.time(23, 59))
        S.t0_range = (unix(d, h0), unix(d, h1))
    elif tm == "Exact start time known":
        c = st.columns(3)
        d = c[0].date_input("Date (UTC)", default_day)
        h = c[1].time_input("Start time (UTC)", dt.time(12, 0), step=60)
        tol = c[2].number_input("± tolerance (s)", 0, 7200, 120)
        S.t0_range = (unix(d, h) - tol, unix(d, h) + tol)
    else:
        c = st.columns(2)
        a0 = c[0].text_input("earliest start (ISO, UTC)", f"{default_day}T00:00:00")
        a1 = c[1].text_input("latest start (ISO, UTC)", f"{default_day}T23:59:59")
        try:
            S.t0_range = (pd.Timestamp(a0, tz="UTC").timestamp(), pd.Timestamp(a1, tz="UTC").timestamp())
        except Exception:
            st.error("bad timestamp")
    st.caption(f"Audio t=0 between {fmt_utc(S.t0_range[0])} and {fmt_utc(S.t0_range[1])}")

    st.header("Flight (ADS-B) data")
    src = st.radio("Source", ["Local files / folder", "Upload", "adsb.lol daily archive (auto-download)",
                              "OpenSky (Trino, needs account)"], horizontal=True)
    if src == "Local files / folder":
        p = st.text_input("Path(s), comma-separated: CSV / parquet / readsb trace_full JSON / folder / .tar",
                          help="Big data stays on disk - only points inside the area & time window are kept.")
        if st.button("Load", disabled=not p):
            try:
                paths = [Path(x.strip()).expanduser() for x in p.split(",") if x.strip()]
                bbox = (ar["lat_min"], ar["lat_max"], ar["lon_min"], ar["lon_max"])
                tw = (S.t0_range[0] - 3600, S.t0_range[1] + 7200)
                parts = []
                trace_like = []
                for q in paths:
                    if q.is_dir() or ".tar" in q.name or q.name.startswith("trace_"):
                        trace_like.append(q)
                    else:
                        parts.append(adsb.load_table(q))
                if trace_like:
                    with st.spinner("parsing traces..."):
                        parts.append(adsb.load_readsb_traces(trace_like, bbox, tw))
                S.flights = pd.concat(parts, ignore_index=True) if parts else None
                st.success(f"{S.flights.flight_id.nunique()} flights loaded")
            except Exception as e:
                st.error(f"load failed: {e}")
    elif src == "Upload":
        ups = st.file_uploader("CSV / parquet / trace JSON files", accept_multiple_files=True)
        if ups and st.button("Load uploaded"):
            parts = []
            for u in ups:
                tmp = Path(tempfile.mkdtemp()) / u.name
                tmp.write_bytes(u.getvalue())
                parts.append(adsb.load_table(tmp))
            S.flights = pd.concat(parts, ignore_index=True)
            st.success(f"{S.flights.flight_id.nunique()} flights loaded")
    elif src.startswith("adsb.lol"):
        st.markdown("Streams one UTC day of the community [adsb.lol](https://adsb.lol) "
                    "`globe_history` archive from GitHub (several GB) and keeps only your area. "
                    "Set a `GITHUB_TOKEN` env var to avoid API rate limits.")
        d = st.date_input("UTC day to download", dt.datetime.fromtimestamp(S.t0_range[0], UTC).date(), key="dl_day")
        if st.button("⬇️ Download & filter", type="primary"):
            bar = st.progress(0.0, "starting")
            bbox = (ar["lat_min"], ar["lat_max"], ar["lon_min"], ar["lon_max"])
            outp = adsb.DATA_DIR / f"adsblol_{d}_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}.parquet"
            try:
                if outp.exists():
                    S.flights = adsb.load_table(outp)
                else:
                    S.flights = adsb.download_adsblol_day(d, bbox, out_path=outp,
                                                          progress=lambda x, m: bar.progress(x, m))
                bar.progress(1.0, f"saved to {outp}")
                st.success(f"{S.flights.flight_id.nunique()} flights")
            except Exception as e:
                st.error(f"download failed: {e}")
    else:
        st.markdown("Requires `pip install pyopensky` and OpenSky credentials configured "
                    "(see the pyopensky docs). Historical access must be granted by OpenSky.")
        if st.button("Query OpenSky"):
            bbox = (ar["lat_min"], ar["lat_max"], ar["lon_min"], ar["lon_max"])
            try:
                with st.spinner("querying..."):
                    S.flights = adsb.opensky_history(S.t0_range[0] - 3600, S.t0_range[1] + 7200, bbox)
                st.success(f"{S.flights.flight_id.nunique()} flights")
            except Exception as e:
                st.error(str(e))
    tdb = st.text_input("Optional: aircraft database CSV to fill missing type codes "
                        "(e.g. OpenSky aircraftDatabase.csv)")
    if tdb and S.flights is not None and st.button("Merge type codes"):
        S.flights = adsb.merge_typecodes(S.flights, tdb)

    if S.flights is not None and len(S.flights):
        fl = S.flights
        summ = adsb.summarize(fl)
        st.markdown(f"**{len(summ)} flights** · " + ", ".join(f"{k}: {v}" for k, v in summ.propulsion.value_counts().items()))
        if st.button("💾 Save flight data as parquet"):
            outp = adsb.DATA_DIR / f"flights_{dt.datetime.now():%Y%m%d_%H%M%S}.parquet"
            adsb.save(fl, outp)
            st.success(f"saved {outp}")
        st.dataframe(summ, width="stretch", height=250)
        show_jets = st.checkbox("Show jets on map", False)
        sub = fl if show_jets else fl[fl.propulsion != "jet"]
        sub = sub[sub.flight_id.isin(sub.flight_id.unique()[:400])]
        fm = go.Figure()
        lat_l, lon_l = [], []
        for _, g in sub.groupby("flight_id"):
            g = g.iloc[:: max(1, len(g) // 300)]
            lat_l += list(g.lat) + [None]
            lon_l += list(g.lon) + [None]
        fm.add_trace(go.Scattermap(lat=lat_l, lon=lon_l, mode="lines", line=dict(width=1), name="tracks"))
        fm.update_layout(map=dict(style="open-street-map", center=dict(lat=fl.lat.median(), lon=fl.lon.median()), zoom=7),
                         height=450, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fm, width="stretch", theme=None)

# ---------------------------------------------------------------------------
# 5. search
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Search")
    ready = S.audio is not None and S.events and S.flights is not None and S.area
    if not ready:
        st.info("Needed: audio (tab 1), at least one saved event (tab 3), flight data + area/time (tab 4).")
    else:
        a = S.audio
        c = st.columns(4)
        grid_m = c[0].select_slider("Coarse grid (m)", [100, 150, 200, 300, 400, 500], 200)
        refine_m = c[1].select_slider("Scoring grid (m)", [25, 50, 100, 200], 100)
        tol = c[2].number_input("Time-offset tolerance (s)", 1.0, 60.0, 6.0, 1.0,
                                help="How precisely the CPA times can be measured on the spectrogram")
        maxp = c[3].number_input("Max scored points", 1000, 1_000_000, 40000, 5000)
        c = st.columns(4)
        excl = c[0].checkbox("Exclude known jets", True)
        sil = c[1].checkbox("Silence filter", True,
                            help="Reject places that would have heard other aircraft while the recording is quiet")
        srad = c[2].number_input("Silence radius props (m)", 200.0, 20000.0, 2500.0, 100.0)
        srj = c[3].number_input("Silence radius jets (m)", 200.0, 30000.0, 4500.0, 100.0)
        spans_default = search.default_clean_spans([search.Event(**e) for e in S.events], a.duration, 15.0)
        spans_txt = st.text_input("Quiet spans (audio s) - no aircraft audible",
                                  "; ".join(f"{s0:.0f}-{s1:.0f}" for s0, s1 in spans_default),
                                  help="Edit if other (unused) aircraft are audible somewhere in the recording")
        c = st.columns(4)
        S.dem_mode = c[0].radio("Ground elevation", ["Constant", "SRTM (auto-download)"], horizontal=False)
        S.dem_const = c[1].number_input("Constant / fallback elevation (m)", -100.0, 5000.0, 0.0 if not S.demo else S.demo["obs"][2], 10.0)
        alt_off = c[2].number_input("ADS-B altitude offset (m)", -300.0, 300.0, 0.0, 5.0,
                                    help="Barometric altitudes can be off by QNH; add a correction if known")
        dts = c[3].number_input("Event-1 time sweep ± (s)", 0.5, 20.0, 4.0, 0.5)
        if st.button("🚀 Run search", type="primary"):
            spans = []
            for part in spans_txt.split(";"):
                if "-" in part:
                    x0, x1 = part.split("-")[:2]
                    spans.append((float(x0), float(x1)))
            evs = [search.Event(**e) for e in S.events]
            fmax = max(e.f_band[1] * max(e.harmonics) for e in evs) * 1.1 + 10
            spec_full = get_spec(1.0, 0.25, min(fmax, a.sr / 2), "freq")
            cfg = search.SearchConfig(
                bbox=(S.area["lat_min"], S.area["lat_max"], S.area["lon_min"], S.area["lon_max"]),
                t0_range=S.t0_range, duration=a.duration, grid_m=grid_m, refine_m=min(refine_m, grid_m),
                c=c_sound(), offset_tol_s=tol, exclude_jets=excl, silence=sil, silence_radius_m=srad,
                silence_radius_jet_m=srj, clean_spans=spans, max_score_points=int(maxp), dt_sweep_s=dts,
                alt_offset_m=alt_off)
            bar = st.progress(0.0, "starting")
            res = search.run_search(S.flights, evs, spec_full, cfg, make_dem(),
                                    progress=lambda x, m: bar.progress(x, m))
            S.result = {"res": res, "cfg": cfg, "events": evs, "spec": spec_full}
            S.heat = None
        if S.result:
            res = S.result["res"]
            evs = S.result["events"]
            with st.expander("Log", expanded=res.results.empty):
                st.code("\n".join(res.log))
            fun = pd.DataFrame(res.stages)
            st.plotly_chart(go.Figure(go.Funnel(y=fun.stage, x=fun["count"], textinfo="value"))
                            .update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10)),
                            width="stretch", theme=None)
            if not res.results.empty:
                best = res.best_per_combo
                if S.demo:
                    o = S.demo["obs"]
                    best = best.assign(err_m=haversine_m(o[0], o[1], best.lat, best.lon))
                c1, c2 = st.columns(2)
                if len(evs) >= 2:
                    fig = go.Figure(go.Scatter(x=res.results.score_0, y=res.results.score_1, mode="markers",
                                               marker=dict(size=4, color="#9e9e9e"), name="all points"))
                    fig.add_trace(go.Scatter(x=best.score_0, y=best.score_1, mode="markers",
                                             marker=dict(size=9, color=best.score, colorscale="Viridis"),
                                             text=best.index.astype(str), name="best per flight pair"))
                    fig.update_layout(xaxis_title=f"{evs[0].name} score", yaxis_title=f"{evs[1].name} score",
                                      height=420, title="Top-right is better")
                else:
                    fig = go.Figure(go.Histogram(x=best.score, nbinsx=50))
                    fig.update_layout(xaxis_title="score", height=420, title="Best score per flight (right is better)")
                c1.plotly_chart(fig, width="stretch", theme=None)
                top = best.head(50)
                fm = go.Figure(go.Scattermap(lat=top.lat, lon=top.lon, mode="markers",
                                             marker=dict(size=10, color=top.score, colorscale="Viridis", showscale=True),
                                             text=[f"#{i} score {s:.2f}" for i, s in zip(top.index, top.score)]))
                fm.update_layout(map=dict(style="open-street-map", center=dict(lat=top.lat.iloc[0], lon=top.lon.iloc[0]),
                                          zoom=9), height=420, margin=dict(l=0, r=0, t=0, b=0))
                c2.plotly_chart(fm, width="stretch", theme=None)
                cols = ["lat", "lon", "score"] + [f"score_{k}" for k in range(len(evs))] + ["start_utc"] + \
                       sum([[f"callsign_{k}", f"type_{k}", f"d_{k}", f"speed_{k}"] for k in range(len(evs))], []) + \
                       (["err_m"] if "err_m" in best else [])
                st.dataframe(best[cols].head(200), width="stretch")
                st.download_button("⬇️ Results CSV", res.results.to_csv(index=False), "planesounds_results.csv")

# ---------------------------------------------------------------------------
# 6. inspect
# ---------------------------------------------------------------------------
with tabs[5]:
    st.header("Inspect a candidate")
    if not S.result or S.result["res"].results.empty:
        st.info("Run a search first.")
    else:
        res, evs, cfg, spec_full = S.result["res"], S.result["events"], S.result["cfg"], S.result["spec"]
        best = res.best_per_combo
        i = st.selectbox("Candidate", range(min(len(best), 100)),
                         format_func=lambda i: f"#{i}: score {best.score[i]:.2f} @ {best.lat[i]:.5f}, {best.lon[i]:.5f}")
        row = best.iloc[i]
        lat, lon = float(row.lat), float(row.lon)
        st.markdown(f"**{lat:.6f}, {lon:.6f}** · recording started ≈ **{fmt_utc(row.T0)}** · "
                    f"[Google Maps](https://www.google.com/maps/search/?api=1&query={lat},{lon}) · "
                    f"[Street View](https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}) · "
                    f"[Satellite](https://www.google.com/maps/@{lat},{lon},400m/data=!3m1!1e3)")
        trs = [res.tracks[row[f"fid_{k}"]] for k in range(len(evs))]
        st.dataframe(pd.DataFrame([{"event": e.name, "flight": t.fid, "callsign": t.callsign, "type": t.typecode,
                                    "class": t.propulsion, "score": row[f"score_{k}"], "slant CPA (m)": row[f"d_{k}"],
                                    "speed (m/s)": row[f"speed_{k}"], "f0 best": row[f"f0_{k}"]}
                                   for k, (e, t) in enumerate(zip(evs, trs))]), width="stretch")
        scorer = search.Scorer(spec_full, evs, cfg.c, cfg.dt_sweep_s, cfg.f0_sweep_frac)
        obs = lla_to_ecef(lat, lon, row.elev)
        curves = scorer.model_curves(trs, obs, row.T0, [row[f"f0_{k}"] for k in range(len(evs))])
        hide = st.checkbox("Hide modelled curves", False)
        cols = st.columns(len(evs))
        for k, (e, (ta, fa)) in enumerate(zip(evs, curves)):
            w0, w1 = e.t_window[0] - 20, e.t_window[1] + 20
            fl, fh = e.f_band
            sub = spec_full.crop(w0, w1, max(0, fl - 10), fh + 10)
            ovs = [] if hide else [line(ta, fa * h, f"model ×{h:g}", "#00e5ff", 2, "dot") for h in e.harmonics]
            fig = spec_figure(sub, f"{e.name}: model from real trajectory", overlays=ovs, height=420)
            fig.update_xaxes(range=[w0, w1])
            fig.update_yaxes(range=[max(0, fl - 10), fh + 10])
            cols[k].plotly_chart(fig, width="stretch", theme=None)

        # map with tracks
        fm = go.Figure()
        for k, t in enumerate(trs):
            m = np.abs(t.t - (row.T0 + evs[k].t_cpa)) < 400
            fm.add_trace(go.Scattermap(lat=t.lat[m], lon=t.lon[m], mode="lines", name=f"{evs[k].name}: {t.callsign or t.fid}"))
        fm.add_trace(go.Scattermap(lat=[lat], lon=[lon], mode="markers", marker=dict(size=14, color="red"), name="candidate"))
        if S.heat is not None and S.heat.get("i") == i:
            h = S.heat["df"]
            fm.add_trace(go.Scattermap(lat=h.lat, lon=h.lon, mode="markers", name="focus score",
                                       marker=dict(size=7, color=h.score, colorscale="Turbo", showscale=True, opacity=0.7)))
        if S.demo:
            fm.add_trace(go.Scattermap(lat=[S.demo["obs"][0]], lon=[S.demo["obs"][1]], mode="markers",
                                       marker=dict(size=12, color="lime"), name="demo truth"))
        fm.update_layout(map=dict(style="open-street-map", center=dict(lat=lat, lon=lon), zoom=13),
                         height=520, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fm, width="stretch", theme=None)
        hc = st.columns(3)
        hr = hc[0].number_input("Focus radius (m)", 100.0, 5000.0, 600.0, 50.0)
        hs = hc[1].number_input("Focus step (m)", 5.0, 200.0, 25.0, 5.0)
        if hc[2].button("🔥 Fine heat-map for this flight combination"):
            bar = st.progress(0.0)
            h = search.focus_heatmap(trs, evs, spec_full, (lat, lon), hr, hs, make_dem(), cfg.c,
                                     cfg.dt_sweep_s, cfg.f0_sweep_frac, progress=lambda x: bar.progress(x))
            S.heat = {"i": i, "df": h}
            st.rerun()
        if S.heat is not None and S.heat.get("i") == i:
            hb = S.heat["df"].sort_values("score", ascending=False).iloc[0]
            st.success(f"Fine-grid best: {hb.lat:.6f}, {hb.lon:.6f} (score {hb.score:.2f}) - "
                       f"[Street View](https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={hb.lat},{hb.lon})")

# ---------------------------------------------------------------------------
# 7. docs
# ---------------------------------------------------------------------------
with tabs[6]:
    st.markdown(Path(__file__).with_name("README.md").read_text() if Path(__file__).with_name("README.md").exists() else "")
