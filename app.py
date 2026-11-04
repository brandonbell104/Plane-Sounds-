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

from planesounds import adsb, audio as au, cuts as cutmod, doppler, fr24src, ridge, search, spectro, synth
from planesounds.dem import ConstantDEM, SRTMDEM
from planesounds.geo import haversine_m, lla_to_ecef

st.set_page_config(page_title="PlaneSounds", page_icon="✈️", layout="wide")
pio.templates.default = "plotly_dark"
S = st.session_state
for k, v in {"audio": None, "events": [], "sel": None, "trace": None, "fit": None, "flights": None,
             "result": None, "spec_cache": {}, "area": None, "demo": None, "heat": None,
             "cuts": [], "detected": None}.items():
    S.setdefault(k, v)

UTC = dt.timezone.utc
# max (time columns, frequency rows) sent to the browser per spectrogram
DETAIL = {"Standard": (1400, 500), "High": (2400, 800), "Very high": (4000, 1200)}
# explicit rgb() stops: the Streamlit frontend re-maps some named/hex scale colours
SPEC_CMAP = [[i / 8, f"rgb({r},{g},{b})"] for i, (r, g, b) in enumerate([
    (2, 2, 8), (20, 14, 54), (59, 15, 112), (100, 26, 128), (140, 41, 129),
    (222, 73, 104), (254, 159, 109), (254, 207, 146), (252, 253, 191)])]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_spec(win_s, hop_s, f_max, whiten, t0=None, t1=None, f_min=0.0, oversample=2):
    a = S.audio
    segs = tuple(seg_bounds()) if S.cuts else None
    key = (id(a), round(win_s, 3), round(hop_s, 3), round(f_max), whiten,
           None if t0 is None else round(t0, 1), None if t1 is None else round(t1, 1), round(f_min),
           oversample, segs)
    cache = S.spec_cache
    if key not in cache:
        if len(cache) > 12:
            cache.clear()
        if t0 is None:
            x, off = a.samples, 0.0
        else:
            pad = max(win_s, 30.0)   # extra context so background estimates have no edge effects
            lo = max(0.0, t0 - pad)
            x, off = a.segment(lo, t1 + pad), lo
        cache[key] = spectro.compute(x, a.sr, f_max=f_max, f_min=f_min, win_s=win_s, hop_s=hop_s,
                                     whiten=whiten, t_offset=off, oversample=oversample,
                                     segments=list(segs) if segs else None)
    return cache[key]


def active_cuts() -> list[float]:
    """Cut times (audio s) the user has enabled."""
    return sorted(float(c["t"]) for c in S.cuts if c.get("use", True))


def seg_of(t) -> int:
    return cutmod.segment_of(t, active_cuts())


def seg_bounds():
    return cutmod.segment_bounds(active_cuts(), S.audio.duration if S.audio is not None else 0)


def add_cut_markers(fig, t_lo, t_hi):
    for c in S.cuts:
        if not (t_lo <= c["t"] <= t_hi):
            continue
        on = c.get("use", True)
        fig.add_vline(x=c["t"], line=dict(color="#00e5ff" if on else "#607d8b", width=2 if on else 1,
                                          dash="dash" if on else "dot"),
                      annotation_text="✂" if on else "✂?", annotation_position="top",
                      annotation_font_color="#00e5ff" if on else "#90a4ae")


def spec_figure(s: spectro.Spectrogram, title="", zmax=None, overlays=(), height=460, field="z",
                selectable=False, show_cuts=True):
    mt, mf = DETAIL.get(S.get("disp_detail", "High"), (2400, 800))
    tt, ff, img = spectro.downsample_for_display(s, mt, mf, field)
    if zmax is None:
        zmax = float(np.percentile(img, 99.5)) if img.size else 1
    fig = go.Figure(go.Heatmap(x=tt, y=ff, z=img, colorscale=SPEC_CMAP, zmin=0 if field == "z" else None,
                               zsmooth="best" if S.get("disp_smooth", True) else False,
                               zmax=zmax, colorbar=dict(title="σ" if field == "z" else "dB"),
                               hovertemplate="t=%{x:.2f}s<br>f=%{y:.2f}Hz<br>%{z:.2f}<extra></extra>"))
    for ov in overlays:
        fig.add_trace(ov)
    if show_cuts and len(tt):
        add_cut_markers(fig, float(tt[0]), float(tt[-1]))
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
    st.subheader("Display")
    st.select_slider("Spectrogram detail", list(DETAIL), "High", key="disp_detail",
                     help="How many pixels are sent to the browser. 'Very high' is sharper but slower to draw.")
    st.checkbox("Smooth pixels", True, key="disp_smooth",
                help="Interpolate between pixels instead of drawing hard blocks")
    st.divider()
    st.subheader("Session")
    if S.events or S.cuts:
        st.download_button("💾 Save events + cuts (JSON)",
                           json.dumps({"events": S.events, "cuts": S.cuts}, indent=1, default=float),
                           "planesounds_session.json", "application/json")
    up = st.file_uploader("Load events JSON", type=["json"], key="ev_upload")
    if up is not None and st.button("Load events"):
        d = json.load(up)
        S.events = d.get("events", [])
        S.cuts = d.get("cuts", S.cuts)
        st.success(f"loaded {len(S.events)} events, {len(S.cuts)} cuts")
    st.divider()
    if S.audio is not None:
        a = S.audio
        st.markdown(f"**Audio:** {a.title or a.source}  \n{a.duration:.0f} s @ {a.sr} Hz")
        if a.upload_date:
            st.markdown(f"YouTube upload date: **{a.upload_date}**")
    if S.audio is not None:
        st.markdown(f"**Cuts:** {len(active_cuts())} → {len(active_cuts()) + 1} segment(s)")
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
            S.cuts, S.detected, S.events, S.result = [], None, [], None
            st.success(f"Loaded {a.duration:.1f} s of audio")
        except Exception as e:
            st.error(f"Could not load audio: {e}")
    demo_cut = b2.checkbox("Demo with an edit (500 s cut out of the middle)", False,
                           help="Simulates a cut YouTube video: 400-900 s is removed and the background "
                                "noise changes after the cut, so the cut detector has something to find.")
    if b2.button("🧪 Load synthetic demo scenario",
                 help="Two simulated propeller flyovers 920 s apart + background traffic, with matching "
                      "fake ADS-B data. Useful to learn the workflow and verify the pipeline."):
        sc = synth.demo_scenario()
        x = sc["audio"]
        if demo_cut:
            sr0 = sc["sr"]
            rng = np.random.default_rng(1)
            tail = x[900 * sr0:] * 1.0 + rng.standard_normal(len(x) - 900 * sr0).astype(np.float32) * 0.04
            x = np.concatenate([x[:400 * sr0], tail]).astype(np.float32)
        S.audio = au.Audio(x, sc["sr"], source="synthetic demo", title="Synthetic demo" + (" (edited)" if demo_cut else ""))
        S.flights = sc["flights"]
        S.demo = {k: sc[k] for k in ("obs", "rec_start", "duration")}
        S.spec_cache, S.trace, S.fit, S.sel, S.events = {}, None, None, None, []
        S.cuts, S.detected, S.result = [], None, None
        lat, lon, _ = sc["obs"]
        S.area = {"lat_min": lat - 0.3, "lat_max": lat + 0.3, "lon_min": lon - 0.4, "lon_max": lon + 0.4}
        st.success("Demo loaded. Truth (hidden from the search): "
                   f"{lat:.5f}, {lon:.5f}; recording started {fmt_utc(sc['rec_start'])}. "
                   + ("Events at ~148 s and ~590 s, with a cut at 400 s (500 s removed)."
                      if demo_cut else "Events are at ~148 s (≈100 Hz) and ~1090 s (≈88 Hz)."))
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
            ov_win = oc[1].select_slider("Window (s)", [0.25, 0.5, 1.0, 2.0, 4.0], 1.0, key="ov_win",
                                         help="Sets frequency sharpness: 1 s -> 1 Hz, 2 s -> 0.5 Hz. Short windows "
                                              "make tall vertical blocks.")
            ov_hop = oc[2].select_slider("Hop (s)", [0.1, 0.25, 0.5, 1.0, 2.0], 0.5, key="ov_hop",
                                         help="Time step between columns (smaller = finer time detail, slower)")
            ov_wh = oc[3].selectbox("Whitening", ["local", "freq", "both", "none"], 0, key="ov_wh",
                                    help="'local' normalises against the background of the surrounding minute, so "
                                         "quiet and loud parts are both readable. 'freq' uses one level for the whole "
                                         "recording (or per segment if cuts are marked). 'both' also removes clicks.")
            so = get_spec(ov_win, ov_hop, ov_fmax, ov_wh)
            mt, _ = DETAIL[S.get("disp_detail", "High")]
            col_s = max(ov_hop, a.duration / mt)
            st.caption(f"Resolution: {1 / ov_win:.2f} Hz × {col_s:.2f} s per pixel. "
                       "**Drag a box on this chart** to open that region in the full-resolution Zoom view below "
                       "(the mouse-wheel/zoom tool only magnifies the overview's pixels).")
            ov_ovl = []
            for d in (S.detected or []):
                ov_ovl.append(line(d["trace_t"], d["trace_f"], f"#{d['id']} CPA {d['t_cpa']:.0f}s", "#76ff03",
                                   width=3))
            ov_ev = st.plotly_chart(spec_figure(so, "Whole recording - dashed cyan = cuts, green = detected flyovers",
                                                overlays=ov_ovl, height=400, selectable=True),
                                    width="stretch", theme=None, on_select="rerun", selection_mode=("box",),
                                    key="ov_chart")
            try:
                ob = ov_ev["selection"]["box"][0] if ov_ev and ov_ev["selection"].get("box") else None
            except Exception:
                ob = None
            if ob:
                bx = (tuple(sorted(ob["x"])), tuple(sorted(ob["y"])))
                if bx != S.get("ov_box_last"):
                    S.ov_box_last = bx
                    (x0, x1), (y0, y1) = bx
                    S.sel = {"t0": max(0.0, x0), "t1": min(a.duration, max(x1, x0 + 5)),
                             "f0": max(0.0, y0), "f1": max(y1, y0 + 5)}
                    st.toast(f"Opened {x0:.0f}-{x1:.0f} s, {y0:.0f}-{y1:.0f} Hz in the Zoom view below")

        # ------------------------------------------------------------ cuts
        st.subheader("✂️ Cuts / edits")
        st.caption("Across a cut, the time gap between two plane sounds is unknown. Mark every cut: the search then "
                   "only uses time offsets *within* an uncut segment, and each segment gets its own start time.")
        cc = st.columns([1.2, 1.2, 1.6, 1.4])
        sens = cc[0].slider("Audio cut sensitivity", 0.3, 3.0, 1.0, 0.1,
                            help="Higher finds more (weaker) cuts, but also more false alarms")
        if cc[0].button("🔍 Detect cuts from audio"):
            with st.spinner("analysing background noise..."):
                found = cutmod.detect_audio_cuts(a.samples, a.sr, sensitivity=sens)
            old = [cutmod.Cut(c["t"], c["source"], c["strength"], c.get("audio_novelty", float("nan")), c.get("note", ""))
                   for c in S.cuts if c["source"] not in ("audio", "silence")]
            S.cuts = [dict(c.to_dict(), use=True) for c in cutmod.merge_cuts(old, found)]
            st.rerun()
        vthr = cc[1].slider("Video scene threshold", 0.1, 0.8, 0.35, 0.05,
                            help="ffmpeg scene-change score; lower finds more cuts")
        is_url = a.source.startswith(("http", "www.", "youtu"))
        can_video = is_url or bool(a.media_path and Path(a.media_path).suffix.lower() not in
                               (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".webm"))
        if cc[1].button("🎬 Detect cuts from video", disabled=not can_video,
                        help="Downloads a low-res copy of the video (URL) or uses the uploaded video file"):
            try:
                with st.spinner("getting video + detecting scene changes..."):
                    vp = au.download_video_lowres(a.source) if is_url else Path(a.media_path)
                    vc = cutmod.detect_video_cuts(vp, vthr, t_offset=a.start_offset)
                    vc = [c for c in vc if c.t <= a.duration]
                    vc = cutmod.annotate_with_audio(vc, a.samples, a.sr)
                old = [cutmod.Cut(c["t"], c["source"], c["strength"], c.get("audio_novelty", float("nan")), c.get("note", ""))
                       for c in S.cuts if "video" not in c["source"]]
                merged = cutmod.merge_cuts(old, vc)
                S.cuts = [dict(c.to_dict(), use=True) for c in merged]
                st.success(f"{len(vc)} scene cuts found")
                st.rerun()
            except Exception as e:
                st.error(f"video cut detection failed: {e}")
        man = cc[2].number_input("Add a cut manually at (s)", 0.0, a.duration, 0.0, 0.5)
        if cc[2].button("➕ Add manual cut"):
            S.cuts = sorted(S.cuts + [{"t": float(man), "source": "manual", "strength": 99.0,
                                       "audio_novelty": float("nan"), "note": "", "use": True}], key=lambda c: c["t"])
            st.rerun()
        if cc[3].button("🗑️ Clear all cuts", disabled=not S.cuts):
            S.cuts = []
            st.rerun()
        if S.cuts:
            cdf = pd.DataFrame(S.cuts)[["use", "t", "source", "strength", "audio_novelty", "note"]]
            ed = st.data_editor(cdf, key="cut_editor", hide_index=True, width="stretch",
                                column_config={"use": st.column_config.CheckboxColumn("use", help="untick false alarms"),
                                               "t": st.column_config.NumberColumn("time (s)", format="%.2f"),
                                               "strength": st.column_config.NumberColumn(format="%.1f"),
                                               "audio_novelty": st.column_config.NumberColumn("audio change (dB)", format="%.1f")},
                                disabled=["source", "strength", "audio_novelty", "note"])
            new_cuts = [dict(r) for r in ed.to_dict("records")]
            if [(c["t"], c["use"]) for c in new_cuts] != [(c["t"], c["use"]) for c in S.cuts]:
                S.cuts = sorted(new_cuts, key=lambda c: c["t"])
                st.rerun()
            st.caption("Segments: " + " | ".join(f"#{i}: {x0:.1f}-{x1:.1f} s" for i, (x0, x1) in enumerate(seg_bounds()))
                       + ".  Tip: video cuts marked *audio may be continuous* are often b-roll over unbroken sound - "
                         "untick them if the background noise really doesn't change.")
        else:
            st.caption("No cuts marked - the whole recording is treated as one continuous take.")

        # ------------------------------------------------------------ auto-detect
        st.subheader("🔎 Auto-detect flyovers")
        dc = st.columns(4)
        d_flo = dc[0].number_input("Lowest freq (Hz)", 10.0, 1000.0, 30.0, 5.0)
        d_fhi = dc[1].number_input("Highest freq (Hz)", 50.0, a.sr / 2 - 10, min(700.0, a.sr / 2 - 10), 10.0)
        d_str = dc[2].number_input("Min ridge strength (σ)", 0.5, 6.0, 1.5, 0.25,
                                   help="Lower to find fainter curves (more false alarms)")
        dc[3].markdown(" ")
        if dc[3].button("🔎 Scan whole recording", type="primary"):
            sd = get_spec(1.0, 0.25, min(d_fhi + 50, a.sr / 2), "freq")
            bar = st.progress(0.0, "scanning (never across a cut)...")
            det = ridge.detect_candidates(sd, cuts=active_cuts(), f_lo=d_flo, f_hi=d_fhi, min_strength=d_str,
                                          progress=lambda x: bar.progress(x, "scanning (never across a cut)..."))
            det.sort(key=lambda d: d["t_cpa"])
            for i, d in enumerate(det):
                d["id"] = i
                d["segment"] = seg_of(d["t_cpa"])
            S.detected = det
            st.rerun()
        if S.detected is not None:
            if not S.detected:
                st.warning("No flyovers detected. Try a lower strength threshold, or look manually.")
            else:
                existing = [e["t_cpa"] for e in S.events]
                ddf = pd.DataFrame([{"add": d.get("good", True) and all(abs(d["t_cpa"] - t) > 10 for t in existing),
                                     "quality": "✅ good" if d.get("good", True) else "⚠️ weak", "id": d["id"],
                                     "segment": d["segment"], "CPA (s)": d["t_cpa"], "f0 (Hz)": d["f0"],
                                     "speed (m/s)": d["v"], "CPA dist (m)": d["d"], "lines (Hz)": str(d["lines"]),
                                     "strength": d["strength"], "fit rms (Hz)": d["rms"]} for d in S.detected])
                st.caption("Only good-quality detections are ticked. Weak ones (poor fit, implausibly slow, faint) "
                           "are often noise - zoom in and check before adding them. One false event can make the "
                           "search find nothing.")
                ded = st.data_editor(ddf.round(2), key="det_editor", hide_index=True, width="stretch",
                                     disabled=[c for c in ddf.columns if c != "add"])
                b1c, b2c = st.columns(2)
                if b1c.button("➕ Add ticked detections as events", type="primary"):
                    c = c_sound()
                    n_add = 0
                    bounds = seg_bounds()
                    for row in ded.to_dict("records"):
                        if not row["add"]:
                            continue
                        d = S.detected[int(row["id"])]
                        fit = doppler.fit_linear(d["trace_t"], d["trace_f"], c, weights=d["trace_s"])
                        rg = fit.ranges()
                        sa, sb = bounds[seg_of(fit.t_cpa)]
                        harm = [h for h in d["harmonics_hint"] if abs(h - round(h)) < 0.05 and round(h) <= 4]
                        e = search.Event(name=f"auto {fit.t_cpa:.0f}s", t_cpa=fit.t_cpa, f0=fit.f0, v_range=rg["v"],
                                         d_range=rg["d"], t_window=(max(sa, fit.t_cpa - 60), min(sb, fit.t_cpa + 60)),
                                         f_band=(d["f_lo"], d["f_hi"]), harmonics=[float(round(h)) for h in harm] or [1.0],
                                         trace_t=list(map(float, d["trace_t"])), trace_f=list(map(float, d["trace_f"])))
                        S.events.append(e.to_dict())
                        n_add += 1
                    S.events.sort(key=lambda e: e["t_cpa"])
                    st.success(f"added {n_add} events - review them in tab 3")
                pick = b2c.selectbox("Zoom to detection", range(len(S.detected)),
                                     format_func=lambda i: f"#{i}: CPA {S.detected[i]['t_cpa']:.1f}s, {S.detected[i]['f0']:.1f} Hz")
                if b2c.button("🔍 Zoom to it"):
                    d = S.detected[pick]
                    S.sel = {"t0": float(d["t0"]), "t1": float(d["t1"]), "f0": float(d["f_lo"]), "f1": float(d["f_hi"])}
                    st.rerun()

        st.subheader("Zoom")
        sel = S.sel or {"t0": 0.0, "t1": min(a.duration, 240.0), "f0": 40.0, "f1": min(400.0, a.sr / 2)}
        zc = st.columns(6)
        z_t0 = zc[0].number_input("t from", 0.0, a.duration, float(sel["t0"]), 1.0)
        z_t1 = zc[1].number_input("t to", 0.0, a.duration, float(sel["t1"]), 1.0)
        z_f0 = zc[2].number_input("f from", 0.0, a.sr / 2, float(sel["f0"]), 1.0)
        z_f1 = zc[3].number_input("f to", 0.0, a.sr / 2, float(sel["f1"]), 1.0)
        z_win = zc[4].select_slider("Window (s)", [0.25, 0.5, 1.0, 2.0, 3.0, 4.0], 1.0,
                                    help="Longer = sharper frequency, blurrier time. 1-2 s works well for props.")
        z_c = zc[5].slider("Contrast", 0.2, 3.0, 1.0, 0.1)
        S.sel = {"t0": z_t0, "t1": max(z_t1, z_t0 + 5), "f0": z_f0, "f1": max(z_f1, z_f0 + 5)}
        z_hop = max(z_win / 8, (S.sel["t1"] - S.sel["t0"]) / 4000)
        sz = get_spec(z_win, z_hop, S.sel["f1"] + 5, S.get("ov_wh", "local"), S.sel["t0"], S.sel["t1"],
                      max(0, S.sel["f0"] - 5), oversample=4)
        sz = sz.crop(S.sel["t0"], S.sel["t1"], S.sel["f0"], S.sel["f1"])
        st.caption(f"Full resolution: {1 / z_win:.2f} Hz frequency resolution, {z_hop:.2f} s time step "
                   f"(whitening: {S.get('ov_wh', 'local')}).")
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
        crossing = [c for c in active_cuts() if sel["t0"] < c < sel["t1"]]
        if crossing:
            st.warning(f"⚠️ The selection crosses a cut at {', '.join(f'{c:.1f}' for c in crossing)} s. Audio on either "
                       "side comes from different moments - trim the selection to one side before tracing.")
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
                                 t_window=(max(seg_bounds()[seg_of(t_c)][0], t_c - twin),
                                           min(seg_bounds()[seg_of(t_c)][1], t_c + twin)),
                                 f_band=(sel["f0"], sel["f1"]), harmonics=hs or [1.0],
                                 trace_t=list(map(float, S.trace[0])), trace_f=list(map(float, S.trace[1])),
                                 score_mode=mode, allow_jets=jets)
                S.events.append(e.to_dict())
                S.events.sort(key=lambda d: d["t_cpa"])
                st.success(f"saved {name}")

        if S.events:
            st.subheader("Saved events")
            df = pd.DataFrame([{"segment": seg_of(e["t_cpa"]), **{k: v for k, v in e.items()
                                                                     if k not in ("trace_t", "trace_f")}}
                               for e in S.events])
            st.dataframe(df, width="stretch")
            if len(S.events) > 1:
                parts = []
                for e0, e1 in zip(S.events[:-1], S.events[1:]):
                    if seg_of(e0["t_cpa"]) == seg_of(e1["t_cpa"]):
                        parts.append(f"{e0['name']} → {e1['name']}: **{e1['t_cpa'] - e0['t_cpa']:.1f} s**")
                    else:
                        parts.append(f"{e0['name']} → {e1['name']}: **unknown (cut in between)**")
                st.markdown("Time offsets used by the search: " + " · ".join(parts))
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
    src = st.radio("Source", ["Flightradar24 (fr24 scraper, historical)", "adsb.lol daily archive (auto-download)",
                              "Local files / folder", "Upload", "OpenSky (Trino, needs account)"], horizontal=True)
    if src.startswith("Flightradar24"):
        st.markdown(
            "Uses the open-source [`fr24`](https://github.com/cathaypacific8747/fr24) Python package - the "
            "*'Python package to scrape the data'* route from the video. It is **historical**: it replays FR24's "
            "feed for your area every few minutes in the time window to find every aircraft, then downloads each "
            "one's **entire flight path**. Results are cached in `~/.planesounds/fr24`, so an interrupted run resumes.")
        st.caption("How far back you can go depends on your FR24 account (anonymous/free: roughly the last week; "
                   "paid plans: months to years). It's unofficial - keep the rate low and use it for personal fun.")
        audio_len = S.audio.duration if S.audio is not None else 1800.0
        f_lo = S.t0_range[0] - 900
        f_hi = S.t0_range[1] + audio_len + 900
        fc = st.columns(4)
        step_min = fc[0].number_input("Snapshot every (min)", 1.0, 30.0, 5.0, 1.0,
                                      help="Shorter catches fast aircraft that cross the area quickly; more requests")
        skip_jets = fc[1].checkbox("Skip jets", False,
                                   help="Much faster in busy airspace, but the silence filter can then only "
                                        "use propeller traffic")
        delay = fc[2].number_input("Pause between requests (s)", 0.0, 10.0, 0.6, 0.1)
        n_snap = int((f_hi - f_lo) / (step_min * 60)) + 1
        fc[3].metric("Snapshots to fetch", f"{n_snap:,}")
        with st.expander("Optional FR24 login (for older data)"):
            fu = st.text_input("FR24 email", key="fr24_user")
            fp = st.text_input("FR24 password", type="password", key="fr24_pw")
            st.caption("Used only for this download, sent only to Flightradar24 and never saved. You can instead set "
                       "the environment variables the fr24 package reads (see its docs).")
        st.caption(f"Window: {fmt_utc(f_lo)} → {fmt_utc(f_hi)} (your recording-start range + audio length "
                   "+ 15 min each side). Narrow the time range in the section above to make this much faster.")
        if (f_hi - f_lo) > 36 * 3600:
            st.warning("That's more than 36 h of snapshots - narrow the time window first.")
        if not fr24src.available():
            st.error("The `fr24` package isn't installed. Run: `pip install \"fr24[curl]\"` then restart the app.")
        elif st.button("⬇️ Scrape Flightradar24", type="primary"):
            bar = st.progress(0.0, "starting")
            bbox = (ar["lat_min"], ar["lat_max"], ar["lon_min"], ar["lon_max"])
            outp = adsb.DATA_DIR / f"fr24_{int(f_lo)}_{int(f_hi)}_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}.parquet"
            try:
                df_fr, stats = fr24src.scrape(bbox, f_lo, f_hi, step_s=step_min * 60, skip_jets=skip_jets,
                                              username=fu or None, password=fp or None, delay_s=delay,
                                              progress=lambda x, m: bar.progress(min(x, 1.0), m), out_path=outp)
                S.flights = df_fr
                st.success(f"{stats.summary()}. Saved to {outp}")
                if stats.errors:
                    with st.expander(f"{len(stats.errors)} warnings"):
                        st.code("\n".join(stats.errors))
            except Exception as e:
                st.error(f"Flightradar24 download failed: {e}")
    elif src == "Local files / folder":
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
        ev_objs = [search.Event(**e) for e in S.events]
        segs_used = sorted({seg_of(e.t_cpa) for e in ev_objs})
        if len(segs_used) > 1:
            st.info(f"Your events are in {len(segs_used)} different uncut segments ({segs_used}). Time offsets are only "
                    "enforced inside a segment. Without timing across a cut, each flight pins the location to a "
                    "*distance* from its track but not *where along it*, so expect several near-tied candidate "
                    "sites. Constraining how much time each cut removed helps a lot.")
            kc = st.columns(3)
            chrono = kc[0].checkbox("Segments are in chronological order", True,
                                    help="Usual for vlogs/edits: later in the video = later in real time")
            rm_max = kc[1].number_input("Max time removed per cut (min, 0 = unknown)", 0.0, 1440.0, 0.0, 1.0,
                                        help="e.g. 10 if you believe each cut skips at most ~10 minutes")
            rm_min = kc[2].number_input("Min time removed per cut (s)", 0.0, 86400.0, 0.0, 1.0)
            cut_removed = (rm_min, rm_max * 60 if rm_max > 0 else None) if chrono else None
        else:
            cut_removed = (0.0, None)
        spans_default = search.default_clean_spans(ev_objs, a.duration, 15.0, active_cuts())
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
                alt_offset_m=alt_off, cuts=active_cuts(), cut_removed_s=cut_removed)
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
                sites = res.sites if res.sites is not None else best
                if S.demo:
                    sites = sites.assign(err_m=haversine_m(o[0], o[1], sites.lat, sites.lon))
                top = sites.head(50)
                fm = go.Figure(go.Scattermap(lat=top.lat, lon=top.lon, mode="markers",
                                             marker=dict(size=10, color=top.score, colorscale="Viridis", showscale=True),
                                             text=[f"#{i} score {s:.2f}" for i, s in zip(top.index, top.score)]))
                fm.update_layout(map=dict(style="open-street-map", center=dict(lat=top.lat.iloc[0], lon=top.lon.iloc[0]),
                                          zoom=9), height=420, margin=dict(l=0, r=0, t=0, b=0))
                c2.plotly_chart(fm, width="stretch", theme=None)
                near = int((sites.rel_score > 0.98).sum()) if "rel_score" in sites else 1
                if near > 1:
                    st.warning(f"{near} distinct sites score within 2 % of the best one - treat them all as candidates "
                               "(check each in tab 6 / Street View).")
                st.markdown("**Candidate sites** (distinct locations, ≥750 m apart per flight combination)")
                cols = ["lat", "lon", "score", "rel_score"] + [f"score_{k}" for k in range(len(evs))] + ["start_utc"] + \
                       [c for c in sites.columns if c.startswith("removed_")] + \
                       sum([[f"callsign_{k}", f"type_{k}", f"d_{k}", f"speed_{k}"] for k in range(len(evs))], []) + \
                       (["err_m"] if "err_m" in sites else [])
                st.dataframe(sites[[c for c in cols if c in sites]].head(200), width="stretch")
                if any(c.startswith("removed_") for c in sites.columns):
                    st.caption("removed_sA_sB = how many seconds of real time the cut between segment A and B skipped "
                               "(estimated from the flight timings).")
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
        best = res.sites if res.sites is not None else res.best_per_combo
        i = st.selectbox("Candidate site", range(min(len(best), 100)),
                         format_func=lambda i: f"#{i}: score {best.score[i]:.2f} @ {best.lat[i]:.5f}, {best.lon[i]:.5f}")
        row = best.iloc[i]
        lat, lon = float(row.lat), float(row.lon)
        ev_T0 = [float(row[f"T0_{k}"]) if f"T0_{k}" in row else float(row.T0) for k in range(len(evs))]
        seg_cols = sorted(c for c in row.index if c.startswith("T0_s"))
        if len(seg_cols) > 1:
            st.markdown("Segment start times (audio t=0 of each uncut segment): " + " · ".join(
                f"segment {c[4:]}: **{fmt_utc(row[c])}**" for c in seg_cols)
                + "  \n(these are the times audio t=0 *would* correspond to if that segment ran from the start)")
        st.markdown(f"**{lat:.6f}, {lon:.6f}** · recording started ≈ **{fmt_utc(row.T0)}** · "
                    f"[Google Maps](https://www.google.com/maps/search/?api=1&query={lat},{lon}) · "
                    f"[Street View](https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}) · "
                    f"[Satellite](https://www.google.com/maps/@{lat},{lon},400m/data=!3m1!1e3)")
        trs = [res.tracks[row[f"fid_{k}"]] for k in range(len(evs))]
        st.dataframe(pd.DataFrame([{"event": e.name, "flight": t.fid, "callsign": t.callsign, "type": t.typecode,
                                    "class": t.propulsion, "score": row[f"score_{k}"], "slant CPA (m)": row[f"d_{k}"],
                                    "speed (m/s)": row[f"speed_{k}"], "f0 best": row[f"f0_{k}"]}
                                   for k, (e, t) in enumerate(zip(evs, trs))]), width="stretch")
        scorer = search.Scorer(spec_full, evs, cfg.c, cfg.dt_sweep_s, cfg.f0_sweep_frac, cfg.cuts)
        obs = lla_to_ecef(lat, lon, row.elev)
        curves = scorer.model_curves(trs, obs, ev_T0, [row[f"f0_{k}"] for k in range(len(evs))])
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
            m = np.abs(t.t - (ev_T0[k] + evs[k].t_cpa)) < 400
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
                                     cfg.dt_sweep_s, cfg.f0_sweep_frac, progress=lambda x: bar.progress(x),
                                     cuts=cfg.cuts)
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
    st.markdown(Path(__file__).with_name("README.md").read_text(encoding="utf-8") if Path(__file__).with_name("README.md").exists() else "")
