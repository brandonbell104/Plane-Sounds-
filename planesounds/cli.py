"""Command-line interface.

    python -m planesounds fetch URL [-o out.wav]           download + decode audio
    python -m planesounds cuts FILE_OR_URL [--video]       find edit points (audio, optionally video)
    python -m planesounds detect FILE_OR_URL [--cuts 400 812]   list candidate Doppler curves
    python -m planesounds measure FILE_OR_URL --t0 --t1 --flo --fhi [--name] [--events events.json]
    python -m planesounds adsblol --day 2025-09-08 --bbox LATMIN LATMAX LONMIN LONMAX -o flights.parquet
    python -m planesounds search FILE_OR_URL --events events.json --flights flights.parquet \
        --bbox ... --start 2025-09-08T00:00 --end 2025-09-08T23:59 [-o results.csv]
    python -m planesounds demo                             run the synthetic end-to-end test
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

from . import adsb, audio, cuts as cutmod, doppler, ridge, search, spectro, synth
from .dem import ConstantDEM, SRTMDEM
from .geo import haversine_m


def _ts(s: str) -> float:
    return pd.Timestamp(s, tz="UTC").timestamp() if pd.Timestamp(s).tzinfo is None else pd.Timestamp(s).timestamp()


def _progress(x, msg):
    sys.stderr.write(f"\r[{x * 100:5.1f}%] {msg[:90]:<90}")
    sys.stderr.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="planesounds", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch")
    p.add_argument("src")
    p.add_argument("-o", "--out", default="audio.wav")
    p.add_argument("--sr", type=int, default=8000)

    p = sub.add_parser("cuts")
    p.add_argument("src")
    p.add_argument("--sr", type=int, default=8000)
    p.add_argument("--sensitivity", type=float, default=1.0)
    p.add_argument("--video", action="store_true", help="also run scene-cut detection on the video")
    p.add_argument("--threshold", type=float, default=0.35)

    p = sub.add_parser("detect")
    p.add_argument("src")
    p.add_argument("--cuts", type=float, nargs="*", default=[])
    p.add_argument("--sr", type=int, default=8000)
    p.add_argument("--fmax", type=float, default=700)

    p = sub.add_parser("measure")
    p.add_argument("src")
    for k in ("t0", "t1", "flo", "fhi"):
        p.add_argument(f"--{k}", type=float, required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--events", default="events.json")
    p.add_argument("--window", type=float, default=60)
    p.add_argument("--sr", type=int, default=8000)
    p.add_argument("--temp", type=float, default=15.0)

    p = sub.add_parser("adsblol")
    p.add_argument("--day", required=True)
    p.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    p.add_argument("-o", "--out", required=True)

    p = sub.add_parser("search")
    p.add_argument("src")
    p.add_argument("--events", required=True)
    p.add_argument("--flights", required=True, nargs="+")
    p.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    p.add_argument("--start", required=True, help="earliest UTC time of audio t=0")
    p.add_argument("--end", required=True, help="latest UTC time of audio t=0")
    p.add_argument("--grid", type=float, default=200)
    p.add_argument("--refine", type=float, default=100)
    p.add_argument("--tol", type=float, default=6)
    p.add_argument("--elev", type=float, default=0.0, help="constant ground elevation (m)")
    p.add_argument("--srtm", action="store_true", help="use SRTM elevation (auto-download)")
    p.add_argument("--no-silence", action="store_true")
    p.add_argument("--cuts", type=float, nargs="*", default=[], help="audio times of edits")
    p.add_argument("--max-removed", type=float, default=None, help="max seconds removed at each cut")
    p.add_argument("--temp", type=float, default=15.0)
    p.add_argument("--sr", type=int, default=8000)
    p.add_argument("-o", "--out", default="results.csv")

    sub.add_parser("demo")
    a = ap.parse_args(argv)

    if a.cmd == "fetch":
        x = audio.load_source(a.src, a.sr)
        audio.write_wav(a.out, x.samples, x.sr)
        print(f"wrote {a.out} ({x.duration:.1f} s) - {x.title} uploaded {x.upload_date}")

    elif a.cmd == "cuts":
        x = audio.load_source(a.src, a.sr)
        found = cutmod.detect_audio_cuts(x.samples, x.sr, a.sensitivity)
        if a.video:
            vp = audio.download_video_lowres(a.src) if a.src.startswith("http") else a.src
            vc = cutmod.annotate_with_audio(cutmod.detect_video_cuts(vp, a.threshold), x.samples, x.sr)
            found = cutmod.merge_cuts(found, vc)
        for c in found:
            print(f"{c.t:9.2f} s  {c.source:<12} strength {c.strength:6.1f}  {c.note}")
        print("--cuts " + " ".join(f"{c.t:.2f}" for c in found))

    elif a.cmd == "detect":
        x = audio.load_source(a.src, a.sr)
        s = spectro.compute(x.samples, x.sr, f_max=min(a.fmax + 50, x.sr / 2))
        for d in sorted(ridge.detect_candidates(s, cuts=a.cuts, f_hi=a.fmax), key=lambda d: d["t_cpa"]):
            print(f"{'OK  ' if d['good'] else 'weak'} seg {d['segment']}  CPA {d['t_cpa']:8.1f} s  f0 {d['f0']:7.1f} Hz  "
                  f"v {d['v']:5.1f} m/s  d {d['d']:6.0f} m  lines {d['lines']}  "
                  f"box {d['t0']:.0f}-{d['t1']:.0f}s {d['f_lo']:.0f}-{d['f_hi']:.0f}Hz")

    elif a.cmd == "measure":
        x = audio.load_source(a.src, a.sr)
        c = doppler.speed_of_sound(a.temp)
        s = spectro.compute(x.samples, x.sr, f_max=a.fhi + 20)
        t, f, st = ridge.clean_trace(*ridge.trace_ridge(s, a.t0, a.t1, a.flo, a.fhi))
        fit = doppler.fit_linear(t, f, c, weights=st)
        rg = fit.ranges()
        evp = Path(a.events)
        evs = json.loads(evp.read_text())["events"] if evp.exists() else []
        e = search.Event(name=a.name or f"event {len(evs) + 1}", t_cpa=fit.t_cpa, f0=fit.f0, v_range=rg["v"],
                         d_range=rg["d"], t_window=(fit.t_cpa - a.window, fit.t_cpa + a.window),
                         f_band=(a.flo, a.fhi), trace_t=list(t), trace_f=list(f))
        evs.append(e.to_dict())
        evp.write_text(json.dumps({"events": sorted(evs, key=lambda d: d["t_cpa"])}, indent=1))
        print(f"f0={fit.f0:.2f} Hz v={fit.v:.1f} m/s d={fit.d:.0f} m CPA={fit.t_cpa:.2f}s rms={fit.rms_hz:.2f} Hz"
              f" -> saved to {evp}")

    elif a.cmd == "adsblol":
        df = adsb.download_adsblol_day(dt.date.fromisoformat(a.day), tuple(a.bbox), out_path=a.out,
                                       progress=_progress)
        print(f"\n{df.flight_id.nunique()} flights -> {a.out}")

    elif a.cmd == "search":
        x = audio.load_source(a.src, a.sr)
        evs = [search.Event(**e) for e in json.loads(Path(a.events).read_text())["events"]]
        fl = pd.concat([adsb.load_table(p) if not Path(p).is_dir() else adsb.load_readsb_traces([p], tuple(a.bbox))
                        for p in a.flights], ignore_index=True)
        fmax = max(e.f_band[1] * max(e.harmonics) for e in evs) * 1.1 + 10
        s = spectro.compute(x.samples, x.sr, f_max=min(fmax, x.sr / 2))
        cfg = search.SearchConfig(bbox=tuple(a.bbox), t0_range=(_ts(a.start), _ts(a.end)), duration=x.duration,
                                  grid_m=a.grid, refine_m=a.refine, offset_tol_s=a.tol,
                                  c=doppler.speed_of_sound(a.temp), silence=not a.no_silence, cuts=a.cuts,
                                  cut_removed_s=(0.0, a.max_removed))
        dem = SRTMDEM(fallback=a.elev) if a.srtm else ConstantDEM(a.elev)
        res = search.run_search(fl, evs, s, cfg, dem, progress=_progress)
        print("\n" + "\n".join(res.log))
        if not res.results.empty:
            res.results.to_csv(a.out, index=False)
            print(res.sites.head(10)[["lat", "lon", "score", "rel_score", "start_utc"]].to_string())
            print(f"full results -> {a.out}")

    elif a.cmd == "demo":
        sc = synth.demo_scenario()
        s = spectro.compute(sc["audio"], sc["sr"], f_max=400)
        evs = []
        for k, (t0, t1, flo, fhi) in enumerate([(60, 260, 60, 140), (980, 1180, 50, 120)]):
            t, f, st = ridge.clean_trace(*ridge.trace_ridge(s, t0, t1, flo, fhi))
            fit = doppler.fit_linear(t, f)
            rg = fit.ranges()
            print(f"event {k + 1}: f0={fit.f0:.1f} v={fit.v:.1f} d={fit.d:.0f} CPA={fit.t_cpa:.1f}")
            evs.append(search.Event(f"event {k + 1}", fit.t_cpa, fit.f0, rg["v"], rg["d"],
                                    (fit.t_cpa - 60, fit.t_cpa + 60), (flo, fhi)))
        o = sc["obs"]
        cfg = search.SearchConfig(bbox=(o[0] - 0.3, o[0] + 0.3, o[1] - 0.4, o[1] + 0.4),
                                  t0_range=(sc["rec_start"] - 6 * 3600, sc["rec_start"] + 6 * 3600),
                                  duration=sc["duration"])
        res = search.run_search(sc["flights"], evs, s, cfg, ConstantDEM(o[2]), progress=_progress)
        print("\n" + "\n".join(res.log))
        top = res.results.iloc[0]
        print(f"best: {top.lat:.5f}, {top.lon:.5f}  truth: {o[0]:.5f}, {o[1]:.5f}  "
              f"error {haversine_m(o[0], o[1], top.lat, top.lon):.0f} m")


if __name__ == "__main__":
    main()
