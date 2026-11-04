# ✈️ PlaneSounds: geolocate a recording from aircraft Doppler curves

PlaneSounds runs on your own PC. Give it a YouTube link (or any audio/video file) and it pulls out
the faint propeller-aircraft **Doppler curves** hidden in the background audio. It then compares
them against historical **ADS-B flight tracks** to work out where the microphone was. This follows
the method shown in the video *"The most powerful geolocation clue"*.

You usually can't *hear* these curves, but a high-resolution, whitened, low-frequency spectrogram
shows them. Each curve tells you three things:

| From the curve | Physics |
|---|---|
| **How big the frequency drop is** | the aircraft's **speed** |
| **How quickly the drop happens** | the **slant distance** at closest approach (CPA) |
| **When the midpoint happens** | the **delayed CPA time** (sound needs time to reach you) |
| **The small wiggles** | the exact flight path: turns, speed changes, climbs |

Two curves a known number of seconds apart, plus the *silence* between them, narrow the location
down a lot. The last step models the exact curve each real trajectory would have produced at each
candidate spot, then scores it against the spectrogram. Ideally that gets you to within roughly 100 m.

---

## Install (Windows / macOS / Linux)

1. Install **Python 3.10+** (https://www.python.org/downloads/; on Windows tick *"Add to PATH"*).
2. Install **ffmpeg**:
   * Windows: `winget install ffmpeg` (or download it from https://ffmpeg.org and add it to PATH)
   * macOS: `brew install ffmpeg`
   * Linux: `sudo apt install ffmpeg`
3. Start it:
   * Windows: double-click **`run_windows.bat`**
   * macOS/Linux: `./run.sh`

   The first run creates a virtual environment and installs the dependencies. Your browser then
   opens at http://localhost:8501.

Manual alternative:

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

## Try the demo first

On tab **1 · Audio**, click **🧪 Load synthetic demo scenario**. It simulates two propeller
flyovers 920 s apart, background jets and props, and matching fake ADS-B data. Then:

1. **2 · Spectrogram**: set the zoom to `60–260 s`, `60–140 Hz`. You can also drag a box on the
   chart and click *Use this box*.
2. **3 · Measure events**: click **✏️ Trace + fit**, then **➕ Save event**.
3. Do the same for `980–1180 s`, `50–120 Hz`.
4. **5 · Search**: click **🚀 Run search**. The top result should land within ~50 m of the hidden
   truth, which **6 · Inspect results** shows as a green dot.

## Workflow for a real video

### 1. Audio
Paste the YouTube URL. yt-dlp downloads the best audio stream, and ffmpeg decodes it to mono
8 kHz **without** any normalisation or noise reduction. Check the metadata (title, description,
upload date) for clues about the date and the area. The upload date is only the *latest* possible
recording date.

### 2. Find the curves
* The overview shows the whole recording. Look for faint, slowly **descending** lines between
  roughly 40 and 400 Hz. Propeller blade-pass tones usually sit at 60–150 Hz and come with
  harmonics. **🔎 Auto-detect** suggests candidates.
* Zoom in. A longer **window** (1–4 s) gives sharper frequency resolution, which makes very faint
  tones pop out. **Whitening** removes constant hum.
* **Listen**: band-pass the audio, boost it and speed it up to confirm it really is an aircraft.
  A propeller sounds like a buzzing drone.

### 3. Measure each event
**Trace + fit** finds the brightest continuous, monotonically falling ridge inside your box and
fits a straight-line Doppler model to it. This gives f₀, speed, CPA slant distance and CPA time.
The manual sliders let you check the fit by eye, like in the video. Before saving, set generous
± ranges, because a filter that cuts out the truth can never recover it. Notes:
* Trace **one** line. Use the clearest harmonic, and list other visible harmonics (e.g. `1, 2`
  relative to the traced line) so the scorer uses them too.
* **Score mode `trace`** compares candidate curves against your traced line instead of spectrogram
  brightness. Use it for curves so faint that the brightness is mostly noise.
* Save every clear pass-by. Two or more events with known gaps are extremely constraining.

### 4. Area, time and flight data
* **Area**: a centre plus radius, a bounding box, or the continental US. Any extra clue (city,
  state, language, coastline) shrinks the problem enormously.
* **Time**: the possible range for *audio t = 0* in **UTC**. A known day works with two or more
  events. A known exact time (± a few minutes) can be enough with a single event.
* **ADS-B data** (one row per position report). Options:
  * **adsb.lol archive (auto)**: streams one UTC day of the free community
    [`globe_history`](https://github.com/adsblol) dump (several GB) from GitHub, keeps only your
    area and saves a small parquet under `~/.planesounds/adsb/`. Set a `GITHUB_TOKEN`
    environment variable if you hit GitHub API rate limits.
  * **Local files**: readsb/tar1090 `trace_full_*.json` (plain or `.gz`), folders of them, `.tar`
    archives, OpenSky CSV/parquet, FlightRadar24 per-flight CSV exports, or any CSV with
    time/lat/lon/alt columns.
  * **OpenSky Trino**: `pip install pyopensky`. This needs an OpenSky account with historical
    access.
  * If type codes are missing, merge them from an aircraft database CSV (e.g. OpenSky's
    `aircraftDatabase.csv`). Unknown types are always kept. Only known jets are dropped.

### 5. Search
The search runs as a funnel. Each stage is cheaper than the next and narrows the candidates:

1. **Flight pre-filter**: keeps flights in the area and time window, and drops known jets.
2. **Speed filter**: keeps only track samples whose 3-D speed is inside each event's range.
3. **Slant-range candidates**: grid cells (200 m by default) whose CPA slant distance to a
   speed-matching flight is inside the measured range. Ground elevation comes from the DEM,
   because ADS-B altitudes are above sea level.
4. **Time-offset filter**: the delayed CPA times (CPA emission time + distance / speed of sound)
   must reproduce the gaps between events, within the tolerance, and fall inside your time window.
5. **Silence filter**: rejects any cell that would have clearly heard *another* aircraft during
   the quiet parts of the recording. Silence is a signal too, and it removes most candidates near
   airports.
6. **Doppler scoring**: refines each survivor to a finer grid (100 m), then models the exact
   received curve from the real trajectory. The model uses the aircraft's velocity at the moment
   it *emitted* the sound, not at the moment you heard it, because otherwise the curve has the
   wrong shape. It then adds up the spectrogram brightness under the curve. Event 1's time and
   every f₀ are swept to absorb measurement error. Later events are **time-locked** to event 1 by
   the flight geometry, so a wrong offset really does lower the score.

The results show a scatter plot of event-1 vs event-2 scores (top-right is best), the best point
per flight combination, a map and a CSV export.

### 6. Inspect
For each candidate you get spectrograms with the modelled curves overlaid, the flight tracks on a
map, and Google Maps / Street View links. The **fine heat-map** re-scores a 25 m grid around a
candidate for that flight combination. Then compare Street View with the video.

---

## Command line

```bash
python -m planesounds demo
```

```bash
python -m planesounds detect "https://youtu.be/..."
```

```bash
python -m planesounds measure "https://youtu.be/..." --t0 1490 --t1 1640 --flo 60 --fhi 140 --events ev.json
```

```bash
python -m planesounds adsblol --day 2025-09-08 --bbox 40.5 41.2 -74.3 -73.2 -o flights.parquet
```

```bash
python -m planesounds search "https://youtu.be/..." --events ev.json --flights flights.parquet --bbox 40.5 41.2 -74.3 -73.2 --start 2025-09-08T00:00 --end 2025-09-08T23:59 --srtm
```

Tests:

```bash
python -m pytest -q tests
```

## Tips and caveats

* **Speed of sound**: set the air temperature in the sidebar. Wind is not modelled; a strong wind
  shifts timings by a few seconds, so increase the tolerance.
* **Altitudes**: geometric (GNSS) altitude is used when available, otherwise barometric altitude.
  Barometric altitude can be off by 100 m or more depending on the weather; use the altitude
  offset setting if you know QNH.
* **Coverage**: ADS-B only sees transponder-equipped aircraft that a receiver picked up. A plane
  with no ADS-B can't be matched, and it can also make the silence filter wrong. If unknown
  aircraft are audible, edit the quiet spans.
* **Cost**: the candidate stages scale with (number of matching flights × area). A city-sized area
  takes seconds. A whole country for a whole day needs a big machine; the reference video used
  50 GB of data. Narrow the area and the time wherever you can.
* **Edited videos**: cuts break the time offsets between events. Measure events only within one
  continuous shot.
* **Fair play**: this is a fun forensic technique. Use it on games and public footage, not to track
  people.

## Project layout

```
app.py                     Streamlit UI
planesounds/audio.py       yt-dlp download, ffmpeg decode, band-pass "listen" helper
planesounds/spectro.py     whitened high-resolution spectrograms, bilinear sampling
planesounds/ridge.py       Viterbi ridge tracing, automatic candidate detection
planesounds/doppler.py     Doppler physics (retarded time), straight-line fit, trajectory model
planesounds/adsb.py        ADS-B loaders (readsb traces, adsb.lol, OpenSky, FR24, CSV)
planesounds/aircraft_types.py  jet / prop / heli classification of ICAO type codes
planesounds/dem.py         ground elevation: constant or SRTM tiles (auto-download)
planesounds/geo.py         WGS84/ECEF, global integer grid
planesounds/search.py      the filtering + scoring pipeline
planesounds/synth.py       synthetic scenario generator (demo + tests)
planesounds/cli.py         command line
```
