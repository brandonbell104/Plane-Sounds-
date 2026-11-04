"""PlaneSounds - geolocate audio recordings from aircraft Doppler signatures.

Pipeline overview
-----------------
1. ``audio``      : fetch audio (YouTube via yt-dlp, or a local file) and resample.
2. ``spectro``    : high-resolution, whitened low-frequency spectrograms.
3. ``ridge``      : trace faint tonal ridges (Doppler curves) through the spectrogram.
4. ``doppler``    : physical Doppler model (with retarded/emission time) + curve fitting.
5. ``adsb``       : load / download historical ADS-B tracks.
6. ``dem``        : ground elevation (constant, SRTM .hgt files, or auto-downloaded tiles).
7. ``search``     : candidate grid -> speed / slant-range / time-offset / silence filters
                    -> full trajectory Doppler modelling and spectrogram scoring.
"""

__version__ = "0.1.0"
