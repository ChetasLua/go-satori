# GO (吾 / 悟)

A 118-second anime fan film set in the world of *Dragon Ball*, drawn, animated, scored and sung entirely in code.
1920×1080, 24 fps, 41 shots, 144 BPM.

吾 (read *go*) means "I". Put a heart (忄) beside it and it becomes 悟 (also *go*): to awaken, and the mark on Goku's chest.

No image models, no image files, no samples, no text-to-speech, no fonts on screen. The picture is Python (numpy, opencv, scipy, pycairo, numba, Pillow); the sound is JavaScript for Node with no dependencies.

Unofficial fan work. *Dragon Ball* and its characters are by Akira Toriyama; this project is not made or endorsed by the author or the rights holders.

## Build

pycairo has no macOS wheel, so the environment comes from conda-forge ([micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)). Run from the repository root. Sound needs Node 24 or newer.

```bash
micromamba create -y -p .mm/env -c conda-forge python=3.12 numpy opencv pycairo scipy pillow numba ffmpeg pysoundfile
```

Picture (plates are painted and cached on the first run, so it is slow the first time):

```bash
./.mm/env/bin/python tools/film.py --jobs 4
```

Picture events for the sound, then voice, band, effects and mix:

```bash
./.mm/env/bin/python tools/events.py
```

```bash
node audio/voice.mjs --jobs 2 && node audio/voice_wordless.mjs
```

```bash
node audio/band.mjs all
```

```bash
node --expose-gc audio/mix.mjs
```

Join them into `dist/go.mp4`:

```bash
./.mm/env/bin/python tools/film.py --assemble --crf 16
```

One frame:

```bash
./.mm/env/bin/python tools/still.py g3 60
```

## What is where

| Path | What it does |
|---|---|
| `film.py` | the timeline: every shot, its start and length |
| `db/hero3.py` `h3_*.py` | posed 3-D fighters redrawn as inked cels from any camera: poses, contact, smears |
| `db/ki*.py` | auras, beams, lightning, impact frames, trails |
| `db/phys*.py` | debris, dust, water, rain, blast and pressure waves |
| `db/land*.py` `sky*.py` `moon*.py` | places, cloud worlds, orbit, the moon |
| `db/letters*.py` | EGO / GO monument, manga page, brush and seal |
| `db/ball.py` `nimbus.py` `glove.py` `rag.py` `props.py` `storm.py` | the glass ball, the cloud, hands, cloth, props, rain |
| `chars/` | the characters |
| `shots/` | the 41 shots |
| `audio/` | score, singer, band, effects, mix |
