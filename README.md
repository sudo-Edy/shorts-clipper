# shorts-clipper

Finds the best 15–60 second Shorts moments in a gameplay recording and writes a
ranked CSV. It only reads your video; the one file it writes is the CSV.

## Setup (Windows, once)

```
pip install faster-whisper numpy
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"   # GPU transcription (RTX cards)
```
ffmpeg must be on PATH. The first run downloads the Whisper model (large-v3 ≈ 3 GB).

## Use

1. While recording, jot hype moments in a text file, one per line:
   ```
   12:34 clutch 1v3
   1:02:03 - boss down
   ```
2. Run:
   ```
   python clipfinder.py "D:\recordings\session1.mp4" --notes marks.txt
   ```
3. Open `session1_shorts.csv`: rank, start, end, duration, score, why, transcript.

4. In DaVinci Resolve (or CutPilot's timeline), put the recording at the start of a
   timeline, then **Timeline > Import > Timeline Markers from EDL** and pick
   `session1_shorts.edl`: every clip becomes a duration marker (red = top 5,
   yellow = 6-10, blue = the rest) named with its reason. Use `--fps` if the
   timeline's frame rate differs from the video's, and `--timeline-start-hour 0`
   if the timeline starts at 00:00:00:00. (Format follows Resolve's marker-EDL
   export; not yet confirmed by an import on this PC.)

Options: `--count 20` (max clips), `--model medium` (faster, less accurate),
`--no-transcribe` (marks + loudness only), `--out path.csv`.

## Cutting real clip files (demo packs, prospect sends)

The CSV/EDL above are for editing in Resolve. To get actual playable clip
files (e.g. a demo pack to send prospects), add `--export-dir`:

```
python clipfinder.py "D:\recordings\session1.mp4" --notes marks.txt \
    --export-dir shorts-demo/rough-cuts --export-count 8
```

This cuts the top 5-8 ranked clips (same ranking as the CSV) into
`shorts-demo/rough-cuts/clip_01.mp4` ... and a `manifest.json` listing, per
clip: `source_start`/`source_end`, `duration_s`, `moment` (why it was picked),
a transcript excerpt, and a `suggested_caption` grounded in the detected mark
label or reaction words (never invented). It refuses to run if any target
file already exists - point it at a fresh/empty folder.

## How clips are ranked

| Signal | Points |
|---|---|
| Near one of your marks (±20 s) | 50 |
| Audio spike (loud vs. the recording's own baseline) | up to 30 |
| Reaction words in the transcript ("no way", "let's go", "clutch"…) | up to 20 |
| Constant commentary | 5 |

Each clip starts ~10 s before the moment (the setup), runs ~30 s, and its edges
snap to pauses in speech so nothing is cut mid-word. Overlapping clips are
dropped; a quiet recording is topped up to 15 with its loudest remaining
stretches. A 6-minute test recording takes ~12 s on an RTX 4090 with large-v3.

## Demo (no real recording needed)

`examples/` has a synthetic-recording demo you can reproduce anywhere ffmpeg +
numpy are installed, no GPU or gameplay footage required:

```
python examples/generate_demo.py                          # writes examples/demo/session.wav + marks.txt
python clipfinder.py examples/demo/session.wav --notes examples/demo/marks.txt \
    --no-transcribe --out examples/demo_output.csv
```

`examples/demo_output.csv` / `.edl` in this repo are real output from that
15-minute synthetic recording (16 ranked clips, 30 s each, marked moments
ranked first); `--no-transcribe` is used because the demo audio has no speech,
not because transcription doesn't work — `test_end_to_end_cli_writes_15_to_20_clip_csv`
in `test_clipfinder.py` runs the same CLI path on every test run and asserts
the CSV lands in the 15-20 row range.
