# shorts-clipper

Finds the best 15–60 second Shorts moments in a gameplay recording and writes a
ranked CSV. It only reads your video; it writes the CSV + a Resolve marker EDL, and (with
`--render N`) the finished vertical Shorts.

**Stack:** Python, faster-whisper (GPU), NumPy, ffmpeg with NVENC · **Tests:** `python -m pytest` (20 tests, no GPU or footage needed)

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
   if the timeline starts at 00:00:00:00. (Format follows Resolve's own marker-EDL
   export.)

5. **Render the Shorts themselves** (optional): add `--render 5` and the top 5
   come out as finished 1080x1920 mp4s in `session1_shorts_clips/`, with big
   word-by-word captions burned in (3 words max, never flashing).
   ```
   python clipfinder.py "D:\recordings\session1.mp4" --notes marks.txt --render 5
   ```
   - `--layout crop` (default) fills the frame with a 9:16 slice; `--crop-x 0.2`
     slides it left (e.g. toward a facecam). `--layout fit` shows the whole
     gameplay frame over a blurred copy of itself, so no HUD is lost.
   - `--audio-track 2` picks which OBS track goes into the Short (default: the first).
   - `--no-captions` for clean clips. GPU (NVENC) encode, CPU fallback.
   - Real run: 20-min 1080p OBS recording -> transcript (large-v3, RTX 4090) +
     ranked CSV + 3 rendered Shorts in 2 min 19 s.

Options: `--count 20` (max clips), `--model medium` (faster, less accurate),
`--no-transcribe` (marks + loudness only), `--out path.csv`.

## Tool mode (used by CreatorOS's Shorts engine)

```
python clipfinder.py video.mp4 --transcribe-only --words-out words.json      # word-timed transcript only
python clipfinder.py video.mp4 --words-in words.json --clips spec.json --clips-dir out/ --layout fit
```
`spec.json` is `[{"start": 12.3, "end": 51.0, "hook": "Top text", "name": "01 my clip", "accent": "FFD400"}]`.
Each clip is rendered exactly as given: 1080x1920, word captions, the hook title at the top (inside every
platform's safe zone), tinted with that clip's accent. It prints `{"rendered": [paths]}` on stdout.

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
