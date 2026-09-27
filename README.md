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

Options: `--count 20` (max clips), `--model medium` (faster, less accurate),
`--no-transcribe` (marks + loudness only), `--out path.csv`.

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
