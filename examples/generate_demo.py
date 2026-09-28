#!/usr/bin/env python3
"""Generate a synthetic 15-minute recording + marks file to demo clipfinder.py
without a real gameplay capture. Writes examples/demo/session.wav (gitignored,
~28 MB) and examples/demo/marks.txt.

    python examples/generate_demo.py
    python clipfinder.py examples/demo/session.wav --notes examples/demo/marks.txt \
        --no-transcribe --out examples/demo_output.csv
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clipfinder import SAMPLE_RATE  # noqa: E402

DURATION_S = 900
BURSTS = [60, 115, 170, 225, 280, 335, 390, 445, 500, 555, 610, 665, 720, 775, 830, 870]
MARKS = {60: "clutch 1v3", 225: "boss fight start", 410: "ace round", 660: "near-death comeback", 830: "victory"}


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "demo"
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(42)
    samples = rng.normal(0, 0.02, DURATION_S * SAMPLE_RATE).astype(np.float32)
    for b in BURSTS:
        samples[b * SAMPLE_RATE:(b + 3) * SAMPLE_RATE] *= 25
    samples = np.clip(samples, -1, 1)

    import wave
    with wave.open(str(out_dir / "session.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((samples * 32767).astype(np.int16).tobytes())

    lines = [f"{t // 60}:{t % 60:02d} {label}" for t, label in sorted(MARKS.items())]
    (out_dir / "marks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'session.wav'} and {out_dir / 'marks.txt'}")


if __name__ == "__main__":
    main()
