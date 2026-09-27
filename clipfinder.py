#!/usr/bin/env python3
"""Shorts clip-finder: rank the best 15-60s moments in a gameplay recording.

Usage:
    python clipfinder.py recording.mp4 --notes marks.txt
    python clipfinder.py recording.mp4 --no-transcribe        # audio energy + marks only

Signals, strongest first:
  1. your marked timestamps (notes file, one per line: "12:34 clutch 1v3")
  2. audio-energy spikes (loud moments vs. the recording's own baseline)
  3. hype words in the transcript (faster-whisper, GPU if available)

Read-only on your media: the audio is decoded by ffmpeg into a temp folder that
is deleted afterwards. The only file written is the CSV report.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
WINDOW_S = 0.5            # energy resolution
MIN_CLIP_S = 15.0
MAX_CLIP_S = 60.0
TARGET_CLIP_S = 30.0
LEAD_IN_S = 10.0          # context before the peak moment (the setup)
MARK_RADIUS_S = 20.0      # a candidate within this distance of a mark gets the mark bonus
MIN_RISE_DB = 6.0         # a spike must be at least this much louder than the typical level
SNAP_TOLERANCE_S = 4.0   # move clip edges to sentence boundaries within this distance

HYPE_WORDS = [
    "let's go", "lets go", "oh my god", "omg", "no way", "what", "clutch", "insane",
    "yes", "bro", "holy", "crazy", "wow", "oh no", "run", "get him", "one shot",
    "ace", "gg", "wtf", "noo", "yeah",
]
MEDIA_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts", ".m4v", ".wav", ".mp3", ".m4a", ".aac", ".flac"}


@dataclass
class Mark:
    t: float
    label: str


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Candidate:
    start: float
    end: float
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


# ---------- parsing ----------

TIME_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d+))?\s*[-–—:,]?\s*(.*)$")


def parse_timestamp(s: str) -> float | None:
    m = TIME_RE.match(s)
    if not m:
        return None
    h, mnt, sec, frac, _ = m.groups()
    t = int(h or 0) * 3600 + int(mnt) * 60 + int(sec)
    if frac:
        t += float("0." + frac)
    return float(t)


def parse_notes(text: str) -> list[Mark]:
    """One mark per line: "12:34 label", "1:02:03 - label", "754s label" or "754 label"."""
    marks = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = TIME_RE.match(line)
        if m:
            marks.append(Mark(parse_timestamp(line), m.group(5).strip()))
            continue
        m = re.match(r"^(\d+(?:\.\d+)?)s?\b\s*[-–—:,]?\s*(.*)$", line)
        if m:
            marks.append(Mark(float(m.group(1)), m.group(2).strip()))
    return sorted(marks, key=lambda k: k.t)


def fmt(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(int(round(t)), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


# ---------- audio ----------

def extract_audio(media: Path, tmpdir: Path) -> Path:
    out = tmpdir / "audio.wav"
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(media), "-vn", "-ac", "1",
           "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out)]
    subprocess.run(cmd, check=True)
    return out


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def energy_curve(samples: np.ndarray, sr: int = SAMPLE_RATE, window_s: float = WINDOW_S) -> np.ndarray:
    """Loudness in dB per window."""
    n = int(sr * window_s)
    usable = len(samples) // n * n
    if usable == 0:
        return np.zeros(0)
    frames = samples[:usable].reshape(-1, n)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return 20 * np.log10(rms + 1e-6)


def find_peaks(db: np.ndarray, window_s: float = WINDOW_S, min_gap_s: float = 20.0, z_min: float = 1.5) -> list[tuple[float, float]]:
    """(time, z-score) of loud moments relative to the recording's own baseline.

    Robust z (median/MAD) so a loud game doesn't hide its own spikes; peaks closer
    than min_gap_s keep only the loudest.
    """
    if len(db) == 0:
        return []
    # smooth over ~1.5s so a single click doesn't count
    k = max(1, int(1.5 / window_s))
    padded = np.pad(db, (k // 2, k - 1 - k // 2), mode="edge")
    smooth = np.convolve(padded, np.ones(k) / k, mode="valid")
    med = np.median(smooth)
    # floor the spread at 2 dB: steady audio would otherwise turn tiny wobbles into "spikes"
    mad = max(np.median(np.abs(smooth - med)) * 1.4826, 2.0)
    z = (smooth - med) / mad
    order = np.argsort(-z)
    picked: list[tuple[float, float]] = []
    for i in order:
        if z[i] < z_min:
            break
        if z_min > 0 and smooth[i] - med < MIN_RISE_DB:
            break
        t = i * window_s + window_s / 2
        if all(abs(t - p) >= min_gap_s for p, _ in picked):
            picked.append((t, float(z[i])))
    return picked


# ---------- transcript ----------

def use_system_certificates() -> None:
    """Trust the OS certificate store for the model download.

    Antivirus/proxies that inspect HTTPS make Python's bundled CA list fail with
    CERTIFICATE_VERIFY_FAILED while browsers work; truststore fixes that.
    """
    try:
        import truststore
    except ImportError:
        try:
            from pip._vendor import truststore
        except ImportError:
            return
    truststore.inject_into_ssl()


def add_nvidia_dll_dirs() -> None:
    """Let CTranslate2 find cuBLAS/cuDNN installed with pip (nvidia-cublas-cu12, nvidia-cudnn-cu12)."""
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    for root in getattr(nvidia, "__path__", []):
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def transcribe(audio: Path, model_name: str) -> list[Segment]:
    use_system_certificates()
    add_nvidia_dll_dirs()
    from faster_whisper import WhisperModel

    def run(**kw) -> list[Segment]:
        # use the cached model when there is one; download only when it's missing
        try:
            model = WhisperModel(model_name, local_files_only=True, **kw)
        except Exception:
            model = WhisperModel(model_name, **kw)
        segments, _ = model.transcribe(str(audio), vad_filter=True, word_timestamps=True)
        # segments is lazy: missing CUDA DLLs only fail here, so consume it inside the try
        words = [(w.start, w.end, w.word) for s in segments for w in (s.words or [])]
        return phrases(words)

    try:
        return run(device="cuda", compute_type="float16")
    except Exception as e:
        print(f"GPU transcription failed ({e}); using the CPU (slower). "
              "To use the GPU: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12", file=sys.stderr)
        return run(device="cpu", compute_type="int8")


def phrases(words: list[tuple[float, float, str]], max_gap_s: float = 1.0) -> list[Segment]:
    """Group timed words into phrases split at pauses.

    Whisper's own segments can stretch across a minute of silence, which would
    smear a reaction ("no way!") onto the wrong clip; word timings don't.
    """
    out: list[Segment] = []
    for start, end, word in words:
        if out and start - out[-1].end <= max_gap_s:
            out[-1].end = end
            out[-1].text += word
        else:
            out.append(Segment(start, end, word))
    for seg in out:
        seg.text = seg.text.strip()
    return out


def hype_hits(text: str) -> list[str]:
    low = " " + re.sub(r"[^a-z' ]", " ", text.lower()) + " "
    return [w for w in HYPE_WORDS if f" {w} " in low]


# ---------- candidates ----------

def window_around(t: float, duration: float) -> tuple[float, float]:
    start = max(0.0, t - LEAD_IN_S)
    end = start + TARGET_CLIP_S
    if end > duration:
        end = duration
        start = max(0.0, end - TARGET_CLIP_S)
    return start, end


def snap(start: float, end: float, segments: list[Segment], duration: float) -> tuple[float, float]:
    """Move edges to sentence boundaries nearby so clips don't cut mid-word."""
    for s in segments:
        if s.start < start < s.end and start - s.start <= SNAP_TOLERANCE_S:
            start = s.start
        if s.start < end < s.end and s.end - end <= SNAP_TOLERANCE_S:
            end = s.end
    end = min(end, duration)
    if end - start > MAX_CLIP_S:
        end = start + MAX_CLIP_S
    if end - start < MIN_CLIP_S:
        end = min(duration, start + MIN_CLIP_S)
        start = max(0.0, end - MIN_CLIP_S)
    return start, end


def text_in(start: float, end: float, segments: list[Segment]) -> str:
    return " ".join(s.text for s in segments if s.end > start and s.start < end)


def overlap(a: Candidate, b: Candidate) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return inter / min(a.duration, b.duration) if inter else 0.0


def build_candidates(duration: float, peaks: list[tuple[float, float]], marks: list[Mark],
                     segments: list[Segment]) -> list[Candidate]:
    anchors: list[tuple[float, str]] = [(m.t, "mark") for m in marks if 0 <= m.t <= duration]
    anchors += [(t, "peak") for t, _ in peaks]
    for s in segments:
        if hype_hits(s.text):
            anchors.append(((s.start + s.end) / 2, "hype"))

    zmax = max((z for _, z in peaks), default=1.0)
    out: list[Candidate] = []
    for t, _ in anchors:
        start, end = snap(*window_around(t, duration), segments, duration)
        c = Candidate(start, end)
        near = [m for m in marks if abs(m.t - t) <= MARK_RADIUS_S or start <= m.t <= end]
        if near:
            c.score += 50
            label = near[0].label or "marked"
            c.reasons.append(f"you marked it ({label})")
        loud = [z for pt, z in peaks if start <= pt <= end]
        if loud:
            c.score += 30 * min(1.0, max(loud) / zmax)
            c.reasons.append("big audio spike" if max(loud) >= 0.7 * zmax else "loud moment")
        c.text = text_in(start, end, segments)
        words = sorted(set(hype_hits(c.text)))
        if words:
            c.score += min(20, 7 * len(words))
            c.reasons.append("reaction: " + ", ".join(f'"{w}"' for w in words[:3]))
        if c.text:
            wps = len(c.text.split()) / max(c.duration, 1)
            if wps >= 1.5:
                c.score += 5
                c.reasons.append("constant commentary")
        out.append(c)
    return out


def rank(cands: list[Candidate], limit: int = 20, max_overlap: float = 0.3) -> list[Candidate]:
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda c: -c.score):
        if c.score <= 0 or not (MIN_CLIP_S <= round(c.duration, 3) <= MAX_CLIP_S):
            continue
        if all(overlap(c, k) <= max_overlap for k in kept):
            kept.append(c)
        if len(kept) == limit:
            break
    return kept


def fill_to_minimum(kept: list[Candidate], db: np.ndarray, duration: float, segments: list[Segment],
                    minimum: int = 15, limit: int = 20) -> list[Candidate]:
    """Quiet recording? Top up with the next-loudest non-overlapping stretches."""
    if len(kept) >= minimum:
        return kept
    extra = build_candidates(duration, find_peaks(db, min_gap_s=MIN_CLIP_S, z_min=-1e9), [], segments)
    for c in extra:
        c.reasons = [r.replace("big audio spike", "louder than usual").replace("loud moment", "louder than usual") for r in c.reasons]
    for c in rank(extra, limit=limit * 3):
        if len(kept) >= limit:
            break
        if all(overlap(c, k) <= 0.3 for k in kept):
            kept.append(c)
    return kept


def write_csv(path: Path, cands: list[Candidate]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "start", "end", "duration_s", "score", "why", "transcript"])
        for i, c in enumerate(cands, 1):
            why = "; ".join(c.reasons) or "strongest remaining moment"
            w.writerow([i, fmt(c.start), fmt(c.end), round(c.duration), round(c.score), why, c.text[:200]])


def safe_output(out: Path, media: Path) -> Path:
    if out.suffix.lower() != ".csv":
        raise SystemExit(f"Output must be a .csv file, got {out}")
    if out.resolve() == media.resolve() or out.suffix.lower() in MEDIA_EXTS:
        raise SystemExit("Refusing to write over a media file.")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", type=Path)
    p.add_argument("--notes", type=Path, help="timestamp notes file (one mark per line)")
    p.add_argument("--out", type=Path, help="CSV path (default: <recording name>_shorts.csv in the current folder)")
    p.add_argument("--count", type=int, default=20, help="max clips (default 20)")
    p.add_argument("--model", default="large-v3", help="faster-whisper model (large-v3, medium, small...)")
    p.add_argument("--no-transcribe", action="store_true", help="skip speech; use marks + audio energy only")
    args = p.parse_args(argv)

    if not args.recording.is_file():
        raise SystemExit(f"Not found: {args.recording}")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is not on PATH.")
    out = safe_output(args.out or Path(f"{args.recording.stem}_shorts.csv"), args.recording)
    marks = parse_notes(args.notes.read_text(encoding="utf-8")) if args.notes else []

    with tempfile.TemporaryDirectory(prefix="clipfinder-") as tmp:
        audio = extract_audio(args.recording, Path(tmp))
        samples = read_wav(audio)
        duration = len(samples) / SAMPLE_RATE
        print(f"Audio: {fmt(duration)} long. Finding loud moments...", file=sys.stderr)
        db = energy_curve(samples)
        peaks = find_peaks(db)
        segments: list[Segment] = []
        if not args.no_transcribe:
            print(f"Transcribing with {args.model}...", file=sys.stderr)
            segments = transcribe(audio, args.model)

    if duration < MIN_CLIP_S:
        raise SystemExit("Recording is shorter than 15 seconds.")
    kept = rank(build_candidates(duration, peaks, marks, segments), limit=args.count)
    kept = fill_to_minimum(kept, db, duration, segments, minimum=min(15, args.count), limit=args.count)
    write_csv(out, kept)
    print(f"{len(kept)} clips -> {out}", file=sys.stderr)
    for i, c in enumerate(kept[:5], 1):
        print(f"  {i}. {fmt(c.start)}-{fmt(c.end)}  {'; '.join(c.reasons)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
