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


Word = tuple[float, float, str]


def transcribe(audio: Path, model_name: str) -> list[Segment]:
    return phrases(transcribe_words(audio, model_name))


def transcribe_words(audio: Path, model_name: str) -> list[Word]:
    use_system_certificates()
    add_nvidia_dll_dirs()
    from faster_whisper import WhisperModel

    def run(**kw) -> list[Word]:
        # use the cached model when there is one; download only when it's missing
        try:
            model = WhisperModel(model_name, local_files_only=True, **kw)
        except Exception:
            model = WhisperModel(model_name, **kw)
        segments, _ = model.transcribe(str(audio), vad_filter=True, word_timestamps=True)
        # segments is lazy: missing CUDA DLLs only fail here, so consume it inside the try
        return [(w.start, w.end, w.word) for s in segments for w in (s.words or [])]

    try:
        return run(device="cuda", compute_type="float16")
    except Exception as e:
        print(f"GPU transcription failed ({e}); using the CPU (slower). "
              "To use the GPU: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12", file=sys.stderr)
        return run(device="cpu", compute_type="int8")


def phrases(words: list[tuple[float, float, str]], max_gap_s: float = 1.0,
            max_len_s: float = 10.0) -> list[Segment]:
    """Group timed words into phrases split at pauses (and every max_len_s).

    Whisper's own segments can stretch across a minute of silence, which would
    smear a reaction ("no way!") onto the wrong clip; word timings don't. Nonstop
    multiplayer chatter has no pauses at all, so phrases are also capped in length.
    """
    out: list[Segment] = []
    for start, end, word in words:
        if out and start - out[-1].end <= max_gap_s and end - out[-1].start <= max_len_s:
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


def timecode(seconds: float, fps: int, start_hour: int = 1) -> str:
    frames = int(round(seconds * fps)) + start_hour * 3600 * fps
    f = frames % fps
    total = frames // fps
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}:{f:02d}"


def write_edl(path: Path, cands: list[Candidate], fps: int, start_hour: int = 1) -> None:
    """Timeline markers for DaVinci Resolve (Timeline > Import > Timeline Markers from EDL).

    One duration marker per clip: red = top 5, yellow = 6-10, blue = the rest.
    Times assume the recording starts at the timeline's start (Resolve default 01:00:00:00).
    """
    lines = ["TITLE: Shorts candidates", "FCM: NON-DROP FRAME", ""]
    for i, c in enumerate(cands, 1):
        color = "Red" if i <= 5 else "Yellow" if i <= 10 else "Blue"
        tc_in, tc_out = timecode(c.start, fps, start_hour), timecode(c.start + 1 / fps, fps, start_hour)
        name = f"#{i} {'; '.join(c.reasons) or 'loud stretch'}".replace("|", "/")[:80]
        lines.append(f"{i:03d}  001      V     C        {tc_in} {tc_out} {tc_in} {tc_out}  ")
        lines.append(f" |C:ResolveColor{color} |M:{name} |D:{max(1, int(round(c.duration * fps)))}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------- rendering vertical shorts ----------

CAPTION_MAX_WORDS = 3
CAPTION_MAX_S = 1.2


def caption_chunks(words: list[Word], start: float, end: float,
                   max_words: int = CAPTION_MAX_WORDS, max_s: float = CAPTION_MAX_S) -> list[tuple[float, float, str]]:
    """Short punchy caption lines, timed relative to the clip start.

    A chunk closes at max_words, at max_s, after punctuation, at a pause >= 0.3 s,
    or at 2 words once it has lasted 0.35 s (stops fast talk flashing); each one
    is held until the next starts (if that is soon) so captions don't flicker.
    """
    inside = [(max(s, start) - start, min(e, end) - start, w.strip()) for s, e, w in words
              if e > start and s < end and w.strip()]
    chunks: list[list] = []
    closed = True
    for s, e, w in inside:
        cur = chunks[-1] if chunks else None
        if (cur and not closed and len(cur[2]) < max_words and e - cur[0] <= max_s and s - cur[1] < 0.3
                and not (len(cur[2]) >= 2 and cur[1] - cur[0] >= 0.35)):
            cur[1] = e
            cur[2].append(w)
        else:
            chunks.append([s, e, [w]])
        closed = w[-1] in ".,!?;:"
    out = []
    for i, (s, e, ws) in enumerate(chunks):
        nxt = chunks[i + 1][0] if i + 1 < len(chunks) else None
        if nxt is not None and nxt - e < 0.5:
            e = nxt
        text = " ".join(ws).rstrip(",;:").upper()
        e = max(e, s + 0.25)
        if nxt is not None:
            e = min(e, nxt)  # never overlap the next caption
        out.append((round(s, 3), round(e, 3), text))
    return out


def _ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def hook_lines(text: str, max_chars: int = 18, max_lines: int = 3) -> str:
    """Wrap a hook title into at most 3 short lines (ASS \\N breaks), cutting with an ellipsis if longer."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;:") + "..."
    return "\\N".join(l.replace("{", "(").replace("}", ")") for l in lines)


def ass_captions(chunks: list[tuple[float, float, str]], width: int = 1080, height: int = 1920,
                 hook: str | None = None, hook_end: float | None = None, accent: str | None = None,
                 style: str = "bold") -> str:
    """Big bold outlined captions in the lower third, the standard Shorts look.
    style="minimal": smaller regular-weight captions, thin outline, lower (documentary look)."""
    cap = ("Style: Cap,Arial Black,112,&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,3,2,80,80,"
           f"{int(height * 0.28)},1\n") if style != "minimal" else (
           "Style: Cap,Arial,70,&H00FFFFFF,&H0000FFFF,&H00000000,&H50000000,0,0,0,0,100,100,0,0,1,3,1,2,90,90,"
           f"{int(height * 0.24)},1\n")
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding\n"
        + cap +
        # Hook title: top of frame, below the platform UI (~15% down), white on a dark box
        "Style: Hook,Arial Black,74,&H00FFFFFF,&H0000FFFF,&H00101010,&HA0000000,-1,0,0,0,100,100,0,0,3,14,0,8,90,90,"
        f"{int(height * 0.15)},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Cap,,0,0,0,,{{\\fscx85\\fscy85\\t(0,80,\\fscx100\\fscy100)}}"
             f"{t.replace('{', '(').replace('}', ')')}" for s, e, t in chunks]
    if hook and hook.strip():
        # optional per-clip accent (RRGGBB; ASS wants BGR) so a batch of Shorts doesn't look templated
        tint = (f"\\c&H{(accent[4:6] + accent[2:4] + accent[0:2]).upper()}&"
                if accent and re.fullmatch(r"[0-9A-Fa-f]{6}", accent) else "")
        lines.insert(0, f"Dialogue: 1,{_ass_time(0)},{_ass_time(hook_end if hook_end else 3600)},Hook,,0,0,0,,"
                        f"{{\\fad(120,0){tint}}}{hook_lines(hook.strip().upper())}")
    return head + "\n".join(lines) + "\n"


def vertical_filter(layout: str = "crop", crop_x: float = 0.5, captions: str | None = None) -> str:
    """ffmpeg -filter_complex graph producing a 1080x1920 [v] stream.

    crop: fill the frame with a 9:16 slice (crop_x 0=left edge, 0.5=centre, 1=right).
    fit:  the whole landscape frame, centred over a blurred, zoomed copy of itself.
    """
    if layout == "fit":
        g = ("[0:v]split=2[bg][fg];"
             "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:2[bgb];"
             "[fg]scale=1080:-2[fgs];[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1")
    else:
        g = f"[0:v]crop='min(iw,ih*9/16)':ih:'(iw-min(iw,ih*9/16))*{min(1.0, max(0.0, crop_x))}':0,scale=1080:1920,setsar=1"
    if captions:
        g += f",subtitles={captions}"
    return g + "[v]"


_ENCODER_CACHE: dict[str, bool] = {}


def has_nvenc() -> bool:
    if "nvenc" not in _ENCODER_CACHE:
        try:
            r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
            _ENCODER_CACHE["nvenc"] = "h264_nvenc" in r.stdout
        except Exception:
            _ENCODER_CACHE["nvenc"] = False
    return _ENCODER_CACHE["nvenc"]


def render_cmd(media: Path, start: float, end: float, out: Path, layout: str = "crop", crop_x: float = 0.5,
               captions: str | None = None, audio_track: int = 0, gpu: bool = True) -> list[str]:
    venc = (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0"] if gpu
            else ["-c:v", "libx264", "-preset", "medium", "-crf", "20"])
    return (["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(media),
             "-filter_complex", vertical_filter(layout, crop_x, captions), "-map", "[v]",
             "-map", f"0:a:{audio_track}?", "-r", "30"] + venc +
            ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)])


def render_shorts(media: Path, cands: list[Candidate], words: list[Word], out_dir: Path, count: int,
                  layout: str = "crop", crop_x: float = 0.5, captions: bool = True, audio_track: int = 0,
                  labels: list[int] | None = None, hooks: list[str] | None = None,
                  names: list[str] | None = None, accents: list[str] | None = None,
                  caption_style: str = "bold") -> list[Path]:
    """Render the top `count` candidates as 1080x1920 mp4s. Never touches the source.
    labels: the rank number to put in each file name (default 1, 2, 3...)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    gpu = has_nvenc()
    for i, c in enumerate(cands[:count], 1):
        n = labels[i - 1] if labels and i - 1 < len(labels) else i
        name = names[i - 1] if names and i - 1 < len(names) and names[i - 1] else f"{n:02d}_{fmt(c.start).replace(':', '-')}"
        out = out_dir / (re.sub(r"[^\w\-]+", "_", name)[:60] + ".mp4")
        hook = hooks[i - 1] if hooks and i - 1 < len(hooks) else None
        accent = accents[i - 1] if accents and i - 1 < len(accents) else None
        if out.resolve() == media.resolve():
            raise SystemExit("Refusing to write over the source recording.")
        with tempfile.TemporaryDirectory(prefix="clipfinder-cap-") as tmp:
            ass = None
            chunks = caption_chunks(words, c.start, c.end) if captions else []
            if chunks or hook:
                (Path(tmp) / "caps.ass").write_text(ass_captions(chunks, hook=hook, accent=accent, style=caption_style), encoding="utf-8")
                ass = "caps.ass"  # relative: ffmpeg runs in tmp, which dodges Windows path escaping in filters
            for use_gpu in ([True, False] if gpu else [False]):
                cmd = render_cmd(media, c.start, c.end, out.resolve(), layout, crop_x, ass, audio_track, use_gpu)
                r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
                if r.returncode == 0:
                    break
            else:
                raise SystemExit(f"ffmpeg failed on clip {i}: {r.stderr.strip()[-400:]}")
        made.append(out)
        print(f"  rendered {out.name}", file=sys.stderr)
    return made


def has_video(media: Path) -> bool:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type",
                        "-of", "csv=p=0", str(media)], capture_output=True, text=True)
    return "video" in r.stdout


def probe_fps(media: Path) -> int:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                            "-of", "csv=p=0", str(media)], capture_output=True, text=True, check=True)
        num, _, den = r.stdout.strip().partition("/")
        return max(1, round(float(num) / float(den or 1)))
    except Exception:
        return 30  # audio-only or unreadable: any rate works for markers


def safe_output(out: Path, media: Path) -> Path:
    if out.suffix.lower() != ".csv":
        raise SystemExit(f"Output must be a .csv file, got {out}")
    if out.resolve() == media.resolve() or out.suffix.lower() in MEDIA_EXTS:
        raise SystemExit("Refusing to write over a media file.")
    return out


def load_words(path: Path) -> list[Word]:
    import json
    return [(float(a), float(b), str(w)) for a, b, w in json.loads(path.read_text(encoding="utf-8"))]


def save_words(path: Path, words: list[Word]) -> None:
    import json
    if path.suffix.lower() in MEDIA_EXTS:
        raise SystemExit("Refusing to write over a media file.")
    path.write_text(json.dumps([[round(a, 3), round(b, 3), w] for a, b, w in words], ensure_ascii=False), encoding="utf-8")


def run_explicit(args) -> int:
    """Tool mode for CreatorOS: transcribe once (--transcribe-only --words-out), then render an exact
    clip list chosen elsewhere (--clips, reusing --words-in). Read-only on the source, like the rest."""
    import json
    if not args.recording.is_file():
        raise SystemExit(f"Not found: {args.recording}")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is not on PATH.")
    if args.words_in:
        words = load_words(args.words_in)
    elif args.no_transcribe:
        words = []
    else:
        with tempfile.TemporaryDirectory(prefix="clipfinder-") as tmp:
            audio = extract_audio(args.recording, Path(tmp))
            print(f"Transcribing with {args.model}...", file=sys.stderr)
            words = transcribe_words(audio, args.model)
    if args.words_out:
        save_words(args.words_out, words)
        print(f"{len(words)} words -> {args.words_out}", file=sys.stderr)
    if not args.clips:
        return 0
    spec = json.loads(args.clips.read_text(encoding="utf-8"))
    cands = [Candidate(float(c["start"]), float(c["end"])) for c in spec]
    if any(c.end <= c.start for c in cands):
        raise SystemExit("Every clip needs end > start.")
    out_dir = args.clips_dir or Path(f"{args.recording.stem}_shorts_clips")
    made = render_shorts(args.recording, cands, words, out_dir, len(cands), args.layout, args.crop_x,
                         not args.no_captions, args.audio_track,
                         hooks=[c.get("hook") or "" for c in spec], names=[c.get("name") or "" for c in spec],
                         accents=[c.get("accent") or "" for c in spec], caption_style=args.caption_style)
    print(json.dumps({"rendered": [str(m) for m in made]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", type=Path)
    p.add_argument("--notes", type=Path, help="timestamp notes file (one mark per line)")
    p.add_argument("--out", type=Path, help="CSV path (default: <recording name>_shorts.csv in the current folder)")
    p.add_argument("--count", type=int, default=20, help="max clips (default 20)")
    p.add_argument("--model", default="large-v3", help="faster-whisper model (large-v3, medium, small...)")
    p.add_argument("--no-transcribe", action="store_true", help="skip speech; use marks + audio energy only")
    p.add_argument("--fps", type=int, help="timeline frame rate for the EDL markers (default: the video's)")
    p.add_argument("--timeline-start-hour", type=int, default=1, help="timeline start hour for the EDL (Resolve default 1)")
    p.add_argument("--render", type=int, metavar="N", default=0,
                   help="also render the top N as vertical 1080x1920 mp4s (into <csv name>_clips/)")
    p.add_argument("--layout", choices=["crop", "fit"], default="crop",
                   help="crop = fill with a 9:16 slice; fit = whole frame over a blurred background")
    p.add_argument("--crop-x", type=float, default=0.5, help="where to slice for --layout crop: 0 left, 0.5 centre, 1 right")
    p.add_argument("--no-captions", action="store_true", help="render without burned-in captions")
    p.add_argument("--caption-style", choices=["bold", "minimal"], default="bold",
                   help="bold = big Shorts captions (default); minimal = smaller documentary-style captions")
    p.add_argument("--audio-track", type=int, default=0, help="which audio stream goes into the rendered shorts")
    p.add_argument("--words-out", type=Path, help="write the word-timed transcript as JSON (for other tools)")
    p.add_argument("--words-in", type=Path, help="reuse a word-timed transcript JSON instead of transcribing again")
    p.add_argument("--transcribe-only", action="store_true", help="only transcribe (use with --words-out); no ranking")
    p.add_argument("--clips", type=Path, help="JSON list of clips to render exactly: "
                   "[{\"start\": s, \"end\": s, \"hook\": \"top text\", \"name\": \"file name\"}] (into --clips-dir)")
    p.add_argument("--clips-dir", type=Path, help="output folder for --clips (default: <recording name>_shorts_clips)")
    args = p.parse_args(argv)
    if args.clips or args.transcribe_only:
        return run_explicit(args)

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
        words: list[Word] = []
        if not args.no_transcribe:
            print(f"Transcribing with {args.model}...", file=sys.stderr)
            words = transcribe_words(audio, args.model)
            segments = phrases(words)

    if duration < MIN_CLIP_S:
        raise SystemExit("Recording is shorter than 15 seconds.")
    kept = rank(build_candidates(duration, peaks, marks, segments), limit=args.count)
    kept = fill_to_minimum(kept, db, duration, segments, minimum=min(15, args.count), limit=args.count)
    write_csv(out, kept)
    edl = out.with_suffix(".edl")
    write_edl(edl, kept, args.fps or probe_fps(args.recording), args.timeline_start_hour)
    print(f"{len(kept)} clips -> {out} (+ Resolve markers: {edl.name})", file=sys.stderr)
    for i, c in enumerate(kept[:5], 1):
        print(f"  {i}. {fmt(c.start)}-{fmt(c.end)}  {'; '.join(c.reasons)}", file=sys.stderr)
    if args.render > 0:
        if not has_video(args.recording):
            print("No video stream; skipping --render.", file=sys.stderr)
        else:
            clips = out.parent / f"{out.stem}_clips"
            print(f"Rendering top {min(args.render, len(kept))} as vertical shorts -> {clips}", file=sys.stderr)
            render_shorts(args.recording, kept, words, clips, args.render, args.layout, args.crop_x,
                          not args.no_captions, args.audio_track)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
