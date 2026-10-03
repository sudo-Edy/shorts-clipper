import csv
import json
import shutil
import wave

import numpy as np
import pytest

from clipfinder import (
    EXPORT_MIN, MAX_CLIP_S, MIN_CLIP_S, SAMPLE_RATE, Candidate, Mark, Segment, build_candidates,
    caption_for, energy_curve, export_rough_cuts, fill_to_minimum, find_peaks, fmt, hype_hits, main,
    parse_notes, rank, safe_output, snap,
)
from pathlib import Path


def test_parse_notes_formats():
    marks = parse_notes("# comment\n12:34 clutch 1v3\n1:02:03 - boss down\n\n754s lag spike\n90\nnot a time\n")
    assert [(m.t, m.label) for m in marks] == [
        (90.0, ""), (754.0, "clutch 1v3"), (754.0, "lag spike"), (3723.0, "boss down"),
    ]


def test_fmt():
    assert fmt(3723.4) == "1:02:03"
    assert fmt(-2) == "0:00:00"


def synthetic(duration=300, bursts=(60, 150, 240)):
    rng = np.random.default_rng(0)
    s = rng.normal(0, 0.02, duration * SAMPLE_RATE).astype(np.float32)
    for b in bursts:
        s[b * SAMPLE_RATE:(b + 3) * SAMPLE_RATE] *= 20
    return s


def test_peaks_find_loud_bursts():
    peaks = find_peaks(energy_curve(synthetic()))
    times = sorted(round(t) for t, _ in peaks)
    assert len(times) == 3
    for got, want in zip(times, (61, 151, 241)):
        assert abs(got - want) <= 2


def test_flat_audio_has_no_peaks():
    assert find_peaks(energy_curve(np.full(60 * SAMPLE_RATE, 0.1, dtype=np.float32))) == []


def test_hype_hits_whole_words_only():
    assert set(hype_hits("OH MY GOD, no way! Let's go")) >= {"oh my god", "no way", "let's go"}
    assert hype_hits("whatever, yesterday") == []


def test_snap_to_sentence_edges_and_clamp():
    segs = [Segment(48, 52, "setup"), Segment(76, 81, "payoff")]
    start, end = snap(50, 80, segs, 300)
    assert (start, end) == (48, 81)
    assert snap(0, 200, [], 300)[1] == MAX_CLIP_S
    assert snap(0, 5, [], 300) == (0, MIN_CLIP_S)


def test_marks_outrank_unmarked_peaks_and_clips_stay_in_bounds():
    peaks = [(60.0, 5.0), (150.0, 4.0)]
    cands = build_candidates(300, peaks, [Mark(152, "clutch")], [])
    ranked = rank(cands)
    assert ranked[0].start <= 152 <= ranked[0].end
    assert "you marked it (clutch)" in ranked[0].reasons
    assert all(MIN_CLIP_S <= c.duration <= MAX_CLIP_S for c in ranked)


def test_rank_drops_overlaps():
    a, b = Candidate(0, 30, 10), Candidate(5, 35, 9)
    c = Candidate(100, 130, 5)
    assert rank([a, b, c]) == [a, c]


def test_fill_to_minimum_reaches_15_on_long_quiet_audio():
    db = energy_curve(synthetic(duration=900, bursts=(100,)))
    kept = rank(build_candidates(900, find_peaks(db), [], []))
    full = fill_to_minimum(kept, db, 900, [])
    assert 15 <= len(full) <= 20
    for i, x in enumerate(full):
        for y in full[i + 1:]:
            inter = max(0, min(x.end, y.end) - max(x.start, y.start))
            assert inter <= 0.3 * min(x.duration, y.duration)


def test_output_never_overwrites_media(tmp_path):
    video = tmp_path / "rec.mp4"
    video.write_bytes(b"x")
    with pytest.raises(SystemExit):
        safe_output(video, video)
    with pytest.raises(SystemExit):
        safe_output(tmp_path / "out.mkv", video)
    assert safe_output(tmp_path / "out.csv", video) == tmp_path / "out.csv"


def test_phrases_split_at_pauses():
    from clipfinder import phrases
    words = [(20.0, 20.4, " Okay"), (20.5, 21.0, " looting."), (96.0, 96.4, " No"), (96.5, 96.9, " way!")]
    segs = phrases(words)
    assert [(s.start, s.end, s.text) for s in segs] == [(20.0, 21.0, "Okay looting."), (96.0, 96.9, "No way!")]


def write_wav(path: Path, samples: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_end_to_end_cli_writes_15_to_20_clip_csv(tmp_path):
    """Run the real CLI (ffmpeg decode -> candidates -> CSV) on a synthetic recording.

    This is the done-when check itself, not just a claim about it: a 10-minute
    recording with marks + loud bursts must come out the other end as an actual
    CSV of 15-20 ranked clips, each 15-60s, each with a one-line reason.
    """
    duration = 600
    bursts = [30, 70, 110, 150, 190, 230, 270, 310, 350, 390, 430, 470, 510, 550, 580]
    samples = synthetic(duration=duration, bursts=bursts)
    recording = tmp_path / "rec.wav"
    write_wav(recording, samples)
    notes = tmp_path / "marks.txt"
    notes.write_text("0:30 clutch 1v3\n9:10 victory\n")
    out = tmp_path / "rec_shorts.csv"

    rc = main([str(recording), "--notes", str(notes), "--no-transcribe", "--out", str(out)])

    assert rc == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert 15 <= len(rows) <= 20
    for row in rows:
        assert MIN_CLIP_S <= int(row["duration_s"]) <= MAX_CLIP_S
        assert row["why"]
    assert any("clutch" in r["why"] for r in rows)
    assert any("victory" in r["why"] for r in rows)
    assert out.with_suffix(".edl").exists()


def test_timecode_and_edl(tmp_path):
    from clipfinder import timecode, write_edl
    assert timecode(0, 60) == "01:00:00:00"
    assert timecode(95.5, 60) == "01:01:35:30"
    assert timecode(3723, 30, start_hour=0) == "01:02:03:00"
    edl = tmp_path / "m.edl"
    cands = [Candidate(95.5, 125.5, 90, ["you marked it (clutch | 1v3)"]), Candidate(10, 40, 5, [])]
    write_edl(edl, cands, 60)
    text = edl.read_text()
    assert text.startswith("TITLE: Shorts candidates\nFCM: NON-DROP FRAME\n")
    assert "001  001      V     C        01:01:35:30 01:01:35:31 01:01:35:30 01:01:35:31" in text
    assert " |C:ResolveColorRed |M:#1 you marked it (clutch / 1v3) |D:1800" in text
    assert "|M:#2 loud stretch |D:1800" in text


def test_caption_for_is_grounded_not_invented():
    assert caption_for(Candidate(0, 20, 50, ["you marked it (clutch 1v3)"])) == "clutch 1v3"
    assert caption_for(Candidate(0, 20, 20, ['reaction: "no way", "clutch"'])) == 'no way, clutch'
    assert caption_for(Candidate(0, 20, 30, ["big audio spike"])) == "big audio spike"
    assert caption_for(Candidate(0, 20, 0, [])) == "highlight moment"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_export_rough_cuts_writes_real_clip_files_and_manifest(tmp_path):
    """This is the done-when check for the demo-pack feature: real clip files you
    can play, plus a manifest with each clip's source timestamp, what happens,
    and a suggested caption - cut from an actual (synthetic) recording, not claimed."""
    duration = 600
    bursts = [30, 70, 110, 150, 190, 230, 270, 310, 350, 390, 430, 470, 510, 550, 580]
    recording = tmp_path / "rec.wav"
    write_wav(recording, synthetic(duration=duration, bursts=bursts))
    notes = tmp_path / "marks.txt"
    notes.write_text("0:30 clutch 1v3\n9:10 victory\n")
    export_dir = tmp_path / "shorts-demo" / "rough-cuts"

    rc = main([str(recording), "--notes", str(notes), "--no-transcribe",
               "--out", str(tmp_path / "rec_shorts.csv"), "--export-dir", str(export_dir)])

    assert rc == 0
    clips = sorted(export_dir.glob("clip_*.wav"))
    assert EXPORT_MIN <= len(clips) <= 8
    for clip in clips:
        assert clip.stat().st_size > 0

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == len(clips)
    for entry in manifest:
        assert entry["file"] in {c.name for c in clips}
        assert entry["source_start"] and entry["source_end"]
        assert MIN_CLIP_S <= entry["duration_s"] <= MAX_CLIP_S
        assert entry["moment"]
        assert entry["suggested_caption"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_export_rough_cuts_refuses_to_overwrite(tmp_path):
    cands = [Candidate(0, 20, 50, ["you marked it (clutch)"])]
    recording = tmp_path / "rec.wav"
    write_wav(recording, synthetic(duration=30, bursts=(5,)))
    export_dir = tmp_path / "out"
    export_dir.mkdir()
    (export_dir / "clip_01.wav").write_bytes(b"already here")

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        export_rough_cuts(recording, cands, export_dir, count=1)
    assert (export_dir / "clip_01.wav").read_bytes() == b"already here"
