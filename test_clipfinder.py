import csv
import shutil
import wave

import numpy as np
import pytest

from clipfinder import (
    MAX_CLIP_S, MIN_CLIP_S, SAMPLE_RATE, Candidate, Mark, Segment, build_candidates, energy_curve,
    fill_to_minimum, find_peaks, fmt, hype_hits, main, parse_notes, rank, safe_output, snap,
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


def test_caption_chunks_short_relative_and_no_flicker():
    from clipfinder import caption_chunks
    words = [(10.0, 10.2, " no"), (10.2, 10.4, " way"), (10.4, 10.6, " bro,"), (10.6, 10.8, " look"),
             (12.0, 12.3, " run!"), (30.0, 30.5, " outside")]
    ch = caption_chunks(words, 10.0, 20.0)
    assert ch[0] == (0.0, 0.4, "NO WAY")                # closes at 2 words once it lasts 0.35 s
    assert ch[1][2] == "BRO"                            # trailing comma stripped, comma closes the cue
    assert ch[2][2] == "LOOK" and ch[3][2] == "RUN!"    # pause >= 0.3 s splits
    assert all(ch[i][1] <= ch[i + 1][0] for i in range(len(ch) - 1))  # no overlaps
    assert all(e > s for s, e, _ in ch) and all(e <= 10.0 + 1e-6 for _, e, _ in ch[:-1])
    assert "OUTSIDE" not in " ".join(t for *_, t in ch)  # outside the clip window


def test_ass_and_filters_are_well_formed():
    from clipfinder import ass_captions, vertical_filter, render_cmd
    ass = ass_captions([(0.0, 1.0, "HI {X}")])
    assert "PlayResX: 1080" in ass and "Dialogue: 0,0:00:00.00,0:00:01.00,Cap" in ass and "{X}" not in ass
    assert vertical_filter("crop", 0.0).endswith("[v]") and "*0.0" in vertical_filter("crop", 0.0)
    assert "boxblur" in vertical_filter("fit") and "subtitles=caps.ass" in vertical_filter("fit", captions="caps.ass")
    cmd = render_cmd(Path("in.mp4"), 5, 35, Path("o.mp4"), gpu=False)
    assert cmd[cmd.index("-ss") + 1] == "5.000" and "libx264" in cmd and "0:a:0?" in cmd


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_render_produces_vertical_mp4(tmp_path):
    import subprocess
    from clipfinder import Candidate, render_shorts
    src = tmp_path / "game.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=20",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=20", "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", str(src)], check=True)
    words = [(2.0, 2.4, " LETS"), (2.4, 2.8, " GO")]
    made = render_shorts(src, [Candidate(1.0, 17.0)], words, tmp_path / "clips", 1)
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:format=duration", "-of", "csv=p=0", str(made[0])],
                       capture_output=True, text=True, check=True)
    assert "1080,1920" in r.stdout
    dur = float(r.stdout.strip().splitlines()[-1])
    assert 15.5 <= dur <= 16.5
    assert src.stat().st_size > 0  # source untouched


def test_phrases_capped_in_nonstop_chatter():
    from clipfinder import phrases
    words = [(i * 0.4, i * 0.4 + 0.3, " w") for i in range(100)]   # 40 s, never pauses
    ph = phrases(words)
    assert len(ph) >= 4 and all(p.end - p.start <= 10.0 for p in ph)


def test_hook_overlay_wraps_and_tints():
    from clipfinder import ass_captions, hook_lines
    assert hook_lines("this is a very long hook title that keeps going and going forever").count(r"\N") <= 2
    ass = ass_captions([(0.0, 1.0, "HI")], hook="Why nobody talks about this", accent="FFD400")
    hook_line = [l for l in ass.splitlines() if ",Hook," in l and l.startswith("Dialogue")][0]
    assert r"\c&H00D4FF&" in hook_line and "WHY NOBODY" in hook_line
    assert "Style: Hook" in ass


def test_explicit_clips_mode_renders_named_clips_with_saved_words(tmp_path):
    import json
    import subprocess
    from clipfinder import main
    src = tmp_path / "doc.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=12",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=12", "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", str(src)], check=True)
    words = tmp_path / "w.json"
    words.write_text(json.dumps([[1.0, 1.4, " The"], [1.4, 1.9, " truth"]]), encoding="utf-8")
    clips = tmp_path / "c.json"
    clips.write_text(json.dumps([{"start": 0.5, "end": 6.5, "hook": "The truth nobody tells you", "name": "01 truth", "accent": "33CCFF"},
                                 {"start": 6.0, "end": 11.0, "hook": "", "name": "02 next"}]), encoding="utf-8")
    out = tmp_path / "out"
    assert main([str(src), "--words-in", str(words), "--clips", str(clips), "--clips-dir", str(out), "--layout", "fit"]) == 0
    assert sorted(p.name for p in out.glob("*.mp4")) == ["01_truth.mp4", "02_next.mp4"]
    words2 = tmp_path / "w2.json"
    assert main([str(src), "--transcribe-only", "--no-transcribe", "--words-out", str(words2)]) == 0
    assert json.loads(words2.read_text(encoding="utf-8")) == []


def test_minimal_caption_style_is_smaller_and_regular_weight():
    from clipfinder import ass_captions
    bold = [l for l in ass_captions([(0.0, 1.0, "HI")]).splitlines() if l.startswith("Style: Cap")][0]
    mini = [l for l in ass_captions([(0.0, 1.0, "HI")], style="minimal").splitlines() if l.startswith("Style: Cap")][0]
    assert "Arial Black,112" in bold and "Arial,70" in mini
