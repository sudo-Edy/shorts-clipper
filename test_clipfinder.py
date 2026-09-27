import numpy as np
import pytest

from clipfinder import (
    MAX_CLIP_S, MIN_CLIP_S, SAMPLE_RATE, Candidate, Mark, Segment, build_candidates, energy_curve,
    fill_to_minimum, find_peaks, fmt, hype_hits, parse_notes, rank, safe_output, snap,
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
