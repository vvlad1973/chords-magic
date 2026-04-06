#!/usr/bin/env python3
"""
Guitar Solo Transcription Pipeline
===================================
Video/Audio → (ffmpeg extract) → preprocess → demucs (htdemucs_6s)
           → guitar stem → basic-pitch → MIDI cleanup
           → fretboard DP (hand-position model) → Guitar Pro (.gp5)

Dependencies:
    pip install demucs basic-pitch mido PyGuitarPro numpy
    # system: ffmpeg must be in PATH
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import mido
import numpy as np
import guitarpro


# ─── Constants ────────────────────────────────────────────────────────────────

STANDARD_TUNING = [64, 59, 55, 50, 45, 40]  # E4 B3 G3 D3 A2 E2  (string 1→6)
MAX_FRET   = 24
HAND_SPAN  = 4       # frets reachable without shifting hand position
SHIFT_PENALTY = 2.0  # extra cost per fret beyond HAND_SPAN

DEFAULT_BPM          = 120
GUITAR_MIN_MIDI      = 40   # E2
GUITAR_MAX_MIDI      = 88   # E6

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
    ".m4v", ".flv", ".wmv", ".ts", ".mts", ".vob",
}


# ─── Step 0: Video → Audio ────────────────────────────────────────────────────

def extract_audio(video_path: Path, work_dir: Path) -> Path:
    """Extract and normalise audio from a video file via ffmpeg."""
    _check_ffmpeg()
    print("[0] Extracting audio from video …")
    out = work_dir / (video_path.stem + "_audio.wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn",                      # drop video stream
            "-ac", "1",                 # mono
            "-ar", "44100",             # sample rate optimal for basic-pitch
            "-af", "loudnorm",          # EBU R128 loudness normalisation
            str(out),
        ],
        check=True,
    )
    print(f"    → {out}")
    return out


# ─── Step 1: Audio Preprocessing ─────────────────────────────────────────────

def preprocess_audio(audio_path: Path, work_dir: Path) -> Path:
    """
    Convert any audio file to mono 44100 Hz WAV with loudnorm.
    basic-pitch performs best on exactly this format.
    """
    _check_ffmpeg()
    print("[1] Preprocessing audio (mono / 44100 Hz / loudnorm) …")
    out = work_dir / (audio_path.stem + "_norm.wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ac", "1",
            "-ar", "44100",
            "-af", "loudnorm",
            str(out),
        ],
        check=True,
    )
    print(f"    → {out}")
    return out


def _check_ffmpeg() -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit("Error: ffmpeg not found. Install it and make sure it is in PATH.")


# ─── Step 2: Stem Separation ──────────────────────────────────────────────────

def separate_stems(audio_path: Path, output_dir: Path) -> Path:
    """Run demucs htdemucs_6s; return path to guitar.wav stem."""
    print("[2] Stem separation with demucs htdemucs_6s …")
    subprocess.run(
        [
            sys.executable, "-m", "demucs",
            "-n", "htdemucs_6s",
            "-o", str(output_dir),
            str(audio_path),
        ],
        check=True,
    )
    stem_wav = output_dir / "htdemucs_6s" / audio_path.stem / "guitar.wav"
    if not stem_wav.exists():
        raise FileNotFoundError(
            f"Expected guitar stem at {stem_wav}\n"
            "Make sure htdemucs_6s model is downloaded."
        )
    print(f"    → {stem_wav}")
    return stem_wav


# ─── Step 3: Audio → MIDI via basic-pitch ─────────────────────────────────────

def audio_to_midi(
    guitar_wav: Path,
    midi_dir: Path,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_ms: float = 60.0,
) -> Path:
    """Transcribe guitar stem to MIDI using Spotify basic-pitch."""
    print(
        f"[3] Audio → MIDI with basic-pitch "
        f"(onset={onset_threshold}, frame={frame_threshold}, "
        f"min_note={min_note_ms}ms) …"
    )
    from basic_pitch.inference import predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH

    # basic-pitch skips (not overwrites) existing files — remove them first
    expected = midi_dir / f"{guitar_wav.stem}_basic_pitch.mid"
    if expected.exists():
        expected.unlink()
        print(f"    Removed existing {expected.name}")
    # Also remove any other .mid files in the dir to avoid picking up stale ones
    for f in midi_dir.glob("*.mid"):
        f.unlink()

    predict_and_save(
        audio_path_list=[str(guitar_wav)],
        output_directory=str(midi_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        minimum_frequency=82.4,    # E2
        maximum_frequency=1318.5,  # E6
        minimum_note_length=min_note_ms,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        melodia_trick=True,
    )

    midi_path = midi_dir / f"{guitar_wav.stem}_basic_pitch.mid"
    if not midi_path.exists():
        candidates = sorted(midi_dir.glob("*.mid"))
        if not candidates:
            raise FileNotFoundError("basic-pitch produced no MIDI file.")
        midi_path = candidates[-1]

    print(f"    → {midi_path}")
    return midi_path


# ─── Step 4: MIDI Cleanup ─────────────────────────────────────────────────────

def clean_notes(
    notes: list[dict],
    tpb: int,
    tempo_us: int = 500_000,
    min_duration_ms: float = 40.0,
    quantize_to: int | None = 16,
    dedup_ticks: int | None = None,
) -> list[dict]:
    """
    Post-process raw MIDI notes from basic-pitch:

    1. Remove notes shorter than *min_duration_ms*.
    2. Quantize start/end to the nearest *quantize_to*-note grid
       (e.g. 16 → sixteenth-note grid).
    3. Remove duplicates: if the same pitch appears twice within
       *dedup_ticks* ticks, keep only the first.

    Args:
        notes:           sorted list of note dicts (pitch, start, end, velocity)
        tpb:             ticks per beat
        tempo_us:        microseconds per beat (for ms→ticks conversion)
        min_duration_ms: notes shorter than this are dropped
        quantize_to:     grid note value (4, 8, 16, 32, None=off)
        dedup_ticks:     duplicate window in ticks (default tpb//8)
    Returns:
        Cleaned, sorted list.
    """
    # ── 1. Remove too-short notes ──
    ticks_per_ms = tpb / (tempo_us / 1000.0)
    min_ticks = min_duration_ms * ticks_per_ms
    notes = [n for n in notes if (n["end"] - n["start"]) >= min_ticks]

    # ── 2. Quantize to grid ──
    if quantize_to is not None:
        grid = max(1, int(tpb * 4 / quantize_to))

        def snap(tick: int) -> int:
            return round(tick / grid) * grid

        quantized = []
        for n in notes:
            s = snap(n["start"])
            e = snap(n["end"])
            if e <= s:
                e = s + grid          # minimum 1 grid unit
            quantized.append({**n, "start": s, "end": e})
        notes = quantized

    notes.sort(key=lambda n: (n["start"], n["pitch"]))

    # ── 3. Remove duplicates ──
    if dedup_ticks is None:
        dedup_ticks = max(1, tpb // 8)

    cleaned: list[dict] = []
    last_start: dict[int, int] = {}   # pitch → last accepted start tick
    for n in notes:
        prev = last_start.get(n["pitch"])
        if prev is None or (n["start"] - prev) > dedup_ticks:
            cleaned.append(n)
            last_start[n["pitch"]] = n["start"]

    removed = len(notes) - len(cleaned)
    if removed:
        print(f"    MIDI cleanup: removed {removed} noisy/duplicate notes "
              f"({len(cleaned)} remain)")
    return cleaned


# ─── Step 5: Fretboard Heuristics (DP + hand-position model) ─────────────────

def midi_to_candidates(
    midi_note: int,
    tuning: list[int] = STANDARD_TUNING,
    max_fret: int = MAX_FRET,
) -> list[tuple[int, int]]:
    """All (string_1indexed, fret) positions for *midi_note* in standard tuning."""
    out = []
    for s, open_midi in enumerate(tuning, start=1):
        fret = midi_note - open_midi
        if 0 <= fret <= max_fret:
            out.append((s, fret))
    return out


def _transition_cost(
    s1: int, f1: int,
    s2: int, f2: int,
) -> float:
    """
    Cost of moving from position (s1, f1) to (s2, f2).

    Model:
    - Fret distance is primary cost (left-hand travel).
    - String distance has half weight (pick/right-hand).
    - A jump exceeding HAND_SPAN forces a position shift:
      every extra fret costs SHIFT_PENALTY (discourages large leaps).
    """
    df = abs(f2 - f1)
    ds = abs(s2 - s1)
    shift_cost = max(0.0, df - HAND_SPAN) * SHIFT_PENALTY
    return df + 0.5 * ds + shift_cost


def assign_positions(notes: list[dict]) -> list[dict]:
    """
    Assign (string, fret) to every note using Viterbi-style DP.

    State: candidate index at each note.
    Transition cost: _transition_cost (includes hand-position penalty).
    Initial cost: 0.05 × fret (mild preference for lower positions).
    """
    if not notes:
        return notes

    candidates: list[list[tuple[int, int]]] = []
    for note in notes:
        pos = midi_to_candidates(note["pitch"])
        if not pos:
            fret = max(0, min(MAX_FRET, note["pitch"] - STANDARD_TUNING[0]))
            pos = [(1, fret)]
        candidates.append(pos)

    n = len(notes)
    INF = float("inf")
    dp   = [[INF] * len(c) for c in candidates]
    back = [[-1]  * len(c) for c in candidates]

    for j, (_, f) in enumerate(candidates[0]):
        dp[0][j] = 0.05 * f

    for i in range(1, n):
        for j, (s2, f2) in enumerate(candidates[i]):
            for k, (s1, f1) in enumerate(candidates[i - 1]):
                cost = dp[i - 1][k] + _transition_cost(s1, f1, s2, f2)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    back[i][j] = k

    # Traceback
    idx = int(np.argmin(dp[n - 1]))
    path = [idx]
    for i in range(n - 1, 0, -1):
        idx = back[i][idx]
        path.append(idx)
    path.reverse()

    return [
        {**note, "string": candidates[i][path[i]][0], "fret": candidates[i][path[i]][1]}
        for i, note in enumerate(notes)
    ]


# ─── Step 6: MIDI → Guitar Pro ───────────────────────────────────────────────

def ticks_to_gp_duration(ticks: int, tpb: int) -> guitarpro.Duration:
    """Nearest Guitar Pro Duration to *ticks* (including dotted values)."""
    beat = tpb
    best_dur  = guitarpro.Duration(4)
    best_diff = abs(ticks - beat)
    for value in (1, 2, 4, 8, 16, 32, 64):
        note_ticks = int(beat * 4 / value)
        for factor, dotted in ((1, False), (1.5, True)):
            t = int(note_ticks * factor)
            diff = abs(ticks - t)
            if diff < best_diff:
                best_diff = diff
                best_dur  = guitarpro.Duration(value, isDotted=dotted)
    return best_dur


def load_midi_notes(midi_path: Path) -> tuple[list[dict], int, int]:
    """Parse MIDI → (notes, tpb, bpm).  Notes filtered to guitar range."""
    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat
    tempo_us = 500_000
    notes: list[dict] = []

    for track in mid.tracks:
        tick = 0
        active: dict[int, tuple[int, int]] = {}
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                tempo_us = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = (tick, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    start, vel = active.pop(msg.note)
                    if GUITAR_MIN_MIDI <= msg.note <= GUITAR_MAX_MIDI:
                        notes.append({
                            "pitch": msg.note,
                            "start": start,
                            "end": tick,
                            "velocity": vel,
                        })

    notes.sort(key=lambda n: n["start"])
    return notes, tpb, round(60_000_000 / tempo_us), tempo_us


def build_gp5(
    notes_with_pos: list[dict],
    tpb: int,
    bpm: int,
    output_path: Path,
) -> None:
    """Write a Guitar Pro 5 file from positioned notes."""
    beats_per_bar   = 4
    ticks_per_bar   = tpb * beats_per_bar

    if notes_with_pos:
        last_tick = max(n["end"] for n in notes_with_pos)
        num_measures = max(1, int(np.ceil(last_tick / ticks_per_bar)))
    else:
        num_measures = 1

    song = guitarpro.Song()
    song.tempo = bpm

    song.measureHeaders = []
    for i in range(num_measures):
        h = guitarpro.MeasureHeader()
        h.number = i + 1
        h.start  = guitarpro.Duration.quarterTime * beats_per_bar * i + guitarpro.Duration.quarterTime
        h.timeSignature.numerator   = beats_per_bar
        h.timeSignature.denominator = guitarpro.Duration(4)
        # Tempo is set globally on song.tempo; per-measure tempo not supported
        # in all PyGuitarPro versions — skip to avoid AttributeError.
        if hasattr(h, 'tempo') and h.tempo is not None:
            try:
                h.tempo.value = bpm
            except AttributeError:
                pass
        song.measureHeaders.append(h)

    track = song.tracks[0]
    track.name = "Guitar Solo"
    track.isPercussionTrack = False
    track.strings = [
        guitarpro.GuitarString(number=i + 1, value=v)
        for i, v in enumerate(STANDARD_TUNING)
    ]
    track.fretCount = MAX_FRET

    track.measures = []
    for mi in range(num_measures):
        bar_start = mi * ticks_per_bar
        bar_end   = bar_start + ticks_per_bar

        measure = guitarpro.Measure(track, song.measureHeaders[mi])
        voice   = measure.voices[0]
        voice.beats = []

        bar_notes = [n for n in notes_with_pos if bar_start <= n["start"] < bar_end]

        if not bar_notes:
            rest          = guitarpro.Beat(voice)
            rest.status   = guitarpro.BeatStatus.rest
            rest.duration = guitarpro.Duration(1)
            voice.beats.append(rest)
        else:
            cursor = bar_start
            for nd in bar_notes:
                gap = nd["start"] - cursor
                if gap >= tpb // 8:
                    rest          = guitarpro.Beat(voice)
                    rest.status   = guitarpro.BeatStatus.rest
                    rest.duration = ticks_to_gp_duration(gap, tpb)
                    voice.beats.append(rest)

                dur_ticks     = max(nd["end"] - nd["start"], tpb // 16)
                beat          = guitarpro.Beat(voice)
                beat.status   = guitarpro.BeatStatus.normal
                beat.duration = ticks_to_gp_duration(dur_ticks, tpb)

                gp_note          = guitarpro.Note(beat)
                gp_note.string   = nd["string"]
                gp_note.fret     = nd["fret"]
                gp_note.velocity = min(127, nd["velocity"])
                beat.notes.append(gp_note)
                voice.beats.append(beat)

                cursor = nd["end"]

        track.measures.append(measure)

    guitarpro.write(song, str(output_path))
    print(f"    → {output_path}")


# ─── Orchestration ────────────────────────────────────────────────────────────

def run(
    input_path: Path,
    output_path: Path,
    work_dir: Path,
    stems_dir: Path,
    midi_dir: Path,
    skip_demucs: bool,
    skip_preprocess: bool,
    force_bpm: int | None,
    onset: float,
    frame: float,
    min_note_ms: float,
    quantize_to: int | None,
    no_dedup: bool,
) -> None:
    step = 0

    def _step(label: str) -> None:
        nonlocal step
        step += 1
        # Labels already printed inside each function; this is just a counter hook.

    # ── 0. Video → audio ──
    if input_path.suffix.lower() in VIDEO_EXTENSIONS:
        audio_path = extract_audio(input_path, work_dir)
    else:
        audio_path = input_path

    # ── 1. Preprocess ──
    if skip_preprocess:
        norm_path = audio_path
        print("[1] Skipping audio preprocessing")
    else:
        norm_path = preprocess_audio(audio_path, work_dir)

    # ── 2. Stem separation ──
    if skip_demucs:
        guitar_wav = norm_path
        print("[2] Skipping demucs — treating input as guitar stem")
    else:
        guitar_wav = separate_stems(norm_path, stems_dir)

    # ── 3. Audio → MIDI ──
    midi_path = audio_to_midi(
        guitar_wav, midi_dir,
        onset_threshold=onset,
        frame_threshold=frame,
        min_note_ms=min_note_ms,
    )

    # ── 4. Load + clean MIDI ──
    print("[4] Loading and cleaning MIDI …")
    raw_notes, tpb, detected_bpm, tempo_us = load_midi_notes(midi_path)
    bpm = force_bpm or detected_bpm
    print(f"    {len(raw_notes)} raw notes  |  BPM={bpm}  |  tpb={tpb}")

    cleaned = clean_notes(
        raw_notes,
        tpb=tpb,
        tempo_us=tempo_us,
        min_duration_ms=min_note_ms * 0.5,   # half of basic-pitch min
        quantize_to=quantize_to,
        dedup_ticks=None if not no_dedup else 0,
    )

    # ── 5. Assign fretboard positions ──
    print("[5] Assigning fretboard positions (hand-position DP) …")
    notes_with_pos = assign_positions(cleaned)

    # ── 6. Write .gp5 ──
    print("[6] Writing Guitar Pro 5 …")
    build_gp5(notes_with_pos, tpb, bpm, output_path)

    print(f"\nDone → {output_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a guitar solo from audio/video to Guitar Pro (.gp5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From video (auto extracts audio)
  python transcribe.py concert.mp4

  # From audio, full pipeline
  python transcribe.py solo.wav

  # Skip stem separation (already a clean guitar recording)
  python transcribe.py clean_guitar.wav --skip-demucs

  # Tune transcription sensitivity
  python transcribe.py solo.wav --onset 0.4 --frame 0.25 --min-note-ms 80

  # 8th-note quantization, force BPM
  python transcribe.py solo.wav --quantize 8 --bpm 140

  # Skip preprocessing (input already 44100 Hz mono)
  python transcribe.py guitar.wav --skip-preprocess --skip-demucs
""",
    )
    parser.add_argument("input", type=Path,
                        help="Input file: audio (wav/mp3/flac) or video (mp4/mkv/avi/…)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output .gp5 (default: <input>.gp5)")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Working directory for intermediate files (default: ./work)")
    parser.add_argument("--stems-dir", type=Path, default=None,
                        help="Directory for demucs stems (default: <work-dir>/stems)")
    parser.add_argument("--midi-dir", type=Path, default=None,
                        help="Directory for MIDI files (default: <work-dir>/midi)")

    # Pipeline control
    parser.add_argument("--skip-demucs", action="store_true",
                        help="Skip stem separation; input treated as guitar stem")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip ffmpeg preprocessing (input already 44100 Hz mono)")

    # basic-pitch tuning
    bp = parser.add_argument_group("basic-pitch tuning")
    bp.add_argument("--onset", type=float, default=0.5, metavar="THRESH",
                    help="Onset detection threshold 0–1 (default 0.5; lower→more notes)")
    bp.add_argument("--frame", type=float, default=0.3, metavar="THRESH",
                    help="Frame detection threshold 0–1 (default 0.3)")
    bp.add_argument("--min-note-ms", type=float, default=60.0, metavar="MS",
                    help="Minimum note duration in ms (default 60)")

    # MIDI cleanup
    cl = parser.add_argument_group("MIDI cleanup")
    cl.add_argument("--quantize", type=int, default=16, metavar="N",
                    help="Quantize to N-th note grid: 4 8 16 32 0=off (default 16)")
    cl.add_argument("--no-dedup", action="store_true",
                    help="Disable duplicate note removal")

    # Other
    parser.add_argument("--bpm", type=int, default=None,
                        help="Override BPM (default: auto-detect from MIDI)")

    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        sys.exit(f"Error: file not found — {input_path}")

    output_path = (args.output or input_path.with_suffix(".gp5")).resolve()
    work_dir    = (args.work_dir or Path("work")).resolve()
    stems_dir   = (args.stems_dir or work_dir / "stems").resolve()
    midi_dir    = (args.midi_dir  or work_dir / "midi").resolve()

    for d in (work_dir, stems_dir, midi_dir):
        d.mkdir(parents=True, exist_ok=True)

    quantize_to = args.quantize if args.quantize > 0 else None

    try:
        run(
            input_path    = input_path,
            output_path   = output_path,
            work_dir      = work_dir,
            stems_dir     = stems_dir,
            midi_dir      = midi_dir,
            skip_demucs   = args.skip_demucs,
            skip_preprocess = args.skip_preprocess,
            force_bpm     = args.bpm,
            onset         = args.onset,
            frame         = args.frame,
            min_note_ms   = args.min_note_ms,
            quantize_to   = quantize_to,
            no_dedup      = args.no_dedup,
        )
    except subprocess.CalledProcessError as exc:
        sys.exit(f"External command failed: {exc}")
    except Exception as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
