#!/usr/bin/env python3
"""
Guitar Solo Transcription Pipeline
===================================
Audio file → demucs (htdemucs_6s) → guitar stem → basic-pitch → MIDI
       → fretboard heuristics → Guitar Pro (.gp5)

Dependencies:
    pip install demucs basic-pitch mido PyGuitarPro numpy
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

# Standard tuning in MIDI note numbers, strings 1-6 (high→low)
STANDARD_TUNING = [64, 59, 55, 50, 45, 40]  # E4 B3 G3 D3 A2 E2

MAX_FRET = 24
DEFAULT_BPM = 120

# Guitar playable MIDI range
GUITAR_MIN_MIDI = 40   # E2
GUITAR_MAX_MIDI = 88   # E6


# ─── Step 1: Stem Separation ──────────────────────────────────────────────────

def separate_stems(audio_path: Path, output_dir: Path) -> Path:
    """
    Run demucs htdemucs_6s on *audio_path* and return path to guitar stem.

    demucs writes stems to:
        <output_dir>/htdemucs_6s/<audio_stem>/guitar.wav
    """
    print("[1/4] Stem separation with demucs htdemucs_6s …")

    cmd = [
        sys.executable, "-m", "demucs",
        "-n", "htdemucs_6s",
        "-o", str(output_dir),
        str(audio_path),
    ]
    subprocess.run(cmd, check=True)

    stem_wav = output_dir / "htdemucs_6s" / audio_path.stem / "guitar.wav"
    if not stem_wav.exists():
        raise FileNotFoundError(
            f"Expected guitar stem at {stem_wav}\n"
            "Make sure htdemucs_6s model is downloaded and demucs ran successfully."
        )

    print(f"    → {stem_wav}")
    return stem_wav


# ─── Step 2: Audio → MIDI via basic-pitch ─────────────────────────────────────

def audio_to_midi(guitar_wav: Path, midi_dir: Path) -> Path:
    """Transcribe *guitar_wav* to MIDI using Spotify basic-pitch."""
    print("[2/4] Audio → MIDI with basic-pitch …")

    from basic_pitch.inference import predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH

    predict_and_save(
        audio_path_list=[str(guitar_wav)],
        output_directory=str(midi_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        minimum_frequency=82.4,    # ~E2 — lowest standard guitar note
        maximum_frequency=1318.5,  # ~E6
        minimum_note_length=60,    # ms  — ignore very short artifacts
        onset_threshold=0.5,
        frame_threshold=0.3,
        melodia_trick=True,
    )

    # basic-pitch names the output "<stem>_basic_pitch.mid"
    midi_path = midi_dir / f"{guitar_wav.stem}_basic_pitch.mid"
    if not midi_path.exists():
        candidates = sorted(midi_dir.glob("*.mid"))
        if not candidates:
            raise FileNotFoundError("basic-pitch produced no MIDI file.")
        midi_path = candidates[-1]

    print(f"    → {midi_path}")
    return midi_path


# ─── Step 3: Fretboard Heuristics ─────────────────────────────────────────────

def midi_to_candidates(
    midi_note: int,
    tuning: list[int] = STANDARD_TUNING,
    max_fret: int = MAX_FRET,
) -> list[tuple[int, int]]:
    """
    Return all (string_1indexed, fret) positions for *midi_note* in standard
    tuning.  String 1 is the highest (e4).
    """
    positions = []
    for string_num, open_midi in enumerate(tuning, start=1):
        fret = midi_note - open_midi
        if 0 <= fret <= max_fret:
            positions.append((string_num, fret))
    return positions


def assign_positions(notes: list[dict]) -> list[dict]:
    """
    Assign guitar (string, fret) to every note using dynamic programming.

    Cost between consecutive notes:
        |Δfret| + 0.5 × |Δstring|

    An additional small bias (0.05 × fret) pushes the solver toward lower
    positions, keeping left-hand travel minimal.

    Args:
        notes: list of dicts with at minimum keys: pitch, start, end, velocity
    Returns:
        Same list enriched with 'string' and 'fret' keys.
    """
    if not notes:
        return notes

    # Build candidate list per note
    candidates: list[list[tuple[int, int]]] = []
    for note in notes:
        pos = midi_to_candidates(note["pitch"])
        if not pos:
            # Note out of range — clamp to string 1, best fret
            fret = max(0, min(MAX_FRET, note["pitch"] - STANDARD_TUNING[0]))
            pos = [(1, fret)]
        candidates.append(pos)

    n = len(notes)
    INF = float("inf")

    dp = [[INF] * len(c) for c in candidates]
    back = [[-1] * len(c) for c in candidates]

    # Initialise: slight preference for lower frets
    for j, (_, f) in enumerate(candidates[0]):
        dp[0][j] = 0.05 * f

    # Forward pass
    for i in range(1, n):
        for j, (s2, f2) in enumerate(candidates[i]):
            for k, (s1, f1) in enumerate(candidates[i - 1]):
                cost = dp[i - 1][k] + abs(f2 - f1) + 0.5 * abs(s2 - s1)
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

    result = []
    for i, note in enumerate(notes):
        s, f = candidates[i][path[i]]
        result.append({**note, "string": s, "fret": f})
    return result


# ─── Step 4: MIDI → Guitar Pro ────────────────────────────────────────────────

def ticks_to_gp_duration(ticks: int, tpb: int) -> guitarpro.Duration:
    """
    Map a note duration in MIDI ticks to the nearest Guitar Pro Duration.

    GP duration values: 1=whole, 2=half, 4=quarter, 8=eighth,
                        16=sixteenth, 32=thirty-second, 64=sixty-fourth
    Also considers dotted variants (×1.5).
    """
    beat = tpb  # ticks per quarter note

    candidates: list[tuple[int, guitarpro.Duration]] = []
    for value in (1, 2, 4, 8, 16, 32, 64):
        note_ticks = int(beat * 4 / value)
        candidates.append((note_ticks, guitarpro.Duration(value)))
        dotted_ticks = int(note_ticks * 1.5)
        candidates.append((dotted_ticks, guitarpro.Duration(value, isDotted=True)))

    best_dur = guitarpro.Duration(4)
    best_diff = abs(ticks - beat)
    for note_ticks, dur in candidates:
        diff = abs(ticks - note_ticks)
        if diff < best_diff:
            best_diff = diff
            best_dur = dur
    return best_dur


def load_midi_notes(midi_path: Path) -> tuple[list[dict], int, int]:
    """
    Parse a MIDI file and return (notes, ticks_per_beat, bpm).

    Notes are returned sorted by start tick, filtered to guitar range.
    """
    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat

    tempo_us = 500_000  # 120 BPM default
    notes: list[dict] = []

    for track in mid.tracks:
        tick = 0
        active: dict[int, tuple[int, int]] = {}  # pitch → (start_tick, velocity)
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
    bpm = round(60_000_000 / tempo_us)
    return notes, tpb, bpm


def build_gp5(
    notes_with_pos: list[dict],
    tpb: int,
    bpm: int,
    output_path: Path,
) -> None:
    """Construct a Guitar Pro 5 song from positioned notes and write it."""

    beats_per_bar = 4
    ticks_per_bar = tpb * beats_per_bar

    # ── Measure count ──
    if notes_with_pos:
        last_tick = max(n["end"] for n in notes_with_pos)
        num_measures = max(1, int(np.ceil(last_tick / ticks_per_bar)))
    else:
        num_measures = 1

    # ── Song skeleton ──
    song = guitarpro.Song()
    song.tempo = bpm

    # ── Measure headers ──
    song.measureHeaders = []
    for i in range(num_measures):
        header = guitarpro.MeasureHeader()
        header.number = i + 1
        header.start = guitarpro.Duration.quarterTime * beats_per_bar * i + guitarpro.Duration.quarterTime
        header.timeSignature.numerator = beats_per_bar
        header.timeSignature.denominator = guitarpro.Duration(4)
        header.tempo.value = bpm
        song.measureHeaders.append(header)

    # ── Track ──
    track = song.tracks[0]
    track.name = "Guitar Solo"
    track.isPercussionTrack = False
    track.strings = [
        guitarpro.GuitarString(number=i + 1, value=v)
        for i, v in enumerate(STANDARD_TUNING)
    ]
    track.fretCount = MAX_FRET

    # ── Measures ──
    track.measures = []
    for mi in range(num_measures):
        bar_start = mi * ticks_per_bar
        bar_end = bar_start + ticks_per_bar

        measure = guitarpro.Measure(track, song.measureHeaders[mi])
        voice = measure.voices[0]
        voice.beats = []

        bar_notes = [n for n in notes_with_pos if bar_start <= n["start"] < bar_end]

        if not bar_notes:
            # Whole-bar rest
            rest = guitarpro.Beat(voice)
            rest.status = guitarpro.BeatStatus.rest
            rest.duration = guitarpro.Duration(1)
            voice.beats.append(rest)
        else:
            cursor = bar_start
            for nd in bar_notes:
                # Rest gap before this note
                gap = nd["start"] - cursor
                min_gap = tpb // 8  # ≈ 32nd note at tpb=480
                if gap >= min_gap:
                    rest = guitarpro.Beat(voice)
                    rest.status = guitarpro.BeatStatus.rest
                    rest.duration = ticks_to_gp_duration(gap, tpb)
                    voice.beats.append(rest)

                # Note beat
                dur_ticks = nd["end"] - nd["start"]
                beat = guitarpro.Beat(voice)
                beat.status = guitarpro.BeatStatus.normal
                beat.duration = ticks_to_gp_duration(max(dur_ticks, tpb // 16), tpb)

                gp_note = guitarpro.Note(beat)
                gp_note.string = nd["string"]
                gp_note.fret = nd["fret"]
                gp_note.velocity = min(127, nd["velocity"])
                beat.notes.append(gp_note)
                voice.beats.append(beat)

                cursor = nd["end"]

        track.measures.append(measure)

    guitarpro.write(song, str(output_path))
    print(f"    → {output_path}")


# ─── Orchestration ────────────────────────────────────────────────────────────

def run(
    audio_path: Path,
    output_path: Path,
    stems_dir: Path,
    midi_dir: Path,
    skip_demucs: bool,
    force_bpm: int | None,
) -> None:

    # 1. Stem separation
    if skip_demucs:
        guitar_wav = audio_path
        print(f"[1/4] Skipping demucs — treating input as guitar stem")
    else:
        guitar_wav = separate_stems(audio_path, stems_dir)

    # 2. Audio → MIDI
    midi_path = audio_to_midi(guitar_wav, midi_dir)

    # 3. Parse MIDI + assign positions
    print("[3/4] Assigning fretboard positions …")
    raw_notes, tpb, detected_bpm = load_midi_notes(midi_path)
    bpm = force_bpm or detected_bpm
    print(f"    {len(raw_notes)} notes  |  BPM={bpm}  |  tpb={tpb}")

    notes_with_pos = assign_positions(raw_notes)

    # 4. Build .gp5
    print("[4/4] Writing Guitar Pro 5 file …")
    build_gp5(notes_with_pos, tpb, bpm, output_path)

    print(f"\nDone → {output_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a guitar solo from audio to Guitar Pro (.gp5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (stem separation + transcription)
  python transcribe.py solo.wav

  # Skip demucs if you already have a clean guitar recording
  python transcribe.py clean_guitar.wav --skip-demucs

  # Specify BPM manually
  python transcribe.py solo.wav --bpm 140 -o output.gp5
""",
    )
    parser.add_argument("input", type=Path, help="Input audio file (wav / mp3 / flac)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output .gp5 file (default: <input>.gp5)")
    parser.add_argument("--stems-dir", type=Path, default=None,
                        help="Directory for demucs output (default: ./stems)")
    parser.add_argument("--midi-dir", type=Path, default=None,
                        help="Directory for MIDI output (default: ./midi)")
    parser.add_argument("--skip-demucs", action="store_true",
                        help="Skip stem separation; treat input as guitar stem")
    parser.add_argument("--bpm", type=int, default=None,
                        help="Override BPM (otherwise auto-detected from MIDI)")
    args = parser.parse_args()

    audio_path = args.input.resolve()
    if not audio_path.exists():
        sys.exit(f"Error: file not found — {audio_path}")

    output_path = (args.output or audio_path.with_suffix(".gp5")).resolve()
    stems_dir = (args.stems_dir or Path("stems")).resolve()
    midi_dir = (args.midi_dir or Path("midi")).resolve()

    stems_dir.mkdir(parents=True, exist_ok=True)
    midi_dir.mkdir(parents=True, exist_ok=True)

    try:
        run(audio_path, output_path, stems_dir, midi_dir, args.skip_demucs, args.bpm)
    except subprocess.CalledProcessError as exc:
        sys.exit(f"External command failed: {exc}")
    except Exception as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
