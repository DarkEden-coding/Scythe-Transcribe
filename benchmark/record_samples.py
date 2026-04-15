#!/usr/bin/env python3
"""
Record benchmark audio samples.

For each paragraph in paragraphs.txt, displays the text, counts down, records
while you read it aloud, and saves the WAV + ground-truth to disk.

Usage:
    python record_samples.py                   # Record all unrecorded paragraphs
    python record_samples.py --redo            # Re-record all paragraphs
    python record_samples.py --list            # Show recording status and exit
    python record_samples.py --index 1,3,5     # Record specific paragraphs (1-based)

Requirements: pip install -r requirements.txt
"""

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ── dependency check ──────────────────────────────────────────────────────────
try:
    import numpy as np
    import sounddevice as sd
    import soundfile as sf
except ImportError as _err:
    print(f"Missing dependency: {_err}")
    print("Run:  pip install -r requirements.txt")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
SAMPLES_DIR = SCRIPT_DIR / "samples" / "recordings"
PARAGRAPHS_FILE = SCRIPT_DIR / "paragraphs.txt"
MANIFEST_FILE = SAMPLES_DIR / "manifest.json"

SAMPLE_RATE = 16_000
CHANNELS = 1


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_paragraphs() -> list[str]:
    with open(PARAGRAPHS_FILE, encoding="utf-8") as f:
        content = f.read()
    return [p.strip() for p in content.split("\n\n") if p.strip()]


def load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"recordings": []}


def save_manifest(manifest: dict) -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def save_wav(audio: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


# ── microphone selection ──────────────────────────────────────────────────────

def list_input_devices() -> list[tuple[int, dict]]:
    """Return [(sd_index, device_info), ...] for all input-capable devices."""
    devices = sd.query_devices()
    return [
        (i, d)
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


def select_microphone() -> tuple[int | None, str]:
    """
    Interactively prompt the user to choose an input device.
    Returns (device_index, device_name).
    device_index is None when the user wants the system default.
    """
    inputs = list_input_devices()
    if not inputs:
        print("No input devices found. Using system default.")
        return None, "system default"

    try:
        default_idx = sd.default.device[0]  # sounddevice default input index
    except Exception:
        default_idx = -1

    print("\nAvailable microphones:")
    print(f"  {'#':>3}  {'Device name':<50}  {'Rate':>6}  {'Ch':>3}")
    print("  " + "-" * 68)

    # Entry 0 is always "system default".
    print(f"  {'0':>3}  {'System default':<50}")

    for pos, (sd_idx, dev) in enumerate(inputs, start=1):
        name = dev["name"]
        rate = int(dev["default_samplerate"])
        ch   = int(dev["max_input_channels"])
        marker = " *" if sd_idx == default_idx else ""
        print(f"  {pos:>3}  {name:<50}  {rate:>6}  {ch:>3}{marker}")

    print("  (* = current system default)")
    print()

    while True:
        try:
            raw = input("  Select microphone [0 = system default]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nUsing system default.")
            return None, "system default"

        if raw == "" or raw == "0":
            print("  Using system default.\n")
            return None, "system default"

        try:
            choice = int(raw)
        except ValueError:
            print("  Please enter a number.")
            continue

        if 1 <= choice <= len(inputs):
            sd_idx, dev = inputs[choice - 1]
            name = dev["name"]
            print(f"  Selected: {name}\n")
            return sd_idx, name

        print(f"  Enter a number between 0 and {len(inputs)}.")


# ── audio recording ───────────────────────────────────────────────────────────

def record_until_enter(device: int | None = None) -> np.ndarray:
    """
    Record from `device` (sounddevice index) at 16 kHz mono.
    Pass device=None to use the system default input.
    Returns a float32 array when the user presses Enter.
    If the device doesn't support 16 kHz natively, records at its default
    rate and resamples down via linear interpolation.
    """
    frames: list[np.ndarray] = []
    active = threading.Event()
    active.set()
    native_rate: list[int] = [SAMPLE_RATE]

    def callback(indata: np.ndarray, _frames, _time, status):
        if status:
            print(f"  [audio: {status}]", file=sys.stderr, flush=True)
        if active.is_set():
            frames.append(indata.copy())

    # Try 16 kHz first; fall back to the device's native rate if unsupported.
    try:
        stream = sd.InputStream(
            device=device,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=callback,
            blocksize=1024,
        )
    except Exception:
        dev_info = sd.query_devices(device=device, kind="input")
        fallback_rate = int(dev_info["default_samplerate"])
        native_rate[0] = fallback_rate
        stream = sd.InputStream(
            device=device,
            samplerate=fallback_rate,
            channels=CHANNELS,
            dtype="float32",
            callback=callback,
            blocksize=1024,
        )

    print("  [Recording... press Enter to stop]", flush=True)
    with stream:
        input()
        active.clear()
        time.sleep(0.05)

    if not frames:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(frames, axis=0).squeeze()

    # Resample to 16 kHz if we recorded at a different rate.
    src_rate = native_rate[0]
    if src_rate != SAMPLE_RATE:
        ratio = SAMPLE_RATE / src_rate
        new_len = int(len(audio) * ratio)
        old_indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(old_indices, np.arange(len(audio)), audio).astype(np.float32)

    return audio


# ── display helpers ───────────────────────────────────────────────────────────

def wrap_lines(text: str, width: int = 68, indent: str = "  ") -> str:
    words = text.split()
    lines: list[str] = []
    line: list[str] = []
    length = 0
    for word in words:
        if length + len(word) + (1 if line else 0) > width:
            lines.append(indent + " ".join(line))
            line = [word]
            length = len(word)
        else:
            line.append(word)
            length += len(word) + (1 if len(line) > 1 else 0)
    if line:
        lines.append(indent + " ".join(line))
    return "\n".join(lines)


def list_status(paragraphs: list[str], manifest: dict) -> None:
    recorded = {r["index"] for r in manifest.get("recordings", [])}
    print(f"\n{'#':>4}  {'Status':>10}  Preview")
    print("-" * 72)
    for i, para in enumerate(paragraphs):
        status = "RECORDED" if i in recorded else "missing"
        preview = para.replace("\n", " ")[:60]
        if len(para) > 60:
            preview += "..."
        print(f"{i + 1:>4}  {status:>10}  {preview}")
    print(f"\n{len(recorded)}/{len(paragraphs)} recorded\n")


# ── recording session ─────────────────────────────────────────────────────────

def run_session(
    indices: list[int],
    paragraphs: list[str],
    manifest: dict,
    device: int | None,
    device_name: str,
) -> None:
    existing: dict[int, dict] = {r["index"]: r for r in manifest.get("recordings", [])}

    for pos, idx in enumerate(indices):
        para = paragraphs[idx]
        sample_id = f"sample_{idx + 1:03d}"
        wav_path = SAMPLES_DIR / f"{sample_id}.wav"
        word_count = len(para.split())
        approx_sec = max(5, round(word_count / 2.5))

        print()
        print("=" * 72)
        print(f"  Paragraph {idx + 1} of {len(paragraphs)}  "
              f"({pos + 1}/{len(indices)} in this session)")
        print("=" * 72)
        print()
        print(wrap_lines(para))
        print()
        print(f"  ({word_count} words, ~{approx_sec}s to read aloud)")
        print()

        while True:
            try:
                choice = input("  [Enter]=record  [s]=skip  [q]=quit: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nStopped. Progress saved.")
                save_manifest(manifest)
                return

            if choice == "q":
                print("\nStopped. Progress saved.")
                save_manifest(manifest)
                return

            if choice == "s":
                print("  Skipped.")
                break

            if choice == "":
                # Countdown
                for n in range(3, 0, -1):
                    print(f"  Starting in {n}...", end="\r", flush=True)
                    time.sleep(1)
                print("  Go!              ")

                try:
                    audio = record_until_enter(device=device)
                except Exception as exc:
                    print(f"  Recording failed: {exc}")
                    print("  Press Enter to try again, or [s] to skip.")
                    continue

                if audio.size < SAMPLE_RATE:  # less than 1 second
                    print("  Recording too short (< 1s). Try again.")
                    continue

                duration = len(audio) / SAMPLE_RATE
                print(f"  Captured {duration:.1f}s of audio.")

                try:
                    confirm = input("  Keep? [Enter]=yes  [r]=redo  [s]=skip: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    confirm = "s"

                if confirm == "s":
                    print("  Skipped.")
                    break
                if confirm == "r":
                    continue

                # Save WAV and update manifest.
                save_wav(audio, wav_path)
                rec = {
                    "index": idx,
                    "id": sample_id,
                    "wav_file": f"{sample_id}.wav",
                    "ground_truth": para,
                    "duration_sec": round(duration, 3),
                    "word_count": word_count,
                    "recorded_at": datetime.now().isoformat(),
                    "microphone": device_name,
                }
                existing[idx] = rec
                manifest["recordings"] = sorted(
                    existing.values(), key=lambda r: r["index"]
                )
                save_manifest(manifest)
                print(f"  Saved {wav_path.name}  ({duration:.1f}s)")
                break


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record benchmark audio samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--redo", action="store_true",
        help="Re-record already-recorded paragraphs too",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List recording status and exit",
    )
    parser.add_argument(
        "--index", type=str,
        help="Comma-separated 1-based indices to record (e.g. 1,3,5)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Sounddevice input device index or name substring to skip the picker",
    )
    args = parser.parse_args()

    paragraphs = load_paragraphs()
    if not paragraphs:
        print(f"No paragraphs found in {PARAGRAPHS_FILE}")
        sys.exit(1)

    manifest = load_manifest()

    if args.list:
        list_status(paragraphs, manifest)
        return

    recorded_indices = {r["index"] for r in manifest.get("recordings", [])}

    if args.index:
        try:
            requested = [int(x.strip()) - 1 for x in args.index.split(",")]
        except ValueError:
            print("--index must be comma-separated integers, e.g. --index 1,3,5")
            sys.exit(1)
        indices = [i for i in requested if 0 <= i < len(paragraphs)]
        if not indices:
            print("No valid indices after filtering.")
            sys.exit(1)
    elif args.redo:
        indices = list(range(len(paragraphs)))
    else:
        indices = [i for i in range(len(paragraphs)) if i not in recorded_indices]

    if not indices:
        print("All paragraphs already recorded. Use --redo to re-record.")
        list_status(paragraphs, manifest)
        return

    # ── microphone selection ──────────────────────────────────────────────────
    if args.device is not None:
        # Resolve the --device argument: try as integer index first, then name.
        try:
            device: int | None = int(args.device)
            device_name = sd.query_devices(device)["name"]
        except (ValueError, TypeError):
            # Treat as a name substring; find the first matching input device.
            matches = [
                (i, d) for i, d in list_input_devices()
                if args.device.lower() in d["name"].lower()
            ]
            if not matches:
                print(f"No input device matching '{args.device}' found.")
                print("Run without --device to pick interactively.")
                sys.exit(1)
            device, dev_info = matches[0]
            device_name = dev_info["name"]
        print(f"\nUsing microphone: {device_name}")
    else:
        device, device_name = select_microphone()

    print(f"Will record {len(indices)} paragraph(s).")
    print("Speak clearly at a normal pace.")
    print("Press Enter to stop each recording.\n")

    try:
        run_session(indices, paragraphs, manifest, device=device, device_name=device_name)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")

    manifest = load_manifest()
    done = len(manifest.get("recordings", []))
    total = len(paragraphs)
    print(f"\nDone. {done}/{total} paragraphs recorded.")
    if done < total:
        print("Run again (without flags) to record the remaining ones.")
        print("Run with --list to see which are missing.")


if __name__ == "__main__":
    main()
