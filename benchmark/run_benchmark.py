#!/usr/bin/env python3
"""
Run ASR benchmark against the running Scythe-Transcribe server.

Reads every recorded sample from samples/recordings/manifest.json, posts each
WAV to the running server using its current configuration, computes WER and
latency metrics, prints a summary table, and saves detailed JSON results.

Usage:
    python run_benchmark.py
    python run_benchmark.py --server http://127.0.0.1:8765
    python run_benchmark.py --no-postprocess   # skip post-processing step

Scythe-Transcribe must be running before you start this script.

Requirements: pip install -r requirements.txt
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ── dependency check ──────────────────────────────────────────────────────────
try:
    import requests
    from rich.console import Console
    from rich.table import Table
    from rich import box
except ImportError as _err:
    print(f"Missing dependency: {_err}")
    print("Run:  pip install -r requirements.txt")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
SAMPLES_DIR = SCRIPT_DIR / "samples" / "recordings"
MANIFEST_FILE = SAMPLES_DIR / "manifest.json"
RESULTS_DIR = SCRIPT_DIR / "results"

DEFAULT_SERVER = "http://127.0.0.1:8765"

console = Console()


# ── WER ───────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation (keep apostrophes), split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def _edit_distance_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """
    Compute (substitutions, deletions, insertions) via DP + traceback.
    Standard Levenshtein edit distance, operation-level breakdown.
    """
    n, m = len(ref), len(hyp)
    # d[i][j] = minimum edit distance between ref[:i] and hyp[:j]
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j - 1], d[i - 1][j], d[i][j - 1])

    # Traceback to count operation types.
    S = D = I = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            S += 1
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return S, D, I


def word_error_rate(reference: str, hypothesis: str) -> dict:
    """
    Compute WER and return a breakdown dict.

    WER = (S + D + I) / N
        S = substitutions, D = deletions, I = insertions, N = reference words.
    """
    ref_words = _normalize(reference)
    hyp_words = _normalize(hypothesis)
    n = len(ref_words)
    if n == 0:
        return {"wer": 0.0, "S": 0, "D": 0, "I": 0, "N": 0}
    S, D, I = _edit_distance_ops(ref_words, hyp_words)
    return {
        "wer": round((S + D + I) / n, 4),
        "S": S,
        "D": D,
        "I": I,
        "N": n,
    }


# ── API helpers ───────────────────────────────────────────────────────────────

def fetch_preferences(server: str) -> dict:
    r = requests.get(f"{server}/api/preferences", timeout=10)
    r.raise_for_status()
    return r.json()


def build_meta(prefs: dict, *, postprocess: bool) -> dict:
    """
    Build the TranscribeJob meta payload that mirrors what the frontend sends.
    'postprocess' gates whether post-processing is requested regardless of
    the preferences value — useful for --no-postprocess benchmarking.
    """
    return {
        "transcription_provider":            prefs.get("transcription_provider", "groq"),
        "transcription_model_groq":          prefs.get("transcription_model_groq", "whisper-large-v3-turbo"),
        "transcription_model_openrouter":    prefs.get("transcription_model_openrouter", ""),
        "openrouter_transcription_instruction": prefs.get("openrouter_transcription_instruction", ""),
        "keyword_replacement_spec":          prefs.get("keyword_replacement_spec", ""),
        "postprocess_enabled":               postprocess and prefs.get("postprocess_enabled", False),
        "postprocess_prompt":                prefs.get("postprocess_prompt", ""),
        "postprocess_provider":              prefs.get("postprocess_provider", ""),
        "postprocess_model":                 prefs.get("postprocess_model", ""),
        "postprocess_groq_reasoning_effort":        prefs.get("postprocess_groq_reasoning_effort", ""),
        "postprocess_openrouter_reasoning_effort":  prefs.get("postprocess_openrouter_reasoning_effort", ""),
    }


def call_transcribe(server: str, wav_path: Path, meta: dict) -> dict:
    """POST multipart form to /api/transcribe; return parsed JSON."""
    wav_bytes = wav_path.read_bytes()
    files = {
        "meta":  (None,           json.dumps(meta), "application/json"),
        "audio": (wav_path.name,  wav_bytes,        "audio/wav"),
    }
    r = requests.post(f"{server}/api/transcribe", files=files, timeout=180)
    r.raise_for_status()
    return r.json()


# ── benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(server: str, *, postprocess: bool) -> dict:
    # Verify server is reachable.
    try:
        requests.get(f"{server}/api/health", timeout=5).raise_for_status()
    except requests.ConnectionError:
        console.print(f"[yellow]Warning: Scythe-Transcribe is not running (could not connect to {server}).[/yellow]")
        console.print("Start the app first, then re-run the benchmark.")
        sys.exit(1)
    except requests.Timeout:
        console.print(f"[yellow]Warning: connection to {server} timed out.[/yellow]")
        console.print("The app may be starting up — wait a moment and try again.")
        sys.exit(1)
    except requests.HTTPError as exc:
        console.print(f"[yellow]Warning: server returned an unexpected status ({exc.response.status_code}).[/yellow]")
        sys.exit(1)

    # Load manifest.
    if not MANIFEST_FILE.exists():
        console.print(f"[red]No manifest found at {MANIFEST_FILE}[/red]")
        console.print("Run  python record_samples.py  first.")
        sys.exit(1)

    with open(MANIFEST_FILE, encoding="utf-8") as f:
        manifest = json.load(f)

    recordings = manifest.get("recordings", [])
    if not recordings:
        console.print("[red]Manifest is empty — no recordings to benchmark.[/red]")
        sys.exit(1)

    # Fetch current configuration from the running server.
    try:
        prefs = fetch_preferences(server)
    except Exception as exc:
        console.print(f"[red]Failed to fetch preferences: {exc}[/red]")
        sys.exit(1)

    meta = build_meta(prefs, postprocess=postprocess)

    # Header.
    provider = meta["transcription_provider"]
    model = (
        meta["transcription_model_groq"]
        if provider == "groq"
        else meta["transcription_model_openrouter"]
    )
    console.print()
    console.rule("[bold]Scythe-Transcribe Benchmark[/bold]")
    console.print(f"  Server:       {server}")
    console.print(f"  Provider:     {provider}")
    console.print(f"  ASR model:    {model or '(default)'}")
    pp_on = meta["postprocess_enabled"]
    console.print(f"  Post-process: {'yes — ' + meta['postprocess_model'] if pp_on else 'no'}")
    console.print(f"  Samples:      {len(recordings)}")
    console.print()

    results: list[dict] = []
    run_at = datetime.now().isoformat()

    for pos, rec in enumerate(recordings):
        wav_file = SAMPLES_DIR / rec["wav_file"]
        label = f"{rec['id']}  ({rec['word_count']}w, {rec['duration_sec']:.1f}s)"

        if not wav_file.exists():
            console.print(f"  [{pos+1}/{len(recordings)}] [yellow]{label} — WAV missing, skipped[/yellow]")
            results.append({
                "sample_id": rec["id"],
                "index": rec["index"],
                "error": "WAV file not found",
                "ground_truth": rec["ground_truth"],
            })
            continue

        console.print(f"  [{pos+1}/{len(recordings)}] {label}... ", end="", highlight=False)

        try:
            t0 = time.perf_counter()
            resp = call_transcribe(server, wav_file, meta)
            wall_ms = (time.perf_counter() - t0) * 1000

            transcript   = resp.get("transcript", "")
            processed    = resp.get("processed")
            # Evaluate WER against post-processed output when available.
            eval_text    = processed if (pp_on and processed) else transcript
            ground_truth = rec["ground_truth"]
            wer          = word_error_rate(ground_truth, eval_text)

            audio_dur    = resp.get("audio_duration_sec") or rec.get("duration_sec", 0)
            asr_ms       = resp.get("transcribe_ms", wall_ms)
            pp_ms        = resp.get("postprocess_ms")
            total_ms     = resp.get("total_ms", wall_ms)
            speed_factor = audio_dur / (asr_ms / 1000) if audio_dur > 0 and asr_ms > 0 else None
            rtf          = (asr_ms / 1000) / audio_dur if audio_dur > 0 and asr_ms > 0 else None

            result = {
                "sample_id":         rec["id"],
                "index":             rec["index"],
                "ground_truth":      ground_truth,
                "transcript":        transcript,
                "processed":         processed,
                "eval_text":         eval_text,
                "wer":               wer,
                "audio_duration_sec": audio_dur,
                "asr_ms":            round(asr_ms, 1),
                "postprocess_ms":    round(pp_ms, 1) if pp_ms is not None else None,
                "total_ms":          round(total_ms, 1),
                "wall_ms":           round(wall_ms, 1),
                "speed_factor":      round(speed_factor, 2) if speed_factor is not None else None,
                "rtf":               round(rtf, 4)          if rtf          is not None else None,
                "silence_detected":  resp.get("silence_detected", False),
                "asr_metadata":      resp.get("asr_metadata"),
                "server_response":   resp,
            }
            results.append(result)

            wer_pct   = f"{wer['wer'] * 100:.1f}%"
            speed_str = f"  {speed_factor:.1f}x" if speed_factor else ""
            console.print(f"WER={wer_pct}  ASR={asr_ms:.0f}ms{speed_str}")

        except requests.HTTPError as exc:
            msg = f"HTTP {exc.response.status_code}: {exc.response.text[:120]}"
            console.print(f"[red]ERROR — {msg}[/red]")
            results.append({
                "sample_id":    rec["id"],
                "index":        rec["index"],
                "error":        msg,
                "ground_truth": rec["ground_truth"],
            })
        except Exception as exc:
            console.print(f"[red]ERROR — {exc}[/red]")
            results.append({
                "sample_id":    rec["id"],
                "index":        rec["index"],
                "error":        str(exc),
                "ground_truth": rec["ground_truth"],
            })

    return {
        "run_at":      run_at,
        "server":      server,
        "preferences": prefs,
        "meta_used":   meta,
        "results":     results,
    }


# ── terminal table ────────────────────────────────────────────────────────────

def print_summary_table(data: dict) -> None:
    ok     = [r for r in data["results"] if "error" not in r]
    errors = [r for r in data["results"] if "error" in r]

    if not ok:
        console.print("[red]No successful results to display.[/red]")
        if errors:
            console.print("\nErrors:")
            for e in errors:
                console.print(f"  {e['sample_id']}: {e['error']}")
        return

    meta   = data.get("meta_used", {})
    pp_on  = meta.get("postprocess_enabled", False)

    table = Table(
        title="\nBenchmark Results",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        header_style="bold",
    )
    table.add_column("#",           justify="right", width=4,  style="dim")
    table.add_column("Sample",      justify="left",  width=12)
    table.add_column("Ref\nwords",  justify="right", width=6)
    table.add_column("Audio",       justify="right", width=7)
    table.add_column("WER",         justify="right", width=7)
    table.add_column("S/D/I",       justify="center", width=9)
    table.add_column("ASR\n(ms)",   justify="right", width=8)
    if pp_on:
        table.add_column("PP\n(ms)", justify="right", width=8)
    table.add_column("Total\n(ms)", justify="right", width=9)
    table.add_column("Speed",       justify="right", width=7)

    wers, asr_mss, total_mss, speeds = [], [], [], []

    for r in ok:
        wer   = r["wer"]
        w_pct = f"{wer['wer'] * 100:.1f}%"
        wstyle = "green" if wer["wer"] < 0.05 else ("yellow" if wer["wer"] < 0.15 else "red")

        audio = f"{r['audio_duration_sec']:.1f}s" if r.get("audio_duration_sec") else "-"
        speed = f"{r['speed_factor']:.1f}x"        if r.get("speed_factor")        else "-"
        sdi   = f"{wer['S']}/{wer['D']}/{wer['I']}"

        row = [
            str(r["index"] + 1),
            r["sample_id"],
            str(wer["N"]),
            audio,
            f"[{wstyle}]{w_pct}[/{wstyle}]",
            sdi,
            f"{r['asr_ms']:.0f}",
        ]
        if pp_on:
            pp_ms = r.get("postprocess_ms")
            row.append(f"{pp_ms:.0f}" if pp_ms is not None else "-")
        row += [f"{r['total_ms']:.0f}", speed]
        table.add_row(*row)

        wers.append(wer["wer"])
        asr_mss.append(r["asr_ms"])
        total_mss.append(r["total_ms"])
        if r.get("speed_factor"):
            speeds.append(r["speed_factor"])

    console.print(table)

    # Aggregate stats.
    avg_wer   = sum(wers)     / len(wers)
    avg_asr   = sum(asr_mss)  / len(asr_mss)
    avg_total = sum(total_mss) / len(total_mss)
    avg_speed = sum(speeds)    / len(speeds) if speeds else None

    wer_style = "green" if avg_wer < 0.05 else ("yellow" if avg_wer < 0.15 else "red")

    console.print("[bold]Aggregate[/bold]")
    console.print(f"  Samples run:      {len(ok)}  ({len(errors)} error(s))")
    console.print(f"  Avg WER:          [{wer_style}]{avg_wer * 100:.2f}%[/{wer_style}]")
    console.print(f"  Median WER:       {sorted(wers)[len(wers)//2] * 100:.2f}%")
    console.print(f"  Avg ASR latency:  {avg_asr:.0f} ms")
    if pp_on:
        pp_times = [r["postprocess_ms"] for r in ok if r.get("postprocess_ms") is not None]
        if pp_times:
            console.print(f"  Avg PP latency:   {sum(pp_times)/len(pp_times):.0f} ms")
    console.print(f"  Avg total:        {avg_total:.0f} ms")
    if avg_speed:
        console.print(f"  Avg speed factor: {avg_speed:.1f}x real-time")

    meta   = data["meta_used"]
    prefs  = data.get("preferences", {})
    provider = meta["transcription_provider"]
    model = (
        meta["transcription_model_groq"]
        if provider == "groq"
        else meta["transcription_model_openrouter"]
    )
    console.print()
    console.print("[bold]Configuration[/bold]")
    console.print(f"  Provider:     {provider}")
    console.print(f"  ASR model:    {model or '(default)'}")
    pp_on2 = meta.get("postprocess_enabled", False)
    console.print(f"  Post-process: {'yes — ' + meta.get('postprocess_model', '') if pp_on2 else 'no'}")
    console.print(f"  Run at:       {data['run_at']}")

    if errors:
        console.print(f"\n[red]Errors ({len(errors)}):[/red]")
        for e in errors:
            console.print(f"  {e['sample_id']}: {e['error']}")


# ── results persistence ───────────────────────────────────────────────────────

def save_results(data: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"benchmark_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ASR benchmark against Scythe-Transcribe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--server", default=DEFAULT_SERVER,
        help=f"Server base URL (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--no-postprocess", action="store_true",
        help="Disable post-processing even if enabled in preferences",
    )
    args = parser.parse_args()

    data     = run_benchmark(args.server, postprocess=not args.no_postprocess)
    print_summary_table(data)
    out_path = save_results(data)
    console.print(f"\n[dim]Results saved → {out_path}[/dim]\n")


if __name__ == "__main__":
    main()
