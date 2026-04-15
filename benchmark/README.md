# Scythe-Transcribe Benchmark

Measures transcription accuracy (Word Error Rate) and latency against a set of
pre-recorded voice samples using whatever configuration is currently active in
the running app.

---

## How it works

1. **record_samples.py** — you read paragraphs aloud from `paragraphs.txt`
   while the script records your voice. Each recording is saved as a WAV file
   alongside its ground-truth text.

2. **run_benchmark.py** — sends each WAV to the running Scythe-Transcribe
   server, collects the transcription response, computes Word Error Rate (WER)
   against the ground truth, and prints a summary table. Results are also saved
   to a JSON file for later review.

---

## Prerequisites

**Python 3.10 or later**, then install the Python dependencies:

```
cd benchmark
pip install -r requirements.txt
```

The benchmark communicates with Scythe-Transcribe over HTTP. **The app must be
running before you call `run_benchmark.py`**, unless you use the `--auto` flag
which handles building and running the server for you (see Step 2).

---

## Important: code changes require a rebuild

The benchmark runs against the **compiled, running binary** — not the source
code. If you change anything in the Rust backend and want those changes
reflected in benchmark results, the new binary must be built and running before
you measure.

If you are iterating on the backend, **use `--auto`** (see Step 2) — it rebuilds
and restarts the server automatically on every run so you never accidentally
benchmark a stale binary.

If you prefer to manage the server yourself, rebuild manually first:

```
cargo build --release -p scythe-transcribe
```

Then restart the app before running the benchmark.

---

## Step 1 — record samples

```
python record_samples.py
```

The script will first ask you to select a microphone from a numbered list of
all input devices detected on your system. Press `0` or Enter to use the
system default.

It then works through each paragraph in `paragraphs.txt` one by one:

- The paragraph text is displayed on screen.
- Press **Enter** to start a 3-second countdown and begin recording.
- Read the paragraph aloud at a natural pace.
- Press **Enter** again to stop recording.
- You are shown the duration and asked to keep, redo, or skip.

Recordings are saved to `samples/recordings/` as `sample_NNN.wav`. A
`manifest.json` is written to the same folder recording the ground-truth text,
duration, word count, microphone used, and timestamp for each sample.

Progress is saved after every paragraph, so you can stop and resume at any time.

### Useful flags

| Flag | Effect |
|------|--------|
| `--list` | Show which paragraphs are recorded and which are missing, then exit |
| `--redo` | Re-record all paragraphs, including ones already done |
| `--index 1,3,5` | Record only the specified paragraphs (1-based) |
| `--device N` | Skip the picker and use sounddevice index N |
| `--device "Blue Yeti"` | Skip the picker and match by device name substring |

---

## Step 2 — run the benchmark

### Recommended: use `--auto`

```
python run_benchmark.py --auto
```

`--auto` takes care of everything in one command:

1. Rebuilds the backend (`cargo build --release -p scythe-transcribe`).
2. Starts a temporary server instance in the background (no tray, no frontend).
3. Waits for it to become ready.
4. Runs the full benchmark.
5. Stops the server when finished.

This ensures you are always measuring the current code. It also means you do
not need to have the app running beforehand, and it will not interfere with any
existing instance you already have open.

### Manual: start the app yourself

If you want to benchmark a specific running instance (for example, the packaged
app with its own settings):

1. Make sure Scythe-Transcribe is already running.
2. Run:
   ```
   python run_benchmark.py
   ```

In both cases the script reads configuration directly from the server
(`GET /api/preferences`), so whatever provider, model, and post-processing
settings are active at that moment are what gets benchmarked. Each WAV is
posted to `POST /api/transcribe` exactly as the frontend would send it.

### All flags

| Flag | Effect |
|------|--------|
| `--auto` | Rebuild, start a temporary server, benchmark, then stop it |
| `--server URL` | Connect to a different server address (default: `http://127.0.0.1:8765`) |
| `--no-postprocess` | Disable post-processing for this run even if it is enabled in preferences |

### Output

A results table is printed to the terminal:

| Column | Description |
|--------|-------------|
| `#` | Paragraph number |
| `Ref words` | Number of words in the ground-truth reference |
| `Audio` | Duration of the recorded clip |
| `WER` | Word Error Rate — green < 5%, yellow < 15%, red ≥ 15% |
| `S/D/I` | Substitutions / Deletions / Insertions that make up the WER |
| `ASR (ms)` | Time taken by the speech recognition API call |
| `PP (ms)` | Post-processing latency (shown only when post-processing is on) |
| `Total (ms)` | Full server-side pipeline time |
| `Speed` | Audio duration ÷ ASR time, e.g. `24x` means 24× faster than real-time |

Aggregate averages and the active configuration are shown below the table.

A JSON file is written to `results/benchmark_YYYYMMDD_HHMMSS.json` containing
every result in full detail, including the raw server responses and ASR metadata.

---

## File layout

```
benchmark/
├── paragraphs.txt              # 15 reference paragraphs (varied length and content)
├── record_samples.py           # Interactive recording script
├── run_benchmark.py            # Benchmark runner
├── requirements.txt            # Python dependencies
├── samples/
│   └── recordings/
│       ├── manifest.json       # Created by record_samples.py
│       ├── sample_001.wav      # Created by record_samples.py
│       └── ...
└── results/
    └── benchmark_YYYYMMDD_HHMMSS.json   # Created by run_benchmark.py
```

WAV files, `manifest.json`, and result files are excluded from version control
via `.gitignore`.
