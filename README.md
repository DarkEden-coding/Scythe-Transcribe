# Scythe-Transcribe

Local speech-to-text using Groq and/or OpenRouter, with a browser UI served from a small Rust app (tray icon on desktop, optional server-only mode).

## Prerequisites

- **Rust** (stable), with `cargo` on your `PATH`
- **Node.js** and **npm** (only needed to build the web UI)

## Build the web UI

The server looks for the compiled SPA at `frontend/dist` (or at the path in `SCYTHE_STATIC_ROOT`).

From the repository root:

```sh
cargo xtask build-ui
```

That runs `npm ci` and `npm run build` in `frontend/`. You can do the same manually:

```sh
cd frontend
npm ci
npm run build
cd ..
```

## Build the Rust binary

Debug:

```sh
cargo build -p scythe-transcribe
```

Release:

```sh
cargo build --release -p scythe-transcribe
```

The `xtask` crate defines `cargo xtask dist`, which builds the UI, runs workspace tests, then builds `scythe-transcribe` in release mode.

## Run

After the UI is built (so `frontend/dist/index.html` exists), from the repository root:

```sh
cargo run -p scythe-transcribe
```

Or run the compiled executable, for example:

```sh
./target/release/scythe-transcribe
```

Open **[http://127.0.0.1:8765/](http://127.0.0.1:8765/)** in a browser. The app binds the HTTP API to that address and port.

### Environment

- `**GROQ_API_KEY`** / `**OPENROUTER_API_KEY**`: used for transcription when not set in the app settings (a `.env` file in the project directory is loaded automatically if present).
- `**SCYTHE_SERVER_ONLY=1**` (or `**SCYTHE_TRAY=0**`): run without the system tray; only the HTTP server runs.
- `**SCYTHE_STATIC_ROOT**`: override the directory that contains `index.html` for the SPA.
- `**SCYTHE_CONFIG_DIR**`: override where preferences and API key storage live.

### Frontend development (Vite)

To work on the React UI with hot reload while the API runs on port 8765, use two terminals:

1. Start the Rust server (with the UI already built once, or API-only behavior as needed).
2. In `frontend/`, run `npm run dev` (Vite defaults to port **5173** and proxies `/api` to `http://127.0.0.1:8765`).

On macOS, global hotkeys and input monitoring may require **Accessibility** (and related) permissions in System Settings.