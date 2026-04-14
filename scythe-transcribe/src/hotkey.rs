//! Global hold-to-record hotkey (rdev + cpal), clipboard paste, runtime icon updates.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat};
use once_cell::sync::Lazy;
use rdev::{listen, simulate, Event, EventType, Key};
use serde_json::{json, Value};

use crate::pipeline::{
    patch_history_timings, text_to_paste, transcribe_job_from_preferences, transcribe_wav_bytes,
};
use crate::runtime_icon::{self, ICON_IDLE, ICON_PROCESSING, ICON_RECORDING};
use crate::settings_store::load_preferences;

static LISTENER_STARTED: OnceLock<()> = OnceLock::new();
static STATUS: Lazy<Mutex<Value>> = Lazy::new(|| {
    Mutex::new(
        serde_json::from_str(include_str!("hotkey_status_default.json"))
            .expect("hotkey_status_default.json"),
    )
});

static HTTP: Lazy<reqwest::Client> = Lazy::new(reqwest::Client::new);
static TOKIO: Lazy<tokio::runtime::Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime for hotkey")
});

const MODIFIERS: &[&str] = &["ctrl", "alt", "shift", "meta"];

fn normalize_part(p: &str) -> String {
    let p = p.trim().to_lowercase();
    match p.as_str() {
        "control" => "ctrl".to_string(),
        "option" => "alt".to_string(),
        "cmd" | "command" | "win" | "super" | "os" => "meta".to_string(),
        _ => p,
    }
}

fn parse_hotkey_combo(raw: &str) -> Vec<String> {
    raw.split('+')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(normalize_part)
        .collect()
}

fn key_token(key: Key) -> Option<&'static str> {
    match key {
        Key::ControlLeft | Key::ControlRight => Some("ctrl"),
        Key::ShiftLeft | Key::ShiftRight => Some("shift"),
        Key::Alt | Key::AltGr => Some("alt"),
        Key::MetaLeft | Key::MetaRight => Some("meta"),
        Key::Space => Some("space"),
        Key::F1 => Some("f1"),
        Key::F2 => Some("f2"),
        Key::F3 => Some("f3"),
        Key::F4 => Some("f4"),
        Key::F5 => Some("f5"),
        Key::F6 => Some("f6"),
        Key::F7 => Some("f7"),
        Key::F8 => Some("f8"),
        Key::F9 => Some("f9"),
        Key::F10 => Some("f10"),
        Key::F11 => Some("f11"),
        Key::F12 => Some("f12"),
        Key::KeyA => Some("a"),
        Key::KeyB => Some("b"),
        Key::KeyC => Some("c"),
        Key::KeyD => Some("d"),
        Key::KeyE => Some("e"),
        Key::KeyF => Some("f"),
        Key::KeyG => Some("g"),
        Key::KeyH => Some("h"),
        Key::KeyI => Some("i"),
        Key::KeyJ => Some("j"),
        Key::KeyK => Some("k"),
        Key::KeyL => Some("l"),
        Key::KeyM => Some("m"),
        Key::KeyN => Some("n"),
        Key::KeyO => Some("o"),
        Key::KeyP => Some("p"),
        Key::KeyQ => Some("q"),
        Key::KeyR => Some("r"),
        Key::KeyS => Some("s"),
        Key::KeyT => Some("t"),
        Key::KeyU => Some("u"),
        Key::KeyV => Some("v"),
        Key::KeyW => Some("w"),
        Key::KeyX => Some("x"),
        Key::KeyY => Some("y"),
        Key::KeyZ => Some("z"),
        Key::Num0 => Some("0"),
        Key::Num1 => Some("1"),
        Key::Num2 => Some("2"),
        Key::Num3 => Some("3"),
        Key::Num4 => Some("4"),
        Key::Num5 => Some("5"),
        Key::Num6 => Some("6"),
        Key::Num7 => Some("7"),
        Key::Num8 => Some("8"),
        Key::Num9 => Some("9"),
        Key::Unknown(_) => None,
        _ => None,
    }
}

fn pressed_tokens(counts: &HashMap<String, i32>) -> Vec<String> {
    let mut v: Vec<String> = counts
        .iter()
        .filter(|(_, c)| **c > 0)
        .map(|(k, _)| k.clone())
        .collect();
    v.sort();
    v
}

fn combo_requirements_met(counts: &HashMap<String, i32>, combo_parts: &[String]) -> bool {
    if combo_parts.is_empty() {
        return false;
    }
    combo_parts
        .iter()
        .all(|p| *counts.get(p).unwrap_or(&0) >= 1)
}

fn update_status(f: impl FnOnce(&mut Value)) {
    if let Ok(mut g) = STATUS.lock() {
        f(&mut g);
    }
}

fn set_capture_state(state: &str) {
    update_status(|s| {
        if s["capture_state"].as_str() == Some(state) {
            return;
        }
        s["capture_state"] = json!(state);
        s["capture_state_changed_at"] = json!(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0));
    });
    match state {
        "recording" => runtime_icon::set_icon_state(ICON_RECORDING),
        "processing" => runtime_icon::set_icon_state(ICON_PROCESSING),
        _ => runtime_icon::set_icon_state(ICON_IDLE),
    }
}

fn float32_mono_to_wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut cursor = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    if let Ok(mut writer) = hound::WavWriter::new(&mut cursor, spec) {
        for s in samples {
            let x = (*s as f64).clamp(-1.0, 1.0);
            let pcm = (x * 32767.0) as i16;
            let _ = writer.write_sample(pcm);
        }
        let _ = writer.finalize();
    }
    cursor.into_inner()
}

struct AudioCapture {
    _stream: cpal::Stream,
    samples: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
}

impl AudioCapture {
    fn start() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| "no input device".to_string())?;
        let cfg = device.default_input_config().map_err(|e| e.to_string())?;
        let sample_rate = cfg.sample_rate().0;
        let samples = Arc::new(Mutex::new(Vec::<f32>::new()));
        let samples2 = samples.clone();

        let stream = match cfg.sample_format() {
            SampleFormat::F32 => device
                .build_input_stream(
                    &cfg.into(),
                    move |data: &[f32], _| {
                        if let Ok(mut g) = samples2.lock() {
                            for s in data.iter().cloned() {
                                g.push(s.to_sample::<f32>());
                            }
                        }
                    },
                    |_e| {},
                    None,
                )
                .map_err(|e| e.to_string())?,
            SampleFormat::I16 => device
                .build_input_stream(
                    &cfg.into(),
                    move |data: &[i16], _| {
                        if let Ok(mut g) = samples2.lock() {
                            for s in data.iter().cloned() {
                                g.push(s.to_sample::<f32>());
                            }
                        }
                    },
                    |_e| {},
                    None,
                )
                .map_err(|e| e.to_string())?,
            SampleFormat::U16 => device
                .build_input_stream(
                    &cfg.into(),
                    move |data: &[u16], _| {
                        if let Ok(mut g) = samples2.lock() {
                            for s in data {
                                g.push((*s as f32 / 32768.0) - 1.0);
                            }
                        }
                    },
                    |_e| {},
                    None,
                )
                .map_err(|e| e.to_string())?,
            _ => return Err("unsupported sample format".to_string()),
        };
        stream.play().map_err(|e| e.to_string())?;
        Ok(Self {
            _stream: stream,
            samples,
            sample_rate,
        })
    }

    fn stop(self) -> (Vec<f32>, u32) {
        let rate = self.sample_rate;
        let v = self.samples.lock().map(|g| g.clone()).unwrap_or_default();
        (v, rate)
    }
}

fn paste_chord() {
    thread::sleep(Duration::from_millis(40));
    #[cfg(target_os = "macos")]
    {
        let _ = simulate(&EventType::KeyPress(Key::MetaLeft));
        let _ = simulate(&EventType::KeyPress(Key::KeyV));
        let _ = simulate(&EventType::KeyRelease(Key::KeyV));
        let _ = simulate(&EventType::KeyRelease(Key::MetaLeft));
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = simulate(&EventType::KeyPress(Key::ControlLeft));
        let _ = simulate(&EventType::KeyPress(Key::KeyV));
        let _ = simulate(&EventType::KeyRelease(Key::KeyV));
        let _ = simulate(&EventType::KeyRelease(Key::ControlLeft));
    }
}

fn paste_at_cursor(text: &str, suppress_until: &Arc<Mutex<Instant>>) {
    if text.trim().is_empty() {
        return;
    }
    let _ = arboard::Clipboard::new().and_then(|mut c| c.set_text(text));
    paste_chord();
    if let Ok(mut g) = suppress_until.lock() {
        *g = Instant::now() + Duration::from_millis(350);
    }
}

struct HotkeyState {
    combo_parts: Vec<String>,
    counts: HashMap<String, i32>,
    prev_active: bool,
    recording: Option<AudioCapture>,
    event_count: u64,
}

impl HotkeyState {
    fn record_event(
        &mut self,
        ev_name: &str,
        tok: &Option<String>,
        configured: &str,
        now: f64,
    ) -> bool {
        self.event_count += 1;
        let combo_active = if let Some(t) = tok {
            let is_mod = MODIFIERS.contains(&t.as_str());
            if is_mod {
                if ev_name == "press" {
                    *self.counts.entry(t.clone()).or_insert(0) += 1;
                } else if let Some(v) = self.counts.get_mut(t) {
                    *v -= 1;
                    if *v <= 0 {
                        self.counts.remove(t);
                    }
                }
            } else if ev_name == "press" {
                self.counts.insert(t.clone(), 1);
            } else {
                self.counts.remove(t);
            }
            combo_requirements_met(&self.counts, &self.combo_parts)
        } else {
            combo_requirements_met(&self.counts, &self.combo_parts)
        };

        update_status(|s| {
            s["configured_combo"] = json!(configured);
            s["combo_parts"] = json!(self.combo_parts);
            s["pressed_tokens"] = json!(pressed_tokens(&self.counts));
            s["combo_active"] = json!(combo_active);
            s["last_event"] = json!(ev_name);
            s["last_token"] = json!(tok);
            s["last_key"] = json!(format!("{tok:?}"));
            s["last_event_at"] = json!(now);
            s["event_count"] = json!(self.event_count);
            if combo_active {
                s["last_combo_matched_at"] = json!(now);
            }
        });
        combo_active
    }
}

fn transcribe_wav_on_thread(
    wav: Vec<u8>,
    transcribing: Arc<AtomicBool>,
    suppress_until: Arc<Mutex<Instant>>,
) {
    thread::spawn(move || {
        let prefs = load_preferences();
        let job = transcribe_job_from_preferences(&prefs);
        let http = HTTP.clone();
        let result = TOKIO.block_on(async { transcribe_wav_bytes(&http, &job, &wav).await });
        let t_after_pipeline = Instant::now();
        match result {
            Ok(res) => {
                let text = text_to_paste(&prefs, &res);
                let t_before_paste = Instant::now();
                paste_at_cursor(&text, &suppress_until);
                let t_after_paste = Instant::now();
                let post_api = t_before_paste
                    .saturating_duration_since(t_after_pipeline)
                    .as_secs_f64()
                    * 1000.0;
                let chord = t_after_paste
                    .saturating_duration_since(t_before_paste)
                    .as_secs_f64()
                    * 1000.0;
                let _ = patch_history_timings(&res.id, post_api, chord);
            }
            Err(e) => {
                update_status(|s| {
                    s["last_transcribe_error"] = json!(format!("status {:?}: {}", e.0, e.1));
                    s["last_transcribe_error_at"] = json!(std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs_f64())
                        .unwrap_or(0.0));
                });
            }
        }
        transcribing.store(false, Ordering::SeqCst);
        set_capture_state("idle");
        runtime_icon::set_icon_state(ICON_IDLE);
    });
}

fn hotkey_on_press(st: &mut HotkeyState, tok: Option<Key>, configured: &str, now: f64) {
    let tok = tok.and_then(key_token).map(str::to_string);

    if tok.is_none() {
        st.record_event("press", &None, configured, now);
        return;
    }
    if st.combo_parts.is_empty() {
        st.record_event("press", &tok, configured, now);
        return;
    }

    let tok_ref = tok.clone();
    let combo_active = st.record_event("press", &tok_ref, configured, now);

    if combo_active && !st.prev_active {
        match AudioCapture::start() {
            Ok(cap) => {
                st.recording = Some(cap);
                set_capture_state("recording");
                update_status(|s| {
                    s["last_recording_started_at"] = json!(now);
                });
            }
            Err(e) => {
                update_status(|s| {
                    s["last_stream_error"] = json!(e);
                    s["last_stream_error_at"] = json!(now);
                });
                set_capture_state("idle");
                runtime_icon::set_icon_state(ICON_IDLE);
                st.prev_active = false;
                return;
            }
        }
    }
    st.prev_active = combo_active;
}

fn hotkey_on_release(
    st: &mut HotkeyState,
    tok: Option<Key>,
    configured: &str,
    now: f64,
    transcribing: &Arc<AtomicBool>,
    suppress_until: &Arc<Mutex<Instant>>,
) {
    let tok = tok.and_then(key_token).map(str::to_string);

    if tok.is_none() {
        st.record_event("release", &None, configured, now);
        return;
    }
    if st.combo_parts.is_empty() {
        st.record_event("release", &tok, configured, now);
        return;
    }

    let tok_ref = tok.clone();
    let combo_active = st.record_event("release", &tok_ref, configured, now);

    let was_active = st.prev_active;
    st.prev_active = combo_active;

    if !was_active || combo_active {
        return;
    }

    let cap = st.recording.take();
    update_status(|s| {
        s["last_recording_stopped_at"] = json!(now);
    });
    let (samples, rate) = if let Some(c) = cap {
        c.stop()
    } else {
        (Vec::new(), 16_000)
    };
    let min_samps = (rate as usize * 12) / 100;
    if samples.len() >= min_samps
        && transcribing
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    {
        set_capture_state("processing");
        runtime_icon::set_icon_state(ICON_PROCESSING);
        let wav = float32_mono_to_wav_bytes(&samples, rate);
        let tr = transcribing.clone();
        let sup = suppress_until.clone();
        transcribe_wav_on_thread(wav, tr, sup);
    } else {
        set_capture_state("idle");
        runtime_icon::set_icon_state(ICON_IDLE);
    }
}

fn run_hotkey_loop() {
    let transcribing = Arc::new(AtomicBool::new(false));
    let suppress_until = Arc::new(Mutex::new(Instant::now()));

    let state = RefCell::new(HotkeyState {
        combo_parts: Vec::new(),
        counts: HashMap::new(),
        prev_active: false,
        recording: None,
        event_count: 0,
    });

    set_capture_state("idle");
    runtime_icon::set_icon_state(ICON_IDLE);

    let transcribing_cb = transcribing.clone();
    let suppress_cb = suppress_until.clone();

    let callback = move |event: Event| {
        if Instant::now() < suppress_cb.lock().map(|g| *g).unwrap_or(Instant::now()) {
            return;
        }
        if transcribing_cb.load(Ordering::SeqCst) {
            return;
        }

        let prefs = load_preferences();
        let configured = prefs.hotkey_toggle_recording.clone();
        let parts = parse_hotkey_combo(&configured);

        let mut st = state.borrow_mut();
        st.combo_parts = parts;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        let (ev_name, key_opt) = match event.event_type {
            EventType::KeyPress(k) => ("press", Some(k)),
            EventType::KeyRelease(k) => ("release", Some(k)),
            _ => return,
        };

        if ev_name == "press" {
            hotkey_on_press(&mut st, key_opt, &configured, now);
            return;
        }

        hotkey_on_release(
            &mut st,
            key_opt,
            &configured,
            now,
            &transcribing_cb,
            &suppress_cb,
        );
    };

    let _ = listen(callback);
}

fn manager_loop() {
    loop {
        update_status(|s| {
            s["state"] = json!("running");
        });
        run_hotkey_loop();
        update_status(|s| {
            s["state"] = json!("stopped");
        });
        set_capture_state("idle");
        runtime_icon::set_icon_state(ICON_IDLE);
        thread::sleep(Duration::from_secs(2));
    }
}

/// Start background listener (first call wins).
pub fn start_hotkey_listener() {
    LISTENER_STARTED.get_or_init(|| {
        thread::spawn(|| {
            manager_loop();
        });
    });
}

#[must_use]
pub fn get_hotkey_listener_status() -> Value {
    let prefs = load_preferences();
    let combo = prefs.hotkey_toggle_recording.clone();
    let parts = parse_hotkey_combo(&combo);
    update_status(|s| {
        s["configured_combo"] = json!(combo);
        s["combo_parts"] = json!(parts);
    });
    let mut v = STATUS.lock().expect("status").clone();
    if let Some(obj) = v.as_object_mut() {
        #[cfg(target_os = "macos")]
        {
            obj.insert(
                "accessibility_trusted".to_string(),
                json!(crate::macos::is_accessibility_trusted()),
            );
            obj.insert(
                "input_monitoring_trusted".to_string(),
                json!(crate::macos::is_input_monitoring_trusted()),
            );
            let mic = crate::macos::microphone_authorization();
            obj.insert(
                "microphone_authorized".to_string(),
                json!(mic.is_authorized()),
            );
            obj.insert(
                "microphone_authorization".to_string(),
                json!(mic.as_str()),
            );
        }
        #[cfg(not(target_os = "macos"))]
        {
            obj.insert("accessibility_trusted".to_string(), json!(true));
            obj.insert("input_monitoring_trusted".to_string(), json!(true));
            obj.insert("microphone_authorized".to_string(), json!(true));
            obj.insert("microphone_authorization".to_string(), json!("authorized"));
        }
    }
    v
}

#[must_use]
pub fn current_app_identity() -> Value {
    let exe = std::env::current_exe()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let mut app_bundle: Option<String> = None;
    if let Ok(p) = std::env::current_exe() {
        for path in
            std::iter::once(p.clone()).chain(p.ancestors().map(std::path::Path::to_path_buf))
        {
            if path.extension().is_some_and(|e| e == "app")
                && path.join("Contents/Info.plist").is_file()
            {
                app_bundle = Some(path.to_string_lossy().to_string());
                break;
            }
        }
    }
    json!({
        "pid": std::process::id(),
        "executable": exe,
        "app_bundle": app_bundle,
    })
}
