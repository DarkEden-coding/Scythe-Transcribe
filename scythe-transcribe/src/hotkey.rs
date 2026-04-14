//! Global hold-to-record hotkey (rdev + cpal), clipboard paste, runtime icon updates.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Sample, SampleFormat};
use once_cell::sync::Lazy;
#[cfg(not(target_os = "macos"))]
use rdev::listen as listen_hotkey_events;
#[cfg(target_os = "macos")]
use rdev::listen_keyboard as listen_hotkey_events;
use rdev::{simulate, Event, EventType, Key};
use serde::Serialize;
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
const HOTKEY_CONFIG_RELOAD_INTERVAL: Duration = Duration::from_millis(500);
const HOTKEY_WATCHDOG_INTERVAL: Duration = Duration::from_millis(100);
pub const AUDIO_INPUT_BUILT_IN: &str = "__builtin_microphone__";
pub const AUDIO_INPUT_SYSTEM_DEFAULT: &str = "__system_default__";

#[derive(Clone, Debug, Serialize)]
pub struct AudioInputDeviceInfo {
    pub id: String,
    pub name: String,
    pub is_default: bool,
    pub is_builtin_candidate: bool,
}

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

fn combo_contains_token(combo_parts: &[String], tok: &str) -> bool {
    combo_parts.iter().any(|p| p == tok)
}

fn update_status(f: impl FnOnce(&mut Value)) {
    if let Ok(mut g) = STATUS.lock() {
        f(&mut g);
    }
}

fn is_builtin_input_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("built-in microphone")
        || lower.contains("built in microphone")
        || lower.contains("macbook pro microphone")
        || lower.contains("macbook air microphone")
        || lower.contains("imac microphone")
}

pub fn audio_input_devices() -> Vec<AudioInputDeviceInfo> {
    let host = cpal::default_host();
    let default_name = host.default_input_device().and_then(|d| d.name().ok());
    let mut devices = Vec::new();
    if let Ok(input_devices) = host.input_devices() {
        for device in input_devices {
            let Ok(name) = device.name() else {
                continue;
            };
            devices.push(AudioInputDeviceInfo {
                id: name.clone(),
                is_default: default_name.as_deref() == Some(name.as_str()),
                is_builtin_candidate: is_builtin_input_name(&name),
                name,
            });
        }
    }
    devices
}

fn find_named_input_device(host: &cpal::Host, name: &str) -> Option<Device> {
    let input_devices = host.input_devices().ok()?;
    input_devices
        .filter_map(|device| {
            let device_name = device.name().ok()?;
            Some((device_name, device))
        })
        .find_map(|(device_name, device)| (device_name == name).then_some(device))
}

fn find_builtin_input_device(host: &cpal::Host) -> Option<Device> {
    let input_devices = host.input_devices().ok()?;
    input_devices
        .filter_map(|device| {
            let device_name = device.name().ok()?;
            Some((device_name, device))
        })
        .find_map(|(device_name, device)| is_builtin_input_name(&device_name).then_some(device))
}

fn select_input_device(host: &cpal::Host, configured: &str) -> Option<Device> {
    let configured = configured.trim();
    if configured == AUDIO_INPUT_BUILT_IN {
        return find_builtin_input_device(host).or_else(|| host.default_input_device());
    }
    if configured.is_empty() || configured == AUDIO_INPUT_SYSTEM_DEFAULT {
        return host.default_input_device();
    }
    find_named_input_device(host, configured).or_else(|| host.default_input_device())
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
    stop_tx: mpsc::Sender<()>,
    join: Option<thread::JoinHandle<()>>,
    samples: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
}

impl AudioCapture {
    fn start() -> Result<Self, String> {
        let prefs = load_preferences();
        let samples = Arc::new(Mutex::new(Vec::<f32>::new()));
        let samples_thread = samples.clone();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<u32, String>>(1);
        let (stop_tx, stop_rx) = mpsc::channel::<()>();

        let join = thread::spawn(move || {
            let result = (|| {
                let host = cpal::default_host();
                let device = select_input_device(&host, &prefs.audio_input_device)
                    .ok_or_else(|| "no input device".to_string())?;
                let cfg = device.default_input_config().map_err(|e| e.to_string())?;
                let sample_rate = cfg.sample_rate().0;
                let stream_config = cfg.clone().into();
                let stream = build_input_stream(
                    &device,
                    &stream_config,
                    cfg.sample_format(),
                    samples_thread,
                )?;
                stream.play().map_err(|e| e.to_string())?;
                Ok((sample_rate, stream))
            })();

            match result {
                Ok((sample_rate, stream)) => {
                    let _ = ready_tx.send(Ok(sample_rate));
                    let _ = stop_rx.recv();
                    let _ = stream.pause();
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            }
        });

        let sample_rate = ready_rx
            .recv()
            .map_err(|_| "audio capture thread stopped".to_string())??;
        Ok(Self {
            stop_tx,
            join: Some(join),
            samples,
            sample_rate,
        })
    }

    fn stop_capture_thread(&mut self) {
        let _ = self.stop_tx.send(());
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }

    fn stop(mut self) -> (Vec<f32>, u32) {
        self.stop_capture_thread();
        let rate = self.sample_rate;
        let v = self.samples.lock().map(|g| g.clone()).unwrap_or_default();
        (v, rate)
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        self.stop_capture_thread();
    }
}

fn build_input_stream(
    device: &Device,
    cfg: &cpal::StreamConfig,
    sample_format: SampleFormat,
    samples: Arc<Mutex<Vec<f32>>>,
) -> Result<cpal::Stream, String> {
    match sample_format {
        SampleFormat::F32 => {
            let samples2 = samples.clone();
            device
                .build_input_stream(
                    cfg,
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
                .map_err(|e| e.to_string())
        }
        SampleFormat::I16 => {
            let samples2 = samples.clone();
            device
                .build_input_stream(
                    cfg,
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
                .map_err(|e| e.to_string())
        }
        SampleFormat::U16 => device
            .build_input_stream(
                cfg,
                move |data: &[u16], _| {
                    if let Ok(mut g) = samples.lock() {
                        for s in data {
                            g.push((*s as f32 / 32768.0) - 1.0);
                        }
                    }
                },
                |_e| {},
                None,
            )
            .map_err(|e| e.to_string()),
        _ => Err("unsupported sample format".to_string()),
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
    configured_combo: String,
    combo_parts: Vec<String>,
    config_loaded_at: Instant,
    counts: HashMap<String, i32>,
    prev_active: bool,
    recording: Option<AudioCapture>,
    event_count: u64,
}

impl HotkeyState {
    fn new_from_preferences() -> Self {
        let configured_combo = load_preferences().hotkey_toggle_recording;
        let combo_parts = parse_hotkey_combo(&configured_combo);
        Self {
            configured_combo,
            combo_parts,
            config_loaded_at: Instant::now(),
            counts: HashMap::new(),
            prev_active: false,
            recording: None,
            event_count: 0,
        }
    }

    fn refresh_config_if_due(&mut self, now: Instant) {
        if now.saturating_duration_since(self.config_loaded_at) < HOTKEY_CONFIG_RELOAD_INTERVAL {
            return;
        }
        self.config_loaded_at = now;

        if self.recording.is_some() {
            return;
        }

        let configured_combo = load_preferences().hotkey_toggle_recording;
        if configured_combo == self.configured_combo {
            return;
        }

        self.configured_combo = configured_combo;
        self.combo_parts = parse_hotkey_combo(&self.configured_combo);
        self.counts.clear();
        self.prev_active = false;
        update_status(|s| {
            s["configured_combo"] = json!(self.configured_combo);
            s["combo_parts"] = json!(self.combo_parts);
            s["pressed_tokens"] = json!(Vec::<String>::new());
            s["combo_active"] = json!(false);
        });
    }

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
                    self.counts.insert(t.clone(), 1);
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

fn finish_recording(
    st: &mut HotkeyState,
    now: f64,
    transcribing: &Arc<AtomicBool>,
    suppress_until: &Arc<Mutex<Instant>>,
    clear_pressed_state: bool,
) {
    let cap = st.recording.take();
    st.prev_active = false;
    if clear_pressed_state {
        st.counts.clear();
        update_status(|s| {
            s["pressed_tokens"] = json!(Vec::<String>::new());
            s["combo_active"] = json!(false);
        });
    }
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

fn hotkey_on_press(st: &mut HotkeyState, tok: Option<String>, configured: &str, now: f64) {
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
    tok: Option<String>,
    configured: &str,
    now: f64,
    transcribing: &Arc<AtomicBool>,
    suppress_until: &Arc<Mutex<Instant>>,
) {
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

    st.prev_active = combo_active;

    if st.recording.is_none() || combo_active {
        return;
    }

    finish_recording(st, now, transcribing, suppress_until, false);
}

fn run_hotkey_loop() {
    let transcribing = Arc::new(AtomicBool::new(false));
    let suppress_until = Arc::new(Mutex::new(Instant::now()));

    let state = Arc::new(Mutex::new(HotkeyState::new_from_preferences()));

    set_capture_state("idle");
    runtime_icon::set_icon_state(ICON_IDLE);

    let transcribing_cb = transcribing.clone();
    let suppress_cb = suppress_until.clone();
    let state_watchdog = state.clone();
    let transcribing_watchdog = transcribing.clone();
    let suppress_watchdog = suppress_until.clone();

    thread::spawn(move || loop {
        thread::sleep(HOTKEY_WATCHDOG_INTERVAL);
        let Ok(mut st) = state_watchdog.lock() else {
            continue;
        };
        if st.recording.is_none() {
            continue;
        }
        match crate::macos::hotkey_combo_physically_pressed(&st.combo_parts) {
            Some(true) => continue,
            None if combo_requirements_met(&st.counts, &st.combo_parts) => continue,
            Some(false) | None => {}
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        finish_recording(
            &mut st,
            now,
            &transcribing_watchdog,
            &suppress_watchdog,
            true,
        );
    });

    let callback = move |event: Event| {
        let (ev_name, key_opt) = match event.event_type {
            EventType::KeyPress(k) => ("press", Some(k)),
            EventType::KeyRelease(k) => ("release", Some(k)),
            _ => return,
        };
        let tok = key_opt.and_then(key_token).map(str::to_string);

        let instant_now = Instant::now();
        let is_suppressed = instant_now < suppress_cb.lock().map(|g| *g).unwrap_or(instant_now);
        let is_transcribing = transcribing_cb.load(Ordering::SeqCst);

        let Ok(mut st) = state.lock() else {
            return;
        };
        st.refresh_config_if_due(instant_now);
        if st.combo_parts.is_empty() {
            return;
        }
        if !tok
            .as_deref()
            .is_some_and(|t| combo_contains_token(&st.combo_parts, t))
        {
            return;
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let configured = st.configured_combo.clone();

        if ev_name == "press" {
            if is_suppressed || is_transcribing {
                return;
            }
            hotkey_on_press(&mut st, tok, &configured, now);
            return;
        }

        if is_suppressed || is_transcribing {
            st.record_event("release", &tok, &configured, now);
            st.prev_active = combo_requirements_met(&st.counts, &st.combo_parts);
            return;
        }

        hotkey_on_release(
            &mut st,
            tok,
            &configured,
            now,
            &transcribing_cb,
            &suppress_cb,
        );
    };

    let _ = listen_hotkey_events(callback);
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
            obj.insert("microphone_authorization".to_string(), json!(mic.as_str()));
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
