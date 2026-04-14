//! Preferences, API keys, caches, and transcription history on disk.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde_json::Value;

use crate::models::AppPreferences;

const APP_NAME: &str = "Scythe-Transcribe";
const MAX_TRANSCRIPTION_HISTORY: usize = 5000;

static DOTENV: OnceLock<()> = OnceLock::new();
static HISTORY_LOCK: Mutex<()> = Mutex::new(());

fn ensure_dotenv() {
    DOTENV.get_or_init(|| {
        let _ = dotenvy::dotenv();
    });
}

/// Match Python `platformdirs.user_config_dir` layout:
/// - Windows: `%LOCALAPPDATA%\\Scythe-Transcribe`
/// - macOS/Linux: config dir + `Scythe-Transcribe` (Linux `~/.config`, macOS Application Support)
#[must_use]
pub fn app_config_dir() -> PathBuf {
    if let Ok(p) = std::env::var("SCYTHE_CONFIG_DIR") {
        return PathBuf::from(p);
    }
    #[cfg(windows)]
    {
        dirs::data_local_dir().expect("LOCALAPPDATA").join(APP_NAME)
    }
    #[cfg(not(windows))]
    {
        dirs::config_dir().expect("config directory").join(APP_NAME)
    }
}

fn ensure_dir(path: &Path) -> std::io::Result<()> {
    fs::create_dir_all(path)
}

#[must_use]
pub fn diagnostics_log_path() -> PathBuf {
    app_config_dir().join("scythe-transcribe.log")
}

fn prefs_path() -> PathBuf {
    app_config_dir().join("preferences.json")
}

fn transcription_history_path() -> PathBuf {
    app_config_dir().join("transcription_history.json")
}

fn api_keys_path() -> PathBuf {
    app_config_dir().join("api_keys.json")
}

#[must_use]
pub fn openrouter_models_cache_path() -> PathBuf {
    app_config_dir().join("openrouter_models_cache.json")
}

#[must_use]
pub fn instance_lock_path() -> PathBuf {
    app_config_dir().join("instance.lock")
}

#[must_use]
pub fn load_preferences() -> AppPreferences {
    let path = prefs_path();
    if !path.is_file() {
        return AppPreferences::default();
    }
    let Ok(text) = fs::read_to_string(&path) else {
        return AppPreferences::default();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

pub fn save_preferences(prefs: &AppPreferences) -> anyhow::Result<()> {
    ensure_dir(&app_config_dir())?;
    let path = prefs_path();
    let text = serde_json::to_string_pretty(prefs)?;
    fs::write(path, text)?;
    Ok(())
}

fn load_api_keys_file() -> HashMap<String, String> {
    let path = api_keys_path();
    if !path.is_file() {
        return HashMap::new();
    }
    let Ok(text) = fs::read_to_string(&path) else {
        return HashMap::new();
    };
    let Ok(v) = serde_json::from_str::<HashMap<String, String>>(&text) else {
        return HashMap::new();
    };
    v
}

fn save_api_keys_file(keys: &HashMap<String, String>) -> anyhow::Result<()> {
    ensure_dir(&app_config_dir())?;
    fs::write(api_keys_path(), serde_json::to_string_pretty(keys)?)?;
    Ok(())
}

#[must_use]
pub fn get_groq_api_key() -> String {
    let data = load_api_keys_file();
    if let Some(k) = data.get("groq") {
        let s = k.trim();
        if !s.is_empty() {
            return s.to_string();
        }
    }
    ensure_dotenv();
    std::env::var("GROQ_API_KEY")
        .unwrap_or_default()
        .trim()
        .to_string()
}

#[must_use]
pub fn get_openrouter_api_key() -> String {
    let data = load_api_keys_file();
    if let Some(k) = data.get("openrouter") {
        let s = k.trim();
        if !s.is_empty() {
            return s.to_string();
        }
    }
    ensure_dotenv();
    std::env::var("OPENROUTER_API_KEY")
        .unwrap_or_default()
        .trim()
        .to_string()
}

pub fn set_groq_api_key(key: &str) -> anyhow::Result<()> {
    let mut data = load_api_keys_file();
    data.insert("groq".to_string(), key.to_string());
    save_api_keys_file(&data)
}

pub fn set_openrouter_api_key(key: &str) -> anyhow::Result<()> {
    let mut data = load_api_keys_file();
    data.insert("openrouter".to_string(), key.to_string());
    save_api_keys_file(&data)
}

#[must_use]
pub fn load_json_cache(path: &Path) -> Option<Value> {
    if !path.is_file() {
        return None;
    }
    let Ok(text) = fs::read_to_string(path) else {
        return None;
    };
    serde_json::from_str(&text).ok()
}

pub fn save_json_cache(path: &Path, data: &Value) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(data)?)?;
    Ok(())
}

fn load_transcription_history_raw(path: &Path) -> Vec<Value> {
    if !path.is_file() {
        return Vec::new();
    }
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_str::<Vec<Value>>(&text) else {
        return Vec::new();
    };
    v
}

#[must_use]
pub fn load_transcription_history() -> Vec<Value> {
    let path = transcription_history_path();
    let _g = HISTORY_LOCK.lock().ok();
    load_transcription_history_raw(&path)
}

pub fn append_transcription_history(entry: Value) -> anyhow::Result<()> {
    let path = transcription_history_path();
    let _g = HISTORY_LOCK.lock().expect("history lock");
    ensure_dir(&app_config_dir())?;
    let mut entries = load_transcription_history_raw(&path);
    entries.insert(0, entry);
    if entries.len() > MAX_TRANSCRIPTION_HISTORY {
        entries.truncate(MAX_TRANSCRIPTION_HISTORY);
    }
    fs::write(path, serde_json::to_string_pretty(&entries)?)?;
    Ok(())
}

pub fn patch_transcription_history_entry(entry_id: &str, patch: &Value) -> anyhow::Result<bool> {
    if entry_id.is_empty() {
        return Ok(false);
    }
    let path = transcription_history_path();
    let _g = HISTORY_LOCK.lock().expect("history lock");
    let mut entries = load_transcription_history_raw(&path);
    for row in &mut entries {
        let Some(id) = row.get("id").and_then(|v| v.as_str()) else {
            continue;
        };
        if id != entry_id {
            continue;
        }
        if let Some(obj) = row.as_object_mut() {
            if let Value::Object(p) = patch.clone() {
                for (k, v) in p {
                    obj.insert(k, v);
                }
            }
            fs::write(path, serde_json::to_string_pretty(&entries)?)?;
            return Ok(true);
        }
    }
    Ok(false)
}
