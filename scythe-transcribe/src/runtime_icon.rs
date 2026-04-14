//! Tray/menu-bar icon state (idle / recording / processing) and manual override.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Mutex;

use once_cell::sync::Lazy;
use serde::Serialize;

pub const ICON_IDLE: &str = "idle";
pub const ICON_RECORDING: &str = "recording";
pub const ICON_PROCESSING: &str = "processing";

static STATE: Lazy<Mutex<RuntimeIconInner>> = Lazy::new(|| {
    Mutex::new(RuntimeIconInner {
        base_state: ICON_IDLE.to_string(),
        override_state: None,
    })
});

static SUBSCRIBERS: Lazy<Mutex<Vec<Sender<String>>>> = Lazy::new(|| Mutex::new(Vec::new()));

struct RuntimeIconInner {
    base_state: String,
    override_state: Option<String>,
}

#[derive(Serialize)]
pub struct IconStatus {
    pub base_state: String,
    pub override_state: Option<String>,
    pub display_state: String,
}

fn display_state(g: &RuntimeIconInner) -> &str {
    g.override_state.as_deref().unwrap_or(g.base_state.as_str())
}

fn notify_display_state(state: &str) {
    let mut subscribers = SUBSCRIBERS.lock().expect("runtime icon subscribers");
    subscribers.retain(|tx| tx.send(state.to_string()).is_ok());
}

#[must_use]
pub fn subscribe_icon_changes() -> Receiver<String> {
    let (tx, rx) = mpsc::channel();
    let mut subscribers = SUBSCRIBERS.lock().expect("runtime icon subscribers");
    subscribers.push(tx);
    rx
}

#[must_use]
pub fn get_icon_status() -> IconStatus {
    let g = STATE.lock().expect("runtime icon");
    let display = display_state(&g);
    IconStatus {
        base_state: g.base_state.clone(),
        override_state: g.override_state.clone(),
        display_state: display.to_string(),
    }
}

pub fn set_icon_state(state: &str) {
    let mut g = STATE.lock().expect("runtime icon");
    if g.base_state.as_str() == state {
        return;
    }
    let old_display = display_state(&g).to_string();
    g.base_state = state.to_string();
    let new_display = display_state(&g).to_string();
    drop(g);
    if old_display != new_display {
        notify_display_state(&new_display);
    }
}

pub fn set_icon_override(state: Option<&str>) -> IconStatus {
    let mut g = STATE.lock().expect("runtime icon");
    let old_display = display_state(&g).to_string();
    g.override_state = state.map(str::to_string);
    let new_display = display_state(&g).to_string();
    drop(g);
    if old_display != new_display {
        notify_display_state(&new_display);
    }
    get_icon_status()
}

fn next_state(s: &str) -> &'static str {
    match s {
        ICON_IDLE => ICON_RECORDING,
        ICON_RECORDING => ICON_PROCESSING,
        _ => ICON_IDLE,
    }
}

#[must_use]
pub fn cycle_icon_override() -> IconStatus {
    let mut g = STATE.lock().expect("runtime icon");
    let current = g.override_state.as_deref().unwrap_or(g.base_state.as_str());
    let next = next_state(current);
    let old_display = display_state(&g).to_string();
    g.override_state = if next == g.base_state {
        None
    } else {
        Some(next.to_string())
    };
    let new_display = display_state(&g).to_string();
    drop(g);
    if old_display != new_display {
        notify_display_state(&new_display);
    }
    get_icon_status()
}

#[must_use]
pub fn icon_file_for_state(state: &str) -> &'static str {
    match state {
        ICON_RECORDING => "icon-red.webp",
        ICON_PROCESSING => "icon-yellow.webp",
        _ => "icon-blue.webp",
    }
}

#[must_use]
pub fn resolve_state_icon_path(state: &str) -> PathBuf {
    resolve_icon_path(icon_file_for_state(state))
}

/// Resolve bundled icon path for loading (dev: `assets/` next to crate, packaged: bundle or exe dir).
#[must_use]
pub fn resolve_icon_path(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let assets = manifest.join("assets").join(name);
    if assets.is_file() {
        return assets;
    }
    if let Some(dir) = crate::paths::packaged_assets_dir() {
        let p = dir.join(name);
        if p.is_file() {
            return p;
        }
    }
    if let Some(dir) = crate::paths::current_exe_parent() {
        let p = dir.join("assets").join(name);
        if p.is_file() {
            return p;
        }
    }
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::{icon_file_for_state, ICON_IDLE, ICON_PROCESSING, ICON_RECORDING};

    #[test]
    fn maps_runtime_states_to_colored_icons() {
        assert_eq!(icon_file_for_state(ICON_IDLE), "icon-blue.webp");
        assert_eq!(icon_file_for_state(ICON_RECORDING), "icon-red.webp");
        assert_eq!(icon_file_for_state(ICON_PROCESSING), "icon-yellow.webp");
        assert_eq!(icon_file_for_state("unknown"), "icon-blue.webp");
    }
}
