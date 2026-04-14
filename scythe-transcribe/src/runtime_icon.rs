//! Tray/menu-bar icon state (idle / recording / processing) and manual override.

use std::path::PathBuf;
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

#[must_use]
pub fn get_icon_status() -> IconStatus {
    let g = STATE.lock().expect("runtime icon");
    let display = g.override_state.as_deref().unwrap_or(g.base_state.as_str());
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
    g.base_state = state.to_string();
}

pub fn set_icon_override(state: Option<&str>) -> IconStatus {
    let mut g = STATE.lock().expect("runtime icon");
    g.override_state = state.map(str::to_string);
    drop(g);
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
    g.override_state = if next == g.base_state {
        None
    } else {
        Some(next.to_string())
    };
    drop(g);
    get_icon_status()
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
