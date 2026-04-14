//! Resolve bundled static files and assets next to the executable or inside a macOS `.app` bundle.

use std::path::PathBuf;

#[must_use]
pub fn current_exe_parent() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf))
}

/// Walks ancestors from the executable until a `.app` bundle is found; returns `Contents/Resources` if present.
#[must_use]
pub fn macos_bundle_resources_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    for anc in exe.ancestors() {
        if anc.extension().is_some_and(|e| e == "app") {
            let res = anc.join("Contents/Resources");
            if res.is_dir() {
                return Some(res);
            }
            return None;
        }
    }
    None
}

#[must_use]
pub fn packaged_static_dir() -> Option<PathBuf> {
    if let Some(res) = macos_bundle_resources_dir() {
        let s = res.join("static");
        if s.join("index.html").is_file() {
            return Some(s);
        }
    }
    if let Some(dir) = current_exe_parent() {
        let s = dir.join("static");
        if s.join("index.html").is_file() {
            return Some(s);
        }
    }
    None
}

#[must_use]
pub fn packaged_assets_dir() -> Option<PathBuf> {
    if let Some(res) = macos_bundle_resources_dir() {
        let a = res.join("assets");
        if a.is_dir() {
            return Some(a);
        }
    }
    if let Some(dir) = current_exe_parent() {
        let a = dir.join("assets");
        if a.is_dir() {
            return Some(a);
        }
    }
    None
}

#[must_use]
pub fn tray_icon_path() -> PathBuf {
    if let Some(res) = macos_bundle_resources_dir() {
        let p = res.join("assets").join("icon-blue.webp");
        if p.is_file() {
            return p;
        }
    }
    if let Some(dir) = current_exe_parent() {
        let p = dir.join("assets").join("icon-blue.webp");
        if p.is_file() {
            return p;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join("icon-blue.webp")
}
