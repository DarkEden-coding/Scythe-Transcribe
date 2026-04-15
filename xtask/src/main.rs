//! Build and packaging tasks (`cargo xtask ...`).

use std::fs;
use std::io;
use std::path::Path;
use std::process::Command;

#[cfg(windows)]
const NPM: &str = "npm.cmd";

#[cfg(not(windows))]
const NPM: &str = "npm";

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let task = args.next().unwrap_or_else(|| "help".to_string());
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask in workspace")
        .to_path_buf();

    match task.as_str() {
        "build-ui" => {
            let frontend = workspace_root.join("frontend");
            let status = Command::new(NPM)
                .args(["ci"])
                .current_dir(&frontend)
                .status()?;
            if !status.success() {
                anyhow::bail!("npm ci failed");
            }
            let status = Command::new(NPM)
                .args(["run", "build"])
                .current_dir(&frontend)
                .status()?;
            if !status.success() {
                anyhow::bail!("npm run build failed");
            }
        }
        "dist" => {
            let status = Command::new("cargo")
                .args(["run", "-p", "xtask", "--", "build-ui"])
                .current_dir(&workspace_root)
                .status()?;
            if !status.success() {
                anyhow::bail!("build-ui failed");
            }
            let status = Command::new("cargo")
                .args(["test", "--workspace", "--all-features"])
                .current_dir(&workspace_root)
                .status()?;
            if !status.success() {
                anyhow::bail!("cargo test failed");
            }
            let status = Command::new("cargo")
                .args(["build", "--release", "-p", "scythe-transcribe"])
                .current_dir(&workspace_root)
                .status()?;
            if !status.success() {
                anyhow::bail!("cargo build --release failed");
            }
            package_platform(&workspace_root)?;
        }
        "help" | "--help" | "-h" => {
            eprintln!("usage: cargo xtask [build-ui|dist]");
            eprintln!(
                "  dist — release build; writes pack/ (Windows: exe + static + assets; macOS: .app bundle)"
            );
        }
        other => {
            anyhow::bail!("unknown task: {other}");
        }
    }
    Ok(())
}

fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

fn copy_dir_if_exists(src: &Path, dst: &Path) -> io::Result<()> {
    if src.is_dir() {
        copy_dir_all(src, dst)?;
    }
    Ok(())
}

fn package_platform(workspace_root: &Path) -> anyhow::Result<()> {
    let pack = workspace_root.join("pack");
    if pack.exists() {
        fs::remove_dir_all(&pack)?;
    }
    fs::create_dir_all(&pack)?;

    let target_release = workspace_root.join("target/release");
    let frontend_dist = workspace_root.join("frontend/dist");
    let crate_assets = workspace_root.join("scythe-transcribe/assets");

    if !frontend_dist.join("index.html").is_file() {
        anyhow::bail!("missing frontend/dist/index.html; run build-ui first");
    }

    #[cfg(target_os = "macos")]
    {
        let app = pack.join("Scythe-Transcribe.app");
        let contents = app.join("Contents");
        let macos = contents.join("MacOS");
        let resources = contents.join("Resources");
        fs::create_dir_all(&macos)?;
        fs::create_dir_all(resources.join("static"))?;

        let bin_src = target_release.join("scythe-transcribe");
        if !bin_src.is_file() {
            anyhow::bail!("missing release binary at {}", bin_src.display());
        }
        let bin_dst = macos.join("scythe-transcribe");
        fs::copy(&bin_src, &bin_dst)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&bin_dst)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&bin_dst, perms)?;
        }

        copy_dir_all(&frontend_dist, &resources.join("static"))?;
        copy_dir_if_exists(&crate_assets, &resources.join("assets"))?;

        let version = env!("CARGO_PKG_VERSION");
        let plist = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>scythe-transcribe</string>
    <key>CFBundleIdentifier</key>
    <string>com.scythe-transcribe.app</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Scythe Transcribe</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>{version}</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Scythe Transcribe records from the microphone when you use the global hold-to-record shortcut.</string>
    <key>NSAccessibilityUsageDescription</key>
    <string>Scythe Transcribe uses accessibility to paste transcriptions into the app you are working in.</string>
</dict>
</plist>
"#
        );
        fs::write(contents.join("Info.plist"), plist)?;
        eprintln!("Packaged: {}", app.display());
    }

    #[cfg(target_os = "windows")]
    {
        let exe_src = target_release.join("scythe-transcribe.exe");
        if !exe_src.is_file() {
            anyhow::bail!("missing release binary at {}", exe_src.display());
        }
        fs::copy(&exe_src, pack.join("scythe-transcribe.exe"))?;
        copy_dir_all(&frontend_dist, &pack.join("static"))?;
        copy_dir_if_exists(&crate_assets, &pack.join("assets"))?;
        eprintln!("Packaged: {}\\scythe-transcribe.exe", pack.display());
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let bin_name = "scythe-transcribe";
        let bin_src = target_release.join(bin_name);
        if !bin_src.is_file() {
            anyhow::bail!("missing release binary at {}", bin_src.display());
        }
        fs::copy(&bin_src, pack.join(bin_name))?;
        copy_dir_all(&frontend_dist, &pack.join("static"))?;
        copy_dir_if_exists(&crate_assets, &pack.join("assets"))?;
        eprintln!("Packaged: {}/{}", pack.display(), bin_name);
    }

    Ok(())
}
