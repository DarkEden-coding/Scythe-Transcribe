//! Login item registration: macOS LaunchAgent plist, Windows Run registry key.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use plist::Dictionary;
use plist::Value;

const PLIST_LABEL: &str = "com.scythe-transcribe.app";
const PLIST_NAME: &str = "com.scythe-transcribe.app.plist";
#[cfg(target_os = "windows")]
const WINDOWS_RUN_VALUE_NAME: &str = "Scythe-Transcribe";

fn launch_agents_dir() -> PathBuf {
    dirs::home_dir()
        .expect("home")
        .join("Library")
        .join("LaunchAgents")
}

fn plist_path() -> PathBuf {
    launch_agents_dir().join(PLIST_NAME)
}

#[cfg(target_os = "macos")]
fn macos_app_bundle_path(executable: &std::path::Path) -> Option<PathBuf> {
    std::iter::once(executable.to_path_buf())
        .chain(executable.ancestors().map(PathBuf::from))
        .find(|path| {
            path.extension().is_some_and(|e| e == "app")
                && path.join("Contents/Info.plist").is_file()
        })
}

#[cfg(target_os = "macos")]
fn macos_program_arguments() -> Vec<String> {
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("/"));
    if macos_app_bundle_path(&exe).is_some() {
        return vec![
            "/usr/bin/open".to_string(),
            "-g".to_string(),
            exe.to_string_lossy().to_string(),
        ];
    }
    vec![exe.to_string_lossy().to_string()]
}

#[cfg(not(target_os = "macos"))]
fn macos_program_arguments() -> Vec<String> {
    vec![]
}

fn plist_payload(argv: &[String]) -> Value {
    let mut dict = Dictionary::new();
    dict.insert("Label".to_string(), Value::String(PLIST_LABEL.to_string()));
    dict.insert(
        "ProgramArguments".to_string(),
        Value::Array(argv.iter().cloned().map(Value::String).collect()),
    );
    dict.insert("RunAtLoad".to_string(), Value::Boolean(true));
    dict.insert("KeepAlive".to_string(), Value::Boolean(false));
    let home = dirs::home_dir().expect("home");
    dict.insert(
        "StandardOutPath".to_string(),
        Value::String(
            home.join("Library/Logs/scythe-transcribe.log")
                .to_string_lossy()
                .to_string(),
        ),
    );
    dict.insert(
        "StandardErrorPath".to_string(),
        Value::String(
            home.join("Library/Logs/scythe-transcribe-error.log")
                .to_string_lossy()
                .to_string(),
        ),
    );
    Value::Dictionary(dict)
}

#[cfg(target_os = "macos")]
pub fn is_startup_enabled() -> bool {
    plist_path().is_file()
}

#[cfg(target_os = "macos")]
pub fn set_startup_enabled(enabled: bool) -> std::io::Result<()> {
    let plist = plist_path();
    if enabled {
        let agents = launch_agents_dir();
        fs::create_dir_all(&agents)?;
        let argv = macos_program_arguments();
        let payload = plist_payload(
            &argv
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>(),
        );
        let mut buf = Vec::new();
        plist::to_writer_xml(&mut buf, &payload).map_err(std::io::Error::other)?;
        fs::write(&plist, buf)?;
        let _ = Command::new("launchctl")
            .args(["load", plist.to_str().unwrap_or("")])
            .output();
    } else if plist.is_file() {
        let _ = Command::new("launchctl")
            .args(["unload", plist.to_str().unwrap_or("")])
            .output();
        fs::remove_file(plist)?;
    }
    Ok(())
}

#[cfg(target_os = "windows")]
pub fn is_startup_enabled() -> bool {
    use winreg::enums::*;
    use winreg::RegKey;
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let Ok(key) = hkcu.open_subkey(r"Software\Microsoft\Windows\CurrentVersion\Run") else {
        return false;
    };
    key.get_value::<String, _>(WINDOWS_RUN_VALUE_NAME).is_ok()
}

#[cfg(target_os = "windows")]
pub fn set_startup_enabled(enabled: bool) -> std::io::Result<()> {
    use winreg::enums::*;
    use winreg::RegKey;
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let (key, _) = hkcu.create_subkey(r"Software\Microsoft\Windows\CurrentVersion\Run")?;
    if enabled {
        let exe = std::env::current_exe()?;
        let cmd = format!("\"{}\"", exe.display());
        key.set_value(WINDOWS_RUN_VALUE_NAME, &cmd)?;
    } else {
        let _ = key.delete_value(WINDOWS_RUN_VALUE_NAME);
    }
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn is_startup_enabled() -> bool {
    false
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn set_startup_enabled(_enabled: bool) -> std::io::Result<()> {
    Ok(())
}
