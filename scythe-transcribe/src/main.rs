//! Entry: optional tray, local HTTP API, global hotkey listener.

// Release Windows builds: no console window (tray + background server only).
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use scythe_transcribe::config::PUBLIC_BASE_URL;
use scythe_transcribe::hotkey;
use scythe_transcribe::instance_lock;
use scythe_transcribe::paths::tray_icon_path;
use scythe_transcribe::server::{self, AppState};
use tao::event::Event;
use tao::event_loop::{ControlFlow, EventLoopBuilder};
#[cfg(target_os = "macos")]
use tao::platform::macos::{ActivationPolicy, EventLoopExtMacOS};
use tray_icon::menu::{Menu, MenuEvent, MenuItem};
use tray_icon::TrayIconBuilder;

fn env_truthy(key: &str) -> bool {
    matches!(
        env::var(key)
            .unwrap_or_default()
            .trim()
            .to_lowercase()
            .as_str(),
        "1" | "true" | "yes"
    )
}

fn env_falsy(key: &str) -> bool {
    matches!(
        env::var(key)
            .unwrap_or_default()
            .trim()
            .to_lowercase()
            .as_str(),
        "0" | "false" | "no"
    )
}

fn open_settings_url() {
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", "", PUBLIC_BASE_URL])
            .spawn();
        return;
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = open::that(PUBLIC_BASE_URL);
    }
}

enum UserEvent {
    Menu(MenuEvent),
}

fn load_tray_icon() -> tray_icon::Icon {
    let path = tray_icon_path();
    let image = image::open(&path)
        .unwrap_or_else(|_| panic!("missing tray icon at {}", path.display()))
        .into_rgba8();
    let (w, h) = image.dimensions();
    tray_icon::Icon::from_rgba(image.into_raw(), w, h).expect("icon rgba")
}

fn run_tray(server_enabled: Arc<AtomicBool>) {
    let mut event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    #[cfg(target_os = "macos")]
    {
        // Menu bar tray only: no Dock icon, no app as frontmost window on launch.
        event_loop.set_activation_policy(ActivationPolicy::Accessory);
        event_loop.set_activate_ignoring_other_apps(true);
    }
    let proxy = event_loop.create_proxy();
    MenuEvent::set_event_handler(Some(move |e| {
        let _ = proxy.send_event(UserEvent::Menu(e));
    }));

    let menu = Menu::new();
    let open_i = MenuItem::new("Open settings", true, None);
    let server_toggle_i = MenuItem::new("Disable server", true, None);
    let quit_i = MenuItem::new("Shutdown", true, None);
    let _ = menu.append_items(&[&open_i, &server_toggle_i, &quit_i]);

    let sync_server_toggle_label = |item: &MenuItem, server_on: bool| {
        item.set_text(if server_on {
            "Disable server"
        } else {
            "Enable server"
        });
    };

    let mut tray = None::<tray_icon::TrayIcon>;
    let server_enabled_menu = server_enabled.clone();

    event_loop.run(move |event, _target, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::NewEvents(tao::event::StartCause::Init) => {
                sync_server_toggle_label(
                    &server_toggle_i,
                    server_enabled_menu.load(Ordering::SeqCst),
                );
                let icon = load_tray_icon();
                tray = Some(
                    TrayIconBuilder::new()
                        .with_menu(Box::new(menu.clone()))
                        .with_tooltip("Scythe-Transcribe")
                        .with_icon(icon)
                        .build()
                        .expect("tray icon"),
                );
            }
            Event::UserEvent(UserEvent::Menu(e)) => {
                if e.id == open_i.id() {
                    if server_enabled_menu.load(Ordering::SeqCst) {
                        open_settings_url();
                    }
                } else if e.id == server_toggle_i.id() {
                    let was_on = server_enabled_menu.load(Ordering::SeqCst);
                    server_enabled_menu.store(!was_on, Ordering::SeqCst);
                    sync_server_toggle_label(&server_toggle_i, !was_on);
                } else if e.id == quit_i.id() {
                    tray.take();
                    *control_flow = ControlFlow::Exit;
                }
            }
            _ => {}
        }
    });
}

fn main() {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    if instance_lock::try_acquire_single_instance().is_none() {
        std::process::exit(0);
    }

    hotkey::start_hotkey_listener();

    let server_enabled = Arc::new(AtomicBool::new(true));
    let state = AppState {
        http: reqwest::Client::new(),
        server_enabled: server_enabled.clone(),
    };

    if env_truthy("SCYTHE_SERVER_ONLY") || env_falsy("SCYTHE_TRAY") {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("runtime");
        rt.block_on(async {
            if let Err(e) = server::serve(state).await {
                tracing::error!("server error: {e}");
            }
        });
        return;
    }

    let st = state.clone();
    thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("runtime");
        rt.block_on(async {
            if let Err(e) = server::serve(st).await {
                tracing::error!("server error: {e}");
            }
        });
    });

    thread::sleep(Duration::from_millis(200));
    run_tray(server_enabled);
}
