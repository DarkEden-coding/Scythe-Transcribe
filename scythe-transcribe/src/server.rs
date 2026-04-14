//! Axum HTTP API and static SPA serving.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::StatusCode;
use axum::middleware::{from_fn, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::Json;
use axum::Router;
use serde_json::{json, Value};
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

use crate::config::{API_HOST, API_PORT, MAX_UPLOAD_BYTES};
use crate::groq;
use crate::hotkey::{self, current_app_identity};
use crate::models::AppPreferences;
use crate::openrouter;
use crate::pipeline::{postprocess_only_http, transcribe_wav_bytes, TranscribeJob};
use crate::runtime_icon;
use crate::settings_store::{
    get_groq_api_key, get_openrouter_api_key, load_json_cache, load_preferences,
    load_transcription_history, openrouter_models_cache_path, save_json_cache, save_preferences,
    set_groq_api_key, set_openrouter_api_key,
};
use crate::startup;

#[derive(Clone)]
pub struct AppState {
    pub http: reqwest::Client,
    /// When false, only `/api/health` responds; other routes return 503.
    pub server_enabled: Arc<AtomicBool>,
}

fn static_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SCYTHE_STATIC_ROOT") {
        let pb = PathBuf::from(p);
        if pb.join("index.html").is_file() {
            return Some(pb);
        }
    }
    if let Some(root) = crate::paths::packaged_static_dir() {
        return Some(root);
    }
    let dev = Path::new(env!("CARGO_MANIFEST_DIR")).join("../frontend/dist");
    if dev.join("index.html").is_file() {
        return Some(dev);
    }
    None
}

fn cors() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
}

pub fn build_router() -> Router {
    let state = AppState {
        http: reqwest::Client::new(),
        server_enabled: Arc::new(AtomicBool::new(true)),
    };
    build_router_with_state(state)
}

pub fn build_router_with_state(state: AppState) -> Router {
    let enabled_flag = state.server_enabled.clone();
    let mut router = Router::new()
        .route("/api/health", get(health))
        .route(
            "/api/frontend-session/{session_id}/heartbeat",
            post(frontend_session_ok),
        )
        .route(
            "/api/frontend-session/{session_id}",
            delete(frontend_session_ok),
        )
        .route(
            "/api/frontend-session/{session_id}/close",
            post(frontend_session_ok),
        )
        .route(
            "/api/preferences",
            get(get_preferences).put(put_preferences),
        )
        .route("/api/audio-input-devices", get(audio_input_devices))
        .route("/api/keys", get(get_keys).put(put_keys))
        .route("/api/groq/chat-models", get(groq_chat_models))
        .route("/api/openrouter/models", get(openrouter_models_cached))
        .route(
            "/api/openrouter/models/refresh",
            post(openrouter_models_refresh),
        )
        .route("/api/accessibility", get(accessibility))
        .route(
            "/api/accessibility/request-prompt",
            post(accessibility_request_prompt),
        )
        .route(
            "/api/accessibility/open-settings",
            post(accessibility_open_settings),
        )
        .route(
            "/api/input-monitoring/request-prompt",
            post(input_monitoring_request_prompt),
        )
        .route(
            "/api/input-monitoring/open-settings",
            post(input_monitoring_open_settings),
        )
        .route(
            "/api/microphone/request-access",
            post(microphone_request_access),
        )
        .route(
            "/api/microphone/open-settings",
            post(microphone_open_settings),
        )
        .route("/api/startup", get(get_startup).put(put_startup))
        .route("/api/transcription-history", get(transcription_history))
        .route(
            "/api/transcribe",
            post(transcribe).layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES)),
        )
        .route("/api/postprocess", post(postprocess))
        .route("/api/runtime-state", get(runtime_state))
        .route("/api/runtime-icon/cycle", post(runtime_icon_cycle))
        .route("/api/runtime-icon/override", put(runtime_icon_override))
        .with_state(state.clone())
        .layer(from_fn({
            let enabled_flag = enabled_flag.clone();
            move |req: axum::http::Request<axum::body::Body>, next: Next| {
                let enabled_flag = enabled_flag.clone();
                async move {
                    let path = req.uri().path();
                    if path == "/api/health" || enabled_flag.load(Ordering::SeqCst) {
                        next.run(req).await
                    } else {
                        StatusCode::SERVICE_UNAVAILABLE.into_response()
                    }
                }
            }
        }));

    if let Some(root) = static_root() {
        let index = root.join("index.html");
        let assets = root.join("assets");
        if assets.is_dir() {
            router = router.nest_service("/assets", ServeDir::new(assets));
        }
        router =
            router.fallback_service(ServeDir::new(&root).not_found_service(ServeFile::new(index)));
    }

    router.layer(TraceLayer::new_for_http()).layer(cors())
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn frontend_session_ok() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn get_preferences() -> Json<AppPreferences> {
    Json(load_preferences())
}

async fn put_preferences(Json(body): Json<AppPreferences>) -> Json<AppPreferences> {
    let _ = save_preferences(&body);
    Json(load_preferences())
}

async fn audio_input_devices() -> Json<Value> {
    Json(json!({
        "builtin_id": hotkey::AUDIO_INPUT_BUILT_IN,
        "system_default_id": hotkey::AUDIO_INPUT_SYSTEM_DEFAULT,
        "devices": hotkey::audio_input_devices(),
    }))
}

async fn get_keys() -> Json<Value> {
    Json(json!({
        "groq_configured": !get_groq_api_key().trim().is_empty(),
        "openrouter_configured": !get_openrouter_api_key().trim().is_empty(),
    }))
}

#[derive(serde::Deserialize)]
struct KeysUpdate {
    groq: Option<String>,
    openrouter: Option<String>,
}

async fn put_keys(Json(body): Json<KeysUpdate>) -> Json<Value> {
    if let Some(k) = body.groq {
        let _ = set_groq_api_key(k.trim());
    }
    if let Some(k) = body.openrouter {
        let _ = set_openrouter_api_key(k.trim());
    }
    get_keys().await
}

async fn groq_chat_models(State(st): State<AppState>) -> Json<Value> {
    let key = get_groq_api_key();
    if key.is_empty() {
        return Json(json!({ "models": [] }));
    }
    let models = groq::list_chat_models(&st.http, &key)
        .await
        .unwrap_or_default();
    Json(json!({ "models": models.into_iter().take(200).collect::<Vec<_>>() }))
}

fn openrouter_cache_stale(raw: &Value) -> bool {
    match raw {
        Value::Array(a) if a.is_empty() => true,
        Value::Array(a) => {
            if let Some(Value::Object(o)) = a.first() {
                !o.contains_key("pricing_prompt") && !o.contains_key("pricing_completion")
            } else {
                true
            }
        }
        _ => true,
    }
}

async fn fetch_openrouter_models_array(
    http: &reqwest::Client,
    key_ref: Option<&str>,
) -> anyhow::Result<Value> {
    let list = openrouter::fetch_models_raw(http, key_ref).await?;
    let infos = openrouter::parse_model_infos(&list.to_vec());
    let v: Vec<Value> = infos
        .into_iter()
        .filter_map(|i| serde_json::to_value(i).ok())
        .collect();
    Ok(Value::Array(v))
}

async fn openrouter_models_cached(State(st): State<AppState>) -> Json<Value> {
    let path = openrouter_models_cache_path();
    let raw = load_json_cache(&path).unwrap_or(Value::Null);
    if openrouter_cache_stale(&raw) {
        let key = get_openrouter_api_key();
        let key_ref = if key.is_empty() {
            None
        } else {
            Some(key.as_str())
        };
        if let Ok(arr) = fetch_openrouter_models_array(&st.http, key_ref).await {
            let _ = save_json_cache(&path, &arr);
            return Json(json!({ "models": arr }));
        }
        return Json(json!({ "models": [] }));
    }
    if let Value::Array(a) = raw {
        return Json(json!({ "models": a }));
    }
    Json(json!({ "models": [] }))
}

async fn openrouter_models_refresh(State(st): State<AppState>) -> Result<Json<Value>, ApiError> {
    let key = get_openrouter_api_key();
    let key_ref = if key.is_empty() {
        None
    } else {
        Some(key.as_str())
    };
    let arr = fetch_openrouter_models_array(&st.http, key_ref)
        .await
        .map_err(|e| ApiError(e.to_string()))?;
    let path = openrouter_models_cache_path();
    let count = arr.as_array().map(|a| a.len()).unwrap_or(0);
    save_json_cache(&path, &arr).map_err(|e| ApiError(e.to_string()))?;
    Ok(Json(json!({ "count": count, "models": arr })))
}

struct ApiError(String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (StatusCode::BAD_GATEWAY, self.0).into_response()
    }
}

async fn accessibility() -> Json<Value> {
    #[cfg(target_os = "macos")]
    {
        hotkey::start_hotkey_listener();
        let mic = crate::macos::microphone_authorization();
        Json(json!({
            "supported": true,
            "trusted": crate::macos::is_accessibility_trusted(),
            "input_monitoring_trusted": crate::macos::is_input_monitoring_trusted(),
            "microphone_authorized": mic.is_authorized(),
            "microphone_authorization": mic.as_str(),
            "hotkey": hotkey::get_hotkey_listener_status(),
            "identity": current_app_identity(),
        }))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Json(json!({
            "supported": false,
            "trusted": true,
            "input_monitoring_trusted": true,
            "microphone_authorized": true,
            "microphone_authorization": "authorized",
            "hotkey": { "state": "unsupported" },
        }))
    }
}

async fn accessibility_request_prompt() -> Json<Value> {
    #[cfg(target_os = "macos")]
    {
        crate::macos::request_accessibility_prompt();
        Json(json!({ "status": "ok", "action": "accessibility_prompt" }))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Json(json!({ "status": "unsupported" }))
    }
}

async fn accessibility_open_settings() -> Json<Value> {
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open")
            .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")
            .spawn();
        Json(json!({ "status": "ok", "panel": "Privacy_Accessibility" }))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Json(json!({ "status": "unsupported" }))
    }
}

async fn input_monitoring_request_prompt() -> Json<Value> {
    #[cfg(target_os = "macos")]
    {
        crate::macos::request_input_monitoring_trust_prompt();
        Json(json!({ "status": "ok", "action": "input_monitoring_prompt" }))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Json(json!({ "status": "unsupported" }))
    }
}

async fn input_monitoring_open_settings() -> Json<Value> {
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open")
            .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent")
            .spawn();
        Json(json!({ "status": "ok", "panel": "Privacy_ListenEvent" }))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Json(json!({ "status": "unsupported" }))
    }
}

async fn microphone_request_access() -> Json<Value> {
    #[cfg(target_os = "macos")]
    {
        crate::macos::request_microphone_access_prompt();
        Json(json!({ "status": "ok", "action": "microphone_request_access" }))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Json(json!({ "status": "unsupported" }))
    }
}

async fn microphone_open_settings() -> Json<Value> {
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open")
            .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")
            .spawn();
        Json(json!({ "status": "ok", "panel": "Privacy_Microphone" }))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Json(json!({ "status": "unsupported" }))
    }
}

async fn get_startup() -> Json<Value> {
    Json(json!({ "enabled": startup::is_startup_enabled() }))
}

async fn put_startup(Json(body): Json<Value>) -> Json<Value> {
    let enabled = body
        .get("enabled")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let _ = startup::set_startup_enabled(enabled);
    Json(json!({ "enabled": startup::is_startup_enabled() }))
}

async fn transcription_history() -> Json<Value> {
    Json(json!({ "entries": load_transcription_history() }))
}

async fn transcribe(
    State(st): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<Value>, Api422> {
    let mut meta: Option<String> = None;
    let mut audio: Option<Bytes> = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| Api422(e.to_string()))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name == "meta" {
            meta = Some(field.text().await.map_err(|e| Api422(e.to_string()))?);
        } else if name == "audio" {
            audio = Some(field.bytes().await.map_err(|e| Api422(e.to_string()))?);
        }
    }
    let meta_s = meta.ok_or_else(|| Api422("missing meta".into()))?;
    let job: TranscribeJob =
        serde_json::from_str(&meta_s).map_err(|e| Api422(format!("Invalid meta JSON: {e}")))?;
    let bytes = audio.ok_or_else(|| Api422("missing audio".into()))?;
    let res = transcribe_wav_bytes(&st.http, &job, &bytes)
        .await
        .map_err(|e| Api422(format!("{}: {}", e.0, e.1)))?;
    Ok(Json(serde_json::to_value(&res).unwrap_or(Value::Null)))
}

struct Api422(String);

impl IntoResponse for Api422 {
    fn into_response(self) -> Response {
        (StatusCode::UNPROCESSABLE_ENTITY, self.0).into_response()
    }
}

async fn postprocess(
    State(st): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, Api422> {
    let out = postprocess_only_http(&st.http, &body)
        .await
        .map_err(|e| Api422(format!("{}: {}", e.0, e.1)))?;
    Ok(Json(json!({ "processed": out })))
}

async fn runtime_state() -> Json<Value> {
    hotkey::start_hotkey_listener();
    let hotkey = hotkey::get_hotkey_listener_status();
    let icon = runtime_icon::get_icon_status();
    let capture = hotkey
        .get("capture_state")
        .and_then(|v| v.as_str())
        .unwrap_or("idle");
    Json(json!({
        "icon_state": icon.display_state,
        "capture_state": capture,
        "capturing_audio": capture == "recording",
        "processing_audio": capture == "processing",
        "os_icon": icon,
        "hotkey": hotkey,
    }))
}

async fn runtime_icon_cycle() -> Json<Value> {
    let os_icon = runtime_icon::cycle_icon_override();
    Json(json!({ "os_icon": os_icon }))
}

#[derive(serde::Deserialize)]
struct IconOverrideBody {
    override_state: Option<String>,
}

async fn runtime_icon_override(Json(body): Json<IconOverrideBody>) -> Json<Value> {
    let o = body.override_state.as_deref();
    let valid = matches!(
        o,
        None | Some("idle") | Some("recording") | Some("processing")
    );
    let os_icon = if valid {
        runtime_icon::set_icon_override(o)
    } else {
        runtime_icon::get_icon_status()
    };
    Json(json!({ "os_icon": os_icon }))
}

/// Run the Axum server (bind `API_HOST:API_PORT`).
pub async fn serve(state: AppState) -> anyhow::Result<()> {
    let app = build_router_with_state(state);
    let addr = format!("{API_HOST}:{API_PORT}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn health_handler_ok() {
        let j = health().await;
        assert_eq!(j.0["status"], "ok");
    }

    #[tokio::test]
    async fn runtime_state_shape() {
        let j = runtime_state().await;
        assert!(j.0.get("hotkey").is_some());
    }
}
