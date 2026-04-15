//! Scythe-Transcribe: local API server, transcription pipeline, and desktop integration.

pub mod config;
pub mod groq;
pub mod hotkey;
pub mod instance_lock;
#[cfg(target_os = "macos")]
pub mod macos;
pub mod models;
pub mod openrouter;
pub mod paths;
pub mod pipeline;
pub mod prompts;
pub mod runtime_icon;
pub mod server;
pub mod settings_store;
pub mod startup;
pub mod text_replacements;
pub mod wav_speech_chunks;

pub use server::build_router;
