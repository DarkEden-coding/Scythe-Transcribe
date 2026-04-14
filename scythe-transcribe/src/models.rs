//! Domain types for providers and preferences.

use serde::{Deserialize, Serialize};

use crate::prompts::OPENROUTER_TRANSCRIPTION_INSTRUCTION;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TranscriptionProvider {
    Groq,
    Openrouter,
}

impl TranscriptionProvider {
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Groq => "groq",
            Self::Openrouter => "openrouter",
        }
    }

    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        if s == Self::Groq.as_str() {
            Some(Self::Groq)
        } else if s == Self::Openrouter.as_str() {
            Some(Self::Openrouter)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatProvider {
    Groq,
    Openrouter,
}

impl ChatProvider {
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Groq => "groq",
            Self::Openrouter => "openrouter",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppPreferences {
    pub transcription_provider: String,
    pub transcription_model_groq: String,
    pub transcription_model_openrouter: String,
    pub postprocess_enabled: bool,
    pub postprocess_prompt: String,
    pub postprocess_provider: String,
    pub postprocess_model: String,
    pub postprocess_groq_reasoning_effort: String,
    pub postprocess_openrouter_reasoning_effort: String,
    pub openrouter_models_cache_hint: String,
    pub keyword_replacement_spec: String,
    pub openrouter_transcription_instruction: String,
    pub hotkey_toggle_recording: String,
}

impl Default for AppPreferences {
    fn default() -> Self {
        Self {
            transcription_provider: TranscriptionProvider::Groq.as_str().to_string(),
            transcription_model_groq: "whisper-large-v3-turbo".to_string(),
            transcription_model_openrouter: String::new(),
            postprocess_enabled: false,
            postprocess_prompt: "Summarize the transcript in bullet points.".to_string(),
            postprocess_provider: ChatProvider::Openrouter.as_str().to_string(),
            postprocess_model: "openai/gpt-4o-mini".to_string(),
            postprocess_groq_reasoning_effort: String::new(),
            postprocess_openrouter_reasoning_effort: String::new(),
            openrouter_models_cache_hint: String::new(),
            keyword_replacement_spec: String::new(),
            openrouter_transcription_instruction: OPENROUTER_TRANSCRIPTION_INSTRUCTION.to_string(),
            hotkey_toggle_recording: "ctrl+shift+space".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterModelInfo {
    pub model_id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub supports_audio_input: bool,
    #[serde(default = "default_true")]
    pub supports_text: bool,
    #[serde(default)]
    pub pricing_prompt: String,
    #[serde(default)]
    pub pricing_completion: String,
}

fn default_true() -> bool {
    true
}
