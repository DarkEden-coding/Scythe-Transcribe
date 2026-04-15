//! Runtime configuration (no CLI parsing).

/// Local API bind address.
pub const API_HOST: &str = "127.0.0.1";
/// Local API port (browser + SPA).
pub const API_PORT: u16 = 8765;

/// Browser URL when served by the local API.
pub const PUBLIC_BASE_URL: &str = concat!("http://", "127.0.0.1", ":", "8765", "/");

/// Vite dev server default port.
pub const VITE_DEV_PORT: u16 = 5173;

/// Maximum upload size for recorded audio (bytes).
pub const MAX_UPLOAD_BYTES: usize = 50 * 1024 * 1024;

/// Post-process: max user characters per chunk.
pub const POSTPROCESS_CHUNK_MAX_USER_CHARS: usize = 4500;
/// Post-process: max parallel chunk requests.
pub const POSTPROCESS_MAX_PARALLEL_CHUNKS: usize = 8;

/// Groq ASR: default minimum/target audio chunk length in seconds.
///
/// Groq transcription is billed with a minimum duration per request, so the
/// default keeps generated chunks near the billing floor instead of producing
/// many short requests.
pub const GROQ_ASR_CHUNK_DURATION_SEC: f64 = 10.0;

/// When refining a nominal chunk boundary, search ±this many seconds for a low-energy pause.
pub const GROQ_ASR_CHUNK_BOUNDARY_SEARCH_SEC: f64 = 0.85;

/// Chunks shorter than this (after pause search) are avoided by boundary constraints when possible.
pub const GROQ_ASR_MIN_CHUNK_SEC: f64 = 1.25;

/// Max concurrent Groq transcription requests when using chunked ASR.
pub const GROQ_ASR_MAX_PARALLEL_CHUNKS: usize = 6;

/// Upper bound for post-process completion `max_tokens` / `max_completion_tokens`.
#[must_use]
pub fn postprocess_max_completion_tokens(system_prompt: &str, user_content: &str) -> u32 {
    let combined = system_prompt.len() + user_content.len();
    let approx_input_tokens = combined.div_ceil(4).max(1);
    let v = (approx_input_tokens * 2).clamp(256, 2048);
    v as u32
}

#[cfg(test)]
mod tests {
    use super::postprocess_max_completion_tokens;

    #[test]
    fn postprocess_max_tokens_clamped() {
        let short = "x".repeat(100);
        assert_eq!(postprocess_max_completion_tokens("", &short), 256);
        let mid = postprocess_max_completion_tokens("sys", &"u".repeat(4000));
        assert!(mid > 256 && mid <= 2048);
        assert_eq!(
            postprocess_max_completion_tokens("", &"z".repeat(100_000)),
            2048
        );
    }
}
