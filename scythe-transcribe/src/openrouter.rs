//! OpenRouter: models, audio chat transcription, and text chat.

use std::collections::HashSet;

use base64::Engine;
use serde_json::Value;

use crate::config::postprocess_max_completion_tokens;
use crate::models::OpenRouterModelInfo;
use crate::prompts::OPENROUTER_TRANSCRIPTION_INSTRUCTION;

pub const OPENROUTER_BASE: &str = "https://openrouter.ai/api/v1";

fn headers(api_key: Option<&str>) -> reqwest::header::HeaderMap {
    let mut h = reqwest::header::HeaderMap::new();
    h.insert(
        reqwest::header::ACCEPT,
        reqwest::header::HeaderValue::from_static("application/json"),
    );
    if let Some(k) = api_key {
        if let Ok(v) = reqwest::header::HeaderValue::from_str(&format!("Bearer {k}")) {
            h.insert(reqwest::header::AUTHORIZATION, v);
        }
        h.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );
    }
    h
}

pub async fn fetch_models_raw(
    client: &reqwest::Client,
    api_key: Option<&str>,
) -> anyhow::Result<Vec<Value>> {
    let url = format!("{OPENROUTER_BASE}/models");
    let res = client.get(url).headers(headers(api_key)).send().await?;
    res.error_for_status_ref()?;
    let data: Value = res.json().await?;
    let inner = data.get("data").and_then(|d| d.as_array());
    let Some(list) = inner else {
        return Ok(vec![]);
    };
    Ok(list
        .iter()
        .filter_map(|x| x.as_object().map(|o| Value::Object(o.clone())))
        .collect())
}

fn format_usd_per_million(token_price: Option<f64>) -> String {
    let Some(t) = token_price else {
        return String::new();
    };
    let per_m = t * 1_000_000.0;
    if per_m <= 0.0 {
        return "free".to_string();
    }
    if per_m < 0.0001 {
        return format!("${:.6}/M", per_m);
    }
    if per_m < 0.01 {
        return format!("${:.4}/M", per_m);
    }
    if per_m < 1.0 {
        return format!("${:.3}/M", per_m);
    }
    if per_m < 100.0 {
        return format!("${:.2}/M", per_m);
    }
    format!("${:.0}/M", per_m)
}

fn pricing_strings(pricing: &Value) -> (String, String) {
    let Some(p) = pricing.as_object() else {
        return (String::new(), String::new());
    };
    let p_in = p.get("prompt").and_then(|v| v.as_f64()).or_else(|| {
        p.get("prompt")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
    });
    let p_out = p.get("completion").and_then(|v| v.as_f64()).or_else(|| {
        p.get("completion")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
    });
    (format_usd_per_million(p_in), format_usd_per_million(p_out))
}

fn architecture_modalities(model: &Value) -> (HashSet<String>, HashSet<String>) {
    let mut in_mod = HashSet::new();
    let mut out_mod = HashSet::new();
    if let Some(arch) = model.get("architecture").and_then(|a| a.as_object()) {
        if let Some(arr) = arch.get("input_modalities").and_then(|a| a.as_array()) {
            for x in arr {
                if let Some(s) = x.as_str() {
                    in_mod.insert(s.to_lowercase());
                }
            }
        }
        if let Some(arr) = arch.get("output_modalities").and_then(|a| a.as_array()) {
            for x in arr {
                if let Some(s) = x.as_str() {
                    out_mod.insert(s.to_lowercase());
                }
            }
        }
    }
    if let Some(top) = model.get("top_provider").and_then(|t| t.as_object()) {
        if let Some(arr) = top.get("input_modalities").and_then(|a| a.as_array()) {
            for x in arr {
                if let Some(s) = x.as_str() {
                    in_mod.insert(s.to_lowercase());
                }
            }
        }
    }
    if let Some(arr) = model.get("input_modalities").and_then(|a| a.as_array()) {
        for x in arr {
            if let Some(s) = x.as_str() {
                in_mod.insert(s.to_lowercase());
            }
        }
    }
    (in_mod, out_mod)
}

#[must_use]
pub fn parse_model_infos(models: &[Value]) -> Vec<OpenRouterModelInfo> {
    let mut result = Vec::new();
    for m in models {
        let Some(mid) = m.get("id").and_then(|v| v.as_str()) else {
            continue;
        };
        if mid.is_empty() {
            continue;
        }
        let name = m
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(mid)
            .to_string();
        let (in_mod, out_mod) = architecture_modalities(m);
        let mut supports_audio = in_mod.contains("audio");
        let supports_text =
            out_mod.contains("text") || in_mod.contains("text") || out_mod.is_empty();
        let lower = mid.to_lowercase();
        if !supports_audio
            && (lower.contains("whisper")
                || lower.contains("gpt-4o-transcribe")
                || lower.contains("audio"))
        {
            supports_audio = true;
        }
        let (p_in, p_out) = pricing_strings(m.get("pricing").unwrap_or(&Value::Null));
        result.push(OpenRouterModelInfo {
            model_id: mid.to_string(),
            name,
            supports_audio_input: supports_audio,
            supports_text,
            pricing_prompt: p_in,
            pricing_completion: p_out,
        });
    }
    result.sort_by(|a, b| a.model_id.to_lowercase().cmp(&b.model_id.to_lowercase()));
    result
}

pub async fn transcribe_with_audio_model(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    wav_bytes: &[u8],
    instruction: &str,
) -> anyhow::Result<String> {
    let b64 = base64::engine::general_purpose::STANDARD.encode(wav_bytes);
    let payload = serde_json::json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": b64,
                        "format": "wav",
                    },
                },
            ],
        }],
        "temperature": 0.2,
    });
    let url = format!("{OPENROUTER_BASE}/chat/completions");
    let res = client
        .post(url)
        .headers(headers(Some(api_key)))
        .json(&payload)
        .send()
        .await?
        .error_for_status()?;
    let data: Value = res.json().await?;
    Ok(extract_assistant_text(&data))
}

pub async fn chat_text(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    system_prompt: &str,
    user_content: &str,
    reasoning_effort: Option<&str>,
) -> anyhow::Result<String> {
    let max_tok = postprocess_max_completion_tokens(system_prompt, user_content);
    let mut payload = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "max_tokens": max_tok,
    });
    if let Some(eff) = reasoning_effort {
        let t = eff.trim();
        if !t.is_empty() {
            payload
                .as_object_mut()
                .expect("obj")
                .insert("reasoning".to_string(), serde_json::json!({ "effort": t }));
        }
    }
    let url = format!("{OPENROUTER_BASE}/chat/completions");
    let res = client
        .post(url)
        .headers(headers(Some(api_key)))
        .json(&payload)
        .send()
        .await?
        .error_for_status()?;
    let data: Value = res.json().await?;
    Ok(extract_assistant_text(&data))
}

#[must_use]
pub fn extract_assistant_text(data: &Value) -> String {
    let Some(choices) = data.get("choices").and_then(|c| c.as_array()) else {
        return String::new();
    };
    let Some(first) = choices.first() else {
        return String::new();
    };
    let Some(msg) = first.get("message") else {
        return String::new();
    };
    let content = msg.get("content");
    if let Some(s) = content.and_then(|c| c.as_str()) {
        return s.trim().to_string();
    }
    if let Some(arr) = content.and_then(|c| c.as_array()) {
        let mut parts = Vec::new();
        for block in arr {
            if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                if let Some(t) = block.get("text").and_then(|x| x.as_str()) {
                    parts.push(t);
                }
            }
        }
        return parts.join("\n").trim().to_string();
    }
    String::new()
}

pub fn default_transcription_instruction() -> &'static str {
    OPENROUTER_TRANSCRIPTION_INSTRUCTION
}

#[cfg(test)]
mod tests {
    use super::extract_assistant_text;
    use serde_json::json;

    #[test]
    fn extract_assistant_string_content() {
        let data = json!({
            "choices": [{ "message": { "content": "  hello  " } }]
        });
        assert_eq!(extract_assistant_text(&data), "hello");
    }

    #[test]
    fn extract_assistant_array_content() {
        let data = json!({
            "choices": [{
                "message": {
                    "content": [
                        { "type": "text", "text": "a" },
                        { "type": "text", "text": "b" }
                    ]
                }
            }]
        });
        assert_eq!(extract_assistant_text(&data), "a\nb");
    }
}
