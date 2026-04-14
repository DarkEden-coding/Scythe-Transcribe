//! Groq API: speech-to-text and chat completions.

use std::collections::HashMap;

use reqwest::multipart;
use serde::Deserialize;
use serde_json::Value;

use crate::config::postprocess_max_completion_tokens;

const NO_SPEECH_THRESHOLD: f64 = 0.6;
const LOGPROB_THRESHOLD: f64 = -1.0;

#[derive(Debug, Clone)]
pub struct GroqTranscriptionResult {
    pub text: String,
    pub silence_detected: bool,
    pub metadata: HashMap<String, serde_json::Value>,
}

fn as_f64(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_i64().map(|i| i as f64))
}

fn segment_summary(seg: &serde_json::Map<String, Value>) -> HashMap<String, serde_json::Value> {
    let text = seg
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let mut m = HashMap::new();
    if let Some(v) = seg.get("id") {
        m.insert("id".to_string(), v.clone());
    }
    for k in [
        "start",
        "end",
        "no_speech_prob",
        "avg_logprob",
        "compression_ratio",
    ] {
        if let Some(v) = seg.get(k) {
            m.insert(k.to_string(), v.clone());
        }
    }
    m.insert("text".to_string(), Value::String(text));
    m
}

fn segment_is_silence(seg: &HashMap<String, serde_json::Value>) -> bool {
    let no_speech = seg.get("no_speech_prob").and_then(as_f64);
    let Some(ns) = no_speech else {
        return false;
    };
    if ns < NO_SPEECH_THRESHOLD {
        return false;
    }
    let avg = seg.get("avg_logprob").and_then(as_f64);
    match avg {
        None => true,
        Some(a) => a <= LOGPROB_THRESHOLD,
    }
}

#[must_use]
pub fn transcription_result_from_json(data: &Value) -> GroqTranscriptionResult {
    let text = data
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    let duration = data.get("duration").and_then(as_f64);
    let language = data
        .get("language")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(std::string::ToString::to_string);

    let mut segments: Vec<HashMap<String, serde_json::Value>> = Vec::new();
    if let Some(arr) = data.get("segments").and_then(|v| v.as_array()) {
        for item in arr {
            if let Some(obj) = item.as_object() {
                segments.push(segment_summary(obj));
            }
        }
    }

    let silence_detected;
    let mut silent_segments = 0usize;
    let mut max_no_speech: Option<f64> = None;
    let mut min_avg_logprob: Option<f64> = None;
    let mut max_compression: Option<f64> = None;

    if !segments.is_empty() {
        silent_segments = segments.iter().filter(|s| segment_is_silence(s)).count();
        silence_detected = silent_segments == segments.len();
        let ns_probs: Vec<f64> = segments
            .iter()
            .filter_map(|s| s.get("no_speech_prob").and_then(as_f64))
            .collect();
        let avgs: Vec<f64> = segments
            .iter()
            .filter_map(|s| s.get("avg_logprob").and_then(as_f64))
            .collect();
        let comps: Vec<f64> = segments
            .iter()
            .filter_map(|s| s.get("compression_ratio").and_then(as_f64))
            .collect();
        if !ns_probs.is_empty() {
            max_no_speech = Some(ns_probs.into_iter().fold(f64::NEG_INFINITY, f64::max));
        }
        if !avgs.is_empty() {
            min_avg_logprob = Some(avgs.into_iter().fold(f64::INFINITY, f64::min));
        }
        if !comps.is_empty() {
            max_compression = Some(comps.into_iter().fold(f64::NEG_INFINITY, f64::max));
        }
    } else {
        silence_detected = text.is_empty();
    }

    let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
    metadata.insert("provider".to_string(), Value::String("groq".to_string()));
    metadata.insert(
        "response_format".to_string(),
        Value::String("verbose_json".to_string()),
    );
    metadata.insert("is_silence".to_string(), Value::Bool(silence_detected));
    if let Some(l) = language {
        metadata.insert("language".to_string(), Value::String(l));
    }
    if let Some(d) = duration {
        metadata.insert("duration".to_string(), serde_json::json!(d));
    }
    metadata.insert(
        "segment_count".to_string(),
        Value::Number(segments.len().into()),
    );
    metadata.insert(
        "silent_segment_count".to_string(),
        Value::Number(silent_segments.into()),
    );
    if let Some(v) = max_no_speech {
        metadata.insert("max_no_speech_prob".to_string(), serde_json::json!(v));
    }
    if let Some(v) = min_avg_logprob {
        metadata.insert("min_avg_logprob".to_string(), serde_json::json!(v));
    }
    if let Some(v) = max_compression {
        metadata.insert("max_compression_ratio".to_string(), serde_json::json!(v));
    }
    if silence_detected {
        metadata.insert("raw_text".to_string(), Value::String(text.clone()));
    }

    GroqTranscriptionResult {
        text,
        silence_detected,
        metadata,
    }
}

pub async fn transcribe_audio(
    client: &reqwest::Client,
    api_key: &str,
    wav_bytes: &[u8],
    model: &str,
    filename: &str,
    prompt: Option<&str>,
) -> anyhow::Result<GroqTranscriptionResult> {
    let part = multipart::Part::bytes(wav_bytes.to_vec())
        .file_name(filename.to_string())
        .mime_str("audio/wav")?;
    let mut form = multipart::Form::new()
        .part("file", part)
        .text("model", model.to_string())
        .text("response_format", "verbose_json")
        .text("temperature", "0.0")
        .text("timestamp_granularities[]", "segment".to_string());
    if let Some(p) = prompt {
        let t = p.trim();
        if !t.is_empty() {
            form = form.text("prompt", t.to_string());
        }
    }

    let url = "https://api.groq.com/openai/v1/audio/transcriptions";
    let res = client
        .post(url)
        .bearer_auth(api_key)
        .multipart(form)
        .send()
        .await?
        .error_for_status()?;
    let v: Value = res.json().await?;
    Ok(transcription_result_from_json(&v))
}

#[derive(Deserialize)]
struct GroqModelsResponse {
    data: Option<Vec<GroqModelId>>,
}

#[derive(Deserialize)]
struct GroqModelId {
    id: String,
}

pub async fn list_chat_models(
    client: &reqwest::Client,
    api_key: &str,
) -> anyhow::Result<Vec<String>> {
    let url = "https://api.groq.com/openai/v1/models";
    let res = client.get(url).bearer_auth(api_key).send().await?;
    if !res.status().is_success() {
        return Ok(vec![]);
    }
    let body: GroqModelsResponse = res
        .json()
        .await
        .unwrap_or(GroqModelsResponse { data: None });
    let mut ids: Vec<String> = body
        .data
        .unwrap_or_default()
        .into_iter()
        .map(|m| m.id)
        .filter(|s| !s.is_empty())
        .collect();
    ids.sort();
    ids.dedup();
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::transcription_result_from_json;
    use serde_json::json;

    #[test]
    fn groq_verbose_json_silence_all_segments() {
        let data = json!({
            "text": "noise",
            "segments": [
                {
                    "text": " ",
                    "no_speech_prob": 0.9,
                    "avg_logprob": -1.5,
                    "compression_ratio": 1.0
                }
            ]
        });
        let r = transcription_result_from_json(&data);
        assert!(r.silence_detected);
        assert_eq!(
            r.metadata
                .get("silent_segment_count")
                .and_then(|v| v.as_u64()),
            Some(1)
        );
    }
}

pub async fn chat_completion(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    system_prompt: &str,
    user_content: &str,
    reasoning_effort: Option<&str>,
) -> anyhow::Result<String> {
    let max_tok = postprocess_max_completion_tokens(system_prompt, user_content);
    let mut body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "max_tokens": max_tok,
        "service_tier": "on_demand",
    });
    if let Some(eff) = reasoning_effort {
        let t = eff.trim();
        if !t.is_empty() {
            body.as_object_mut()
                .expect("object")
                .insert("reasoning_effort".to_string(), Value::String(t.to_string()));
        }
    }
    let url = "https://api.groq.com/openai/v1/chat/completions";
    let res = client
        .post(url)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?
        .error_for_status()?;
    let v: Value = res.json().await?;
    extract_assistant_text(&v)
}

fn extract_assistant_text(data: &Value) -> anyhow::Result<String> {
    let empty = Ok(String::new());
    let Some(choices) = data.get("choices").and_then(|c| c.as_array()) else {
        return empty;
    };
    let Some(first) = choices.first() else {
        return empty;
    };
    let Some(msg) = first.get("message") else {
        return empty;
    };
    let content = msg.get("content");
    if let Some(s) = content.and_then(|c| c.as_str()) {
        return Ok(s.trim().to_string());
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
        return Ok(parts.join("\n").trim().to_string());
    }
    empty
}
