//! Transcription and optional LLM post-processing.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::StatusCode;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::config::{
    GROQ_ASR_CHUNK_BOUNDARY_SEARCH_SEC, GROQ_ASR_CHUNK_DURATION_SEC, GROQ_ASR_MAX_PARALLEL_CHUNKS,
    GROQ_ASR_MIN_CHUNK_SEC, MAX_UPLOAD_BYTES, POSTPROCESS_CHUNK_MAX_USER_CHARS,
    POSTPROCESS_MAX_PARALLEL_CHUNKS,
};
use crate::groq;
use crate::models::{AppPreferences, ChatProvider, TranscriptionProvider};
use crate::openrouter;
use crate::prompts::{OPENROUTER_TRANSCRIPTION_INSTRUCTION, OPENROUTER_TRANSCRIPTION_NONE_OUTPUT};
use crate::settings_store::{
    append_transcription_history, get_groq_api_key, get_openrouter_api_key,
    patch_transcription_history_entry,
};
use crate::text_replacements::{
    apply_replacements, groq_asr_prompt_from_replacement_spec, parse_replacement_spec,
};

#[derive(Debug, Deserialize, Serialize)]
pub struct TranscribeJob {
    #[serde(default)]
    pub transcription_provider: String,
    #[serde(default)]
    pub transcription_model_groq: String,
    #[serde(default)]
    pub transcription_model_openrouter: String,
    #[serde(default)]
    pub groq_asr_min_audio_chunk_sec: f64,
    #[serde(default)]
    pub openrouter_transcription_instruction: String,
    #[serde(default)]
    pub keyword_replacement_spec: String,
    #[serde(default)]
    pub postprocess_enabled: bool,
    #[serde(default)]
    pub postprocess_prompt: String,
    #[serde(default)]
    pub postprocess_provider: String,
    #[serde(default)]
    pub postprocess_model: String,
    #[serde(default)]
    pub postprocess_groq_reasoning_effort: String,
    #[serde(default)]
    pub postprocess_openrouter_reasoning_effort: String,
}

pub fn transcribe_job_from_preferences(prefs: &AppPreferences) -> TranscribeJob {
    TranscribeJob {
        transcription_provider: prefs.transcription_provider.clone(),
        transcription_model_groq: prefs.transcription_model_groq.clone(),
        transcription_model_openrouter: prefs.transcription_model_openrouter.clone(),
        groq_asr_min_audio_chunk_sec: prefs.groq_asr_min_audio_chunk_sec,
        openrouter_transcription_instruction: prefs.openrouter_transcription_instruction.clone(),
        keyword_replacement_spec: prefs.keyword_replacement_spec.clone(),
        postprocess_enabled: prefs.postprocess_enabled,
        postprocess_prompt: prefs.postprocess_prompt.clone(),
        postprocess_provider: prefs.postprocess_provider.clone(),
        postprocess_model: prefs.postprocess_model.clone(),
        postprocess_groq_reasoning_effort: prefs.postprocess_groq_reasoning_effort.clone(),
        postprocess_openrouter_reasoning_effort: prefs
            .postprocess_openrouter_reasoning_effort
            .clone(),
    }
}

fn effective_groq_asr_min_audio_chunk_sec(raw: f64) -> f64 {
    if raw.is_finite() && raw > 0.0 {
        raw.clamp(1.0, 120.0)
    } else {
        GROQ_ASR_CHUNK_DURATION_SEC
    }
}

fn segment_instruction(index: usize, total: usize) -> String {
    format!(
        "\n\n[Segment {} of {total} of the same transcript. \
         Process only the user's segment below; output only the processed text for this segment.]",
        index + 1
    )
}

#[must_use]
pub fn chunk_transcript_for_postprocess(text: &str, max_chars: usize) -> Vec<String> {
    let mut max_chars = max_chars;
    if max_chars < 256 {
        max_chars = 256;
    }
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }
    let mut chunks: Vec<String> = Vec::new();
    let mut buf = String::new();
    for part in text.split("\n\n") {
        let sep = if buf.is_empty() { "" } else { "\n\n" };
        let candidate = if buf.is_empty() {
            part.to_string()
        } else {
            format!("{buf}{sep}{part}")
        };
        if candidate.len() <= max_chars {
            buf = candidate;
            continue;
        }
        if !buf.is_empty() {
            chunks.push(buf.clone());
            buf.clear();
        }
        if part.len() <= max_chars {
            buf = part.to_string();
            continue;
        }
        let mut offset = 0usize;
        while offset < part.len() {
            let end = (offset + max_chars).min(part.len());
            chunks.push(part[offset..end].to_string());
            offset = end;
        }
    }
    if !buf.is_empty() {
        chunks.push(buf);
    }
    chunks
}

pub async fn postprocess_transcript_text(
    client: &reqwest::Client,
    transcript: &str,
    sys_prompt: &str,
    post_model: &str,
    postprocess_provider: &str,
    groq_reasoning_effort: Option<&str>,
    openrouter_reasoning_effort: Option<&str>,
) -> Result<(String, f64, f64, usize), (StatusCode, String)> {
    let t_ready = std::time::Instant::now();
    let base_sys = sys_prompt.trim();
    let base_sys = if base_sys.is_empty() {
        "You are a helpful assistant."
    } else {
        base_sys
    };
    let model = post_model.trim();
    if model.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Post-process model required when enabled.".to_string(),
        ));
    }

    let chunks = chunk_transcript_for_postprocess(transcript, POSTPROCESS_CHUNK_MAX_USER_CHARS);
    let pprov = if postprocess_provider.is_empty() {
        ChatProvider::Openrouter.as_str()
    } else {
        postprocess_provider
    };

    let t_before_api = std::time::Instant::now();
    let prep_ms = t_before_api
        .saturating_duration_since(t_ready)
        .as_secs_f64()
        * 1000.0;

    let n = chunks.len();
    let max_p = POSTPROCESS_MAX_PARALLEL_CHUNKS.max(1).min(n.max(1));
    let sem = Arc::new(Semaphore::new(max_p));
    let mut results: Vec<Option<String>> = vec![None; n];
    let t_api_start = std::time::Instant::now();
    let mut futs: FuturesUnordered<_> = FuturesUnordered::new();
    for (idx, ch) in chunks.iter().enumerate() {
        let sys_seg = if n == 1 {
            base_sys.to_string()
        } else {
            format!("{base_sys}{}", segment_instruction(idx, n))
        };
        let ch = ch.clone();
        let client = client.clone();
        let model = model.to_string();
        let pprov = pprov.to_string();
        let ge = groq_reasoning_effort.map(str::to_string);
        let oe = openrouter_reasoning_effort.map(str::to_string);
        let sem = sem.clone();
        futs.push(async move {
            let _permit = sem.acquire_owned().await.map_err(|_| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "semaphore closed".to_string(),
                )
            })?;
            let out = run_post_segment(
                &client,
                &pprov,
                &model,
                &sys_seg,
                &ch,
                ge.as_deref(),
                oe.as_deref(),
            )
            .await?;
            Ok::<_, (StatusCode, String)>((idx, out))
        });
    }
    while let Some(item) = futs.next().await {
        let (idx, text) = item?;
        results[idx] = Some(text);
    }
    let api_wall_ms = std::time::Instant::now()
        .saturating_duration_since(t_api_start)
        .as_secs_f64()
        * 1000.0;
    let merged = results
        .into_iter()
        .map(|s| s.unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n\n");
    Ok((merged, prep_ms, api_wall_ms, n))
}

async fn run_post_segment(
    client: &reqwest::Client,
    pprov: &str,
    model: &str,
    sys_seg: &str,
    user_chunk: &str,
    groq_eff: Option<&str>,
    or_eff: Option<&str>,
) -> Result<String, (StatusCode, String)> {
    if pprov == ChatProvider::Groq.as_str() {
        let gkey = get_groq_api_key();
        if gkey.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                "Groq API key not configured.".to_string(),
            ));
        }
        groq::chat_completion(client, &gkey, model, sys_seg, user_chunk, groq_eff)
            .await
            .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))
    } else {
        let okey = get_openrouter_api_key();
        if okey.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                "OpenRouter API key not configured.".to_string(),
            ));
        }
        openrouter::chat_text(client, &okey, model, sys_seg, user_chunk, or_eff)
            .await
            .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))
    }
}

fn wav_duration_seconds(raw: &[u8]) -> Option<f64> {
    let reader = hound::WavReader::new(Cursor::new(raw)).ok()?;
    let sample_rate = reader.spec().sample_rate;
    if sample_rate == 0 {
        return None;
    }
    Some(reader.duration() as f64 / sample_rate as f64)
}

fn merge_groq_chunk_transcriptions(
    chunks: &[groq::GroqTranscriptionResult],
) -> groq::GroqTranscriptionResult {
    let mut text_parts: Vec<String> = Vec::new();
    let mut segment_count = 0usize;
    let mut silent_segment_count = 0usize;
    let mut max_no_speech: Option<f64> = None;
    let mut min_avg_logprob: Option<f64> = None;
    let mut max_compression: Option<f64> = None;
    let mut duration_sum = 0.0f64;
    let mut language: Option<String> = None;

    for c in chunks {
        let t = c.text.trim();
        if !t.is_empty() {
            text_parts.push(t.to_string());
        }
        if language.is_none() {
            if let Some(l) = c.metadata.get("language").and_then(|v| v.as_str()) {
                if !l.is_empty() {
                    language = Some(l.to_string());
                }
            }
        }
        segment_count += c
            .metadata
            .get("segment_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        silent_segment_count += c
            .metadata
            .get("silent_segment_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        if let Some(v) = c
            .metadata
            .get("max_no_speech_prob")
            .and_then(|x| x.as_f64())
        {
            max_no_speech = Some(max_no_speech.map_or(v, |m| m.max(v)));
        }
        if let Some(v) = c.metadata.get("min_avg_logprob").and_then(|x| x.as_f64()) {
            min_avg_logprob = Some(min_avg_logprob.map_or(v, |m| m.min(v)));
        }
        if let Some(v) = c
            .metadata
            .get("max_compression_ratio")
            .and_then(|x| x.as_f64())
        {
            max_compression = Some(max_compression.map_or(v, |m| m.max(v)));
        }
        if let Some(v) = c.metadata.get("duration").and_then(|x| x.as_f64()) {
            duration_sum += v;
        }
    }

    let text = text_parts.join(" ");
    let silence_detected = text.trim().is_empty();

    let mut metadata: HashMap<String, Value> = HashMap::new();
    metadata.insert("provider".to_string(), Value::String("groq".to_string()));
    metadata.insert(
        "response_format".to_string(),
        Value::String("verbose_json".to_string()),
    );
    metadata.insert("is_silence".to_string(), Value::Bool(silence_detected));
    if let Some(l) = language {
        metadata.insert("language".to_string(), Value::String(l));
    }
    if duration_sum > 0.0 {
        metadata.insert("duration".to_string(), json!(duration_sum));
    }
    metadata.insert(
        "segment_count".to_string(),
        Value::Number(segment_count.into()),
    );
    metadata.insert(
        "silent_segment_count".to_string(),
        Value::Number(silent_segment_count.into()),
    );
    if let Some(v) = max_no_speech {
        metadata.insert("max_no_speech_prob".to_string(), json!(v));
    }
    if let Some(v) = min_avg_logprob {
        metadata.insert("min_avg_logprob".to_string(), json!(v));
    }
    if let Some(v) = max_compression {
        metadata.insert("max_compression_ratio".to_string(), json!(v));
    }
    metadata.insert(
        "asr_parallel_chunks".to_string(),
        Value::Number(chunks.len().into()),
    );
    metadata.insert("asr_speech_aware_splits".to_string(), Value::Bool(true));
    if silence_detected {
        metadata.insert("raw_text".to_string(), Value::String(text.clone()));
    }

    groq::GroqTranscriptionResult {
        text,
        silence_detected,
        metadata,
    }
}

#[derive(Clone, Serialize)]
pub struct TranscribeResponse {
    pub transcript: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processed: Option<String>,
    pub silence_detected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asr_metadata: Option<HashMap<String, Value>>,
    pub id: String,
    pub created_at: f64,
    pub transcript_chars: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_duration_sec: Option<f64>,
    pub transcribe_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pre_postprocess_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postprocess_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postprocess_prep_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postprocess_api_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postprocess_chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hotkey_post_api_to_paste_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hotkey_paste_chord_ms: Option<f64>,
    pub total_ms: f64,
}

async fn transcribe_asr_groq(
    client: &reqwest::Client,
    job: &TranscribeJob,
    raw: &[u8],
) -> Result<(String, bool, HashMap<String, Value>), (StatusCode, String)> {
    let key = get_groq_api_key();
    if key.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Groq API key not configured.".to_string(),
        ));
    }
    let model = job.transcription_model_groq.trim();
    let model = if model.is_empty() {
        "whisper-large-v3-turbo"
    } else {
        model
    };
    let whisper_ctx = groq_asr_prompt_from_replacement_spec(&job.keyword_replacement_spec, 1000);
    let audio_chunk_sec = effective_groq_asr_min_audio_chunk_sec(job.groq_asr_min_audio_chunk_sec);

    let chunks: Vec<Vec<u8>> = if audio_chunk_sec > 0.0 {
        crate::wav_speech_chunks::split_wav_for_parallel_groq(
            raw,
            audio_chunk_sec,
            GROQ_ASR_CHUNK_BOUNDARY_SEARCH_SEC,
            audio_chunk_sec.max(GROQ_ASR_MIN_CHUNK_SEC),
        )
        .unwrap_or_else(|_| vec![raw.to_vec()])
    } else {
        vec![raw.to_vec()]
    };

    let groq_result = if chunks.len() <= 1 {
        groq::transcribe_audio(
            client,
            &key,
            raw,
            model,
            "recording.wav",
            whisper_ctx.as_deref(),
        )
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?
    } else {
        let n = chunks.len();
        let max_p = GROQ_ASR_MAX_PARALLEL_CHUNKS.max(1).min(n);
        let sem = Arc::new(Semaphore::new(max_p));
        let asr_model = model.to_string();
        let mut futs: FuturesUnordered<_> = FuturesUnordered::new();
        for (idx, wav) in chunks.into_iter().enumerate() {
            let client = client.clone();
            let key = key.clone();
            let asr_model = asr_model.clone();
            let wc = whisper_ctx.clone();
            let sem = sem.clone();
            futs.push(async move {
                let _permit = sem.acquire_owned().await.map_err(|_| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "semaphore closed".to_string(),
                    )
                })?;
                let r = groq::transcribe_audio(
                    &client,
                    &key,
                    &wav,
                    asr_model.as_str(),
                    &format!("chunk_{idx:03}.wav"),
                    wc.as_deref(),
                )
                .await
                .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
                Ok::<_, (StatusCode, String)>((idx, r))
            });
        }
        let mut slots: Vec<Option<groq::GroqTranscriptionResult>> = vec![None; n];
        while let Some(item) = futs.next().await {
            let (idx, r) = item?;
            slots[idx] = Some(r);
        }
        let ordered: Vec<groq::GroqTranscriptionResult> = slots
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                s.ok_or_else(|| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("missing ASR chunk result {i}"),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        merge_groq_chunk_transcriptions(&ordered)
    };

    let mut meta: HashMap<String, Value> = HashMap::new();
    for (k, v) in &groq_result.metadata {
        meta.insert(k.clone(), v.clone());
    }
    meta.insert("model".to_string(), Value::String(model.to_string()));
    meta.insert("asr_audio_chunk_sec".to_string(), json!(audio_chunk_sec));
    Ok((groq_result.text, groq_result.silence_detected, meta))
}

async fn transcribe_asr_openrouter(
    client: &reqwest::Client,
    job: &TranscribeJob,
    raw: &[u8],
) -> Result<(String, bool, HashMap<String, Value>), (StatusCode, String)> {
    let key = get_openrouter_api_key();
    if key.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "OpenRouter API key not configured.".to_string(),
        ));
    }
    let model = job.transcription_model_openrouter.trim();
    if model.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "OpenRouter transcription model required.".to_string(),
        ));
    }
    let or_instr = job.openrouter_transcription_instruction.trim();
    let or_instr = if or_instr.is_empty() {
        OPENROUTER_TRANSCRIPTION_INSTRUCTION
    } else {
        or_instr
    };
    let t = openrouter::transcribe_with_audio_model(client, &key, model, raw, or_instr)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    let silence = t.text.trim() == OPENROUTER_TRANSCRIPTION_NONE_OUTPUT;
    let mut meta = t.metadata;
    meta.insert("model".to_string(), json!(model));
    meta.insert("response_format".to_string(), json!("chat_completions"));
    meta.insert("is_silence".to_string(), json!(silence));
    Ok((t.text, silence, meta))
}

pub async fn transcribe_wav_bytes(
    client: &reqwest::Client,
    job: &TranscribeJob,
    raw: &[u8],
) -> Result<TranscribeResponse, (StatusCode, String)> {
    if raw.len() > MAX_UPLOAD_BYTES {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "Audio upload too large.".to_string(),
        ));
    }
    if raw.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Empty audio upload.".to_string()));
    }

    let entry_id = uuid::Uuid::new_v4().to_string();
    let audio_duration_sec = wav_duration_seconds(raw);
    let pipeline_start = std::time::Instant::now();
    let provider = TranscriptionProvider::parse(job.transcription_provider.as_str()).ok_or((
        StatusCode::BAD_REQUEST,
        "Unknown transcription provider.".to_string(),
    ))?;

    let t_asr_start = std::time::Instant::now();
    let (transcript, silence_detected, asr_metadata) = match provider {
        TranscriptionProvider::Groq => {
            let (t, s, m) = transcribe_asr_groq(client, job, raw).await?;
            (t, s, Some(m))
        }
        TranscriptionProvider::Openrouter => {
            let (t, s, m) = transcribe_asr_openrouter(client, job, raw).await?;
            (t, s, Some(m))
        }
    };

    let transcribe_ms = std::time::Instant::now()
        .saturating_duration_since(t_asr_start)
        .as_secs_f64()
        * 1000.0;

    let silence_flag = silence_detected;
    let corrected = if silence_flag {
        OPENROUTER_TRANSCRIPTION_NONE_OUTPUT.to_string()
    } else {
        let pairs = parse_replacement_spec(&job.keyword_replacement_spec);
        apply_replacements(&transcript, &pairs)
    };
    let silence_detected = silence_flag;
    let t_transcript_ready = std::time::Instant::now();

    let mut processed: Option<String> = None;
    let mut postprocess_ms: Option<f64> = None;
    let mut pre_postprocess_ms: Option<f64> = None;
    let mut postprocess_prep_ms: Option<f64> = None;
    let mut postprocess_api_ms: Option<f64> = None;
    let mut postprocess_chunks: Option<usize> = None;

    if job.postprocess_enabled && !silence_detected {
        let sys_prompt = job.postprocess_prompt.trim();
        let sys_prompt = if sys_prompt.is_empty() {
            "You are a helpful assistant."
        } else {
            sys_prompt
        };
        let post_model = job.postprocess_model.trim();
        if post_model.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                "Post-process model required when enabled.".to_string(),
            ));
        }
        let pprov = job.postprocess_provider.as_str();
        let groq_eff = job.postprocess_groq_reasoning_effort.trim();
        let groq_eff = if groq_eff.is_empty() {
            None
        } else {
            Some(groq_eff)
        };
        let or_eff = job.postprocess_openrouter_reasoning_effort.trim();
        let or_eff = if or_eff.is_empty() {
            None
        } else {
            Some(or_eff)
        };
        let pre_pp = (std::time::Instant::now().saturating_duration_since(t_transcript_ready))
            .as_secs_f64()
            * 1000.0;
        pre_postprocess_ms = Some(pre_pp);
        let t_pp_start = std::time::Instant::now();
        let (proc, prep, api_ms, n_chunks) = postprocess_transcript_text(
            client, &corrected, sys_prompt, post_model, pprov, groq_eff, or_eff,
        )
        .await?;
        processed = Some(proc);
        postprocess_ms = Some(
            std::time::Instant::now()
                .saturating_duration_since(t_pp_start)
                .as_secs_f64()
                * 1000.0,
        );
        postprocess_prep_ms = Some(prep);
        postprocess_api_ms = Some(api_ms);
        postprocess_chunks = Some(n_chunks);
    }

    let total_ms = std::time::Instant::now()
        .saturating_duration_since(pipeline_start)
        .as_secs_f64()
        * 1000.0;
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let response = TranscribeResponse {
        transcript: corrected.clone(),
        processed,
        silence_detected,
        asr_metadata,
        id: entry_id.clone(),
        created_at,
        transcript_chars: if silence_detected { 0 } else { corrected.len() },
        audio_duration_sec,
        transcribe_ms,
        pre_postprocess_ms,
        postprocess_ms,
        postprocess_prep_ms,
        postprocess_api_ms,
        postprocess_chunks,
        hotkey_post_api_to_paste_ms: None,
        hotkey_paste_chord_ms: None,
        total_ms,
    };

    let v = serde_json::to_value(&response)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let _ = append_transcription_history(v);

    Ok(response)
}

#[must_use]
pub fn text_to_paste(prefs: &AppPreferences, result: &TranscribeResponse) -> String {
    if result.silence_detected || result.transcript.trim() == OPENROUTER_TRANSCRIPTION_NONE_OUTPUT {
        return String::new();
    }
    if prefs.postprocess_enabled {
        if let Some(ref p) = result.processed {
            return p.clone();
        }
    }
    result.transcript.clone()
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct PostprocessOnlyRequest {
    transcript: String,
    postprocess_prompt: String,
    postprocess_model: String,
    postprocess_provider: String,
    postprocess_groq_reasoning_effort: String,
    postprocess_openrouter_reasoning_effort: String,
}

impl Default for PostprocessOnlyRequest {
    fn default() -> Self {
        Self {
            transcript: String::new(),
            postprocess_prompt: "You are a helpful assistant.".to_string(),
            postprocess_model: String::new(),
            postprocess_provider: String::new(),
            postprocess_groq_reasoning_effort: String::new(),
            postprocess_openrouter_reasoning_effort: String::new(),
        }
    }
}

pub async fn postprocess_only_http(
    client: &reqwest::Client,
    body: &Value,
) -> Result<String, (StatusCode, String)> {
    let b: PostprocessOnlyRequest = serde_json::from_value(body.clone()).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Invalid postprocess body: {e}"),
        )
    })?;
    let post_model = b.postprocess_model.trim();
    if post_model.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "postprocess_model required.".to_string(),
        ));
    }
    let pprov = b.postprocess_provider.as_str();
    let pprov = if pprov.is_empty() {
        ChatProvider::Openrouter.as_str()
    } else {
        pprov
    };
    let groq_eff = b.postprocess_groq_reasoning_effort.trim();
    let groq_eff = if groq_eff.is_empty() {
        None
    } else {
        Some(groq_eff)
    };
    let or_eff = b.postprocess_openrouter_reasoning_effort.trim();
    let or_eff = if or_eff.is_empty() {
        None
    } else {
        Some(or_eff)
    };
    let (out, _, _, _) = postprocess_transcript_text(
        client,
        b.transcript.as_str(),
        b.postprocess_prompt.as_str(),
        post_model,
        pprov,
        groq_eff,
        or_eff,
    )
    .await?;
    Ok(out)
}

pub fn patch_history_timings(
    id: &str,
    post_api_to_paste_ms: f64,
    paste_chord_ms: f64,
) -> anyhow::Result<()> {
    let patch = json!({
        "hotkey_post_api_to_paste_ms": post_api_to_paste_ms,
        "hotkey_paste_chord_ms": paste_chord_ms,
    });
    let _ = patch_transcription_history_entry(id, &patch)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::AppPreferences;
    use crate::prompts::OPENROUTER_TRANSCRIPTION_NONE_OUTPUT;

    #[test]
    fn chunk_transcript_splits_paragraphs_and_oversized() {
        let text = "a\n\nb";
        let chunks = chunk_transcript_for_postprocess(text, 100);
        assert_eq!(chunks.len(), 1);
        let long = "x".repeat(5000);
        let chunks = chunk_transcript_for_postprocess(&long, 4500);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn text_to_paste_silence_empty() {
        let prefs = AppPreferences::default();
        let r = TranscribeResponse {
            transcript: "None".to_string(),
            processed: None,
            silence_detected: true,
            asr_metadata: None,
            id: "i".to_string(),
            created_at: 0.0,
            transcript_chars: 0,
            audio_duration_sec: None,
            transcribe_ms: 0.0,
            pre_postprocess_ms: None,
            postprocess_ms: None,
            postprocess_prep_ms: None,
            postprocess_api_ms: None,
            postprocess_chunks: None,
            hotkey_post_api_to_paste_ms: None,
            hotkey_paste_chord_ms: None,
            total_ms: 0.0,
        };
        assert!(text_to_paste(&prefs, &r).is_empty());
        let mut r2 = r.clone();
        r2.silence_detected = false;
        r2.transcript = OPENROUTER_TRANSCRIPTION_NONE_OUTPUT.to_string();
        assert!(text_to_paste(&prefs, &r2).is_empty());
    }
}
