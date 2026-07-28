//! AssemblyAI streaming transcription and recorded-audio fallbacks.

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use anyhow::{anyhow, Context};
use futures::{SinkExt, StreamExt};
use reqwest::header::{HeaderValue, AUTHORIZATION, RETRY_AFTER};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

const STREAMING_ENDPOINT: &str = "wss://streaming.assemblyai.com/v3/ws";
const SYNC_ENDPOINT: &str = "https://sync.assemblyai.com/transcribe";
const REST_ENDPOINT: &str = "https://api.assemblyai.com";
const MODEL: &str = "universal-3-5-pro";
const SYNC_MAX_DURATION_SECONDS: f64 = 120.0;
const STREAM_TERMINATION_TIMEOUT: Duration = Duration::from_secs(15);
const ASYNC_POLL_INTERVAL: Duration = Duration::from_millis(500);
const ASYNC_POLL_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// The finalized output of one AssemblyAI streaming session.
#[derive(Debug, Clone)]
pub struct StreamingResult {
    pub text: String,
    pub session_id: Option<String>,
    pub audio_duration_seconds: Option<f64>,
    pub finalized_turns: usize,
}

/// A completed AssemblyAI transcription plus provider metadata.
#[derive(Debug)]
pub struct TranscriptionResult {
    pub text: String,
    pub metadata: HashMap<String, Value>,
}

enum StreamCommand {
    Audio(Vec<u8>),
    Terminate,
}

/// A callback-safe sender for an active AssemblyAI streaming session.
#[derive(Clone)]
pub struct StreamingAudioSender {
    commands: mpsc::UnboundedSender<StreamCommand>,
}

impl StreamingAudioSender {
    /// Queue one 50–1000 ms PCM16 mono chunk without blocking the audio callback.
    pub fn send_audio(&self, pcm16: Vec<u8>) -> anyhow::Result<()> {
        self.commands
            .send(StreamCommand::Audio(pcm16))
            .map_err(|_| anyhow!("AssemblyAI streaming session closed"))
    }
}

/// An active AssemblyAI WebSocket session that accepts live PCM16 audio.
pub struct StreamingSession {
    commands: mpsc::UnboundedSender<StreamCommand>,
    result: Option<oneshot::Receiver<anyhow::Result<StreamingResult>>>,
    terminated: bool,
}

impl StreamingSession {
    /// Connect to AssemblyAI's edge-routed streaming endpoint.
    pub async fn connect(
        api_key: &str,
        sample_rate: u32,
        keyterms: &[String],
    ) -> anyhow::Result<Self> {
        if api_key.trim().is_empty() {
            return Err(anyhow!("AssemblyAI API key not configured"));
        }
        let url = streaming_url(sample_rate, keyterms)?;
        let mut request = url.into_client_request()?;
        request.headers_mut().insert(
            AUTHORIZATION,
            HeaderValue::from_str(api_key.trim()).context("invalid AssemblyAI API key header")?,
        );
        let (socket, _) = tokio_tungstenite::connect_async(request)
            .await
            .context("connect to AssemblyAI streaming")?;
        let (commands_tx, commands_rx) = mpsc::unbounded_channel();
        let (result_tx, result_rx) = oneshot::channel();
        tokio::spawn(run_stream(socket, commands_rx, result_tx));
        Ok(Self {
            commands: commands_tx,
            result: Some(result_rx),
            terminated: false,
        })
    }

    /// Clone a nonblocking sender suitable for a native audio callback.
    pub fn audio_sender(&self) -> StreamingAudioSender {
        StreamingAudioSender {
            commands: self.commands.clone(),
        }
    }

    /// Gracefully terminate the session and wait for its last finalized turn.
    pub async fn finish(mut self) -> anyhow::Result<StreamingResult> {
        self.terminated = true;
        self.commands
            .send(StreamCommand::Terminate)
            .map_err(|_| anyhow!("AssemblyAI streaming session closed before termination"))?;
        let receiver = self
            .result
            .take()
            .ok_or_else(|| anyhow!("AssemblyAI streaming result already consumed"))?;
        tokio::time::timeout(STREAM_TERMINATION_TIMEOUT, receiver)
            .await
            .map_err(|_| anyhow!("timed out waiting for AssemblyAI termination"))?
            .map_err(|_| anyhow!("AssemblyAI streaming result channel closed"))?
    }
}

impl Drop for StreamingSession {
    fn drop(&mut self) {
        if !self.terminated {
            let _ = self.commands.send(StreamCommand::Terminate);
        }
    }
}

/// Build the current Universal-3.5 Pro streaming URL and its low-latency options.
fn streaming_url(sample_rate: u32, keyterms: &[String]) -> anyhow::Result<String> {
    let mut url = reqwest::Url::parse(STREAMING_ENDPOINT)?;
    {
        let mut query = url.query_pairs_mut();
        query.append_pair("sample_rate", &sample_rate.to_string());
        query.append_pair("encoding", "pcm_s16le");
        query.append_pair("speech_model", MODEL);
        query.append_pair("mode", "min_latency");
        query.append_pair("language_codes", "[\"en\"]");
        query.append_pair("include_partial_turns", "false");
        if !keyterms.is_empty() {
            query.append_pair("keyterms_prompt", &serde_json::to_string(keyterms)?);
        }
    }
    Ok(url.into())
}

/// Drive the WebSocket until graceful termination and collect only finalized turns.
async fn run_stream<S>(
    socket: tokio_tungstenite::WebSocketStream<S>,
    mut commands: mpsc::UnboundedReceiver<StreamCommand>,
    result_tx: oneshot::Sender<anyhow::Result<StreamingResult>>,
) where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let (mut writer, mut reader) = socket.split();
    let mut turns = BTreeMap::<i64, String>::new();
    let mut session_id = None;
    let outcome = loop {
        tokio::select! {
            command = commands.recv() => {
                match command {
                    Some(StreamCommand::Audio(bytes)) => {
                        if let Err(error) = writer.send(Message::Binary(bytes.into())).await {
                            break Err(anyhow!("send AssemblyAI audio: {error}"));
                        }
                    }
                    Some(StreamCommand::Terminate) | None => {
                        if let Err(error) = writer.send(Message::Text(json!({"type": "Terminate"}).to_string().into())).await {
                            break Err(anyhow!("terminate AssemblyAI stream: {error}"));
                        }
                    }
                }
            }
            message = reader.next() => {
                match message {
                    Some(Ok(Message::Text(text))) => {
                        let event: Value = match serde_json::from_str(text.as_ref()) {
                            Ok(event) => event,
                            Err(error) => break Err(anyhow!("parse AssemblyAI event: {error}")),
                        };
                        match event.get("type").and_then(Value::as_str) {
                            Some("Begin") => {
                                session_id = event.get("id").and_then(Value::as_str).map(str::to_string);
                            }
                            Some("Turn") if event.get("end_of_turn").and_then(Value::as_bool) == Some(true) => {
                                let order = event.get("turn_order").and_then(Value::as_i64).unwrap_or(turns.len() as i64);
                                let text = event.get("transcript").and_then(Value::as_str).unwrap_or("").trim();
                                if !text.is_empty() {
                                    turns.insert(order, text.to_string());
                                }
                            }
                            Some("Termination") => {
                                let audio_duration_seconds = event.get("audio_duration_seconds").and_then(Value::as_f64);
                                let text = turns.values().cloned().collect::<Vec<_>>().join(" ");
                                break Ok(StreamingResult {
                                    text,
                                    session_id,
                                    audio_duration_seconds,
                                    finalized_turns: turns.len(),
                                });
                            }
                            Some("Error") => {
                                let message = event.get("error").or_else(|| event.get("message")).and_then(Value::as_str).unwrap_or("unknown streaming error");
                                break Err(anyhow!("AssemblyAI streaming error: {message}"));
                            }
                            _ => {}
                        }
                    }
                    Some(Ok(Message::Close(frame))) => {
                        break Err(anyhow!("AssemblyAI stream closed before termination: {frame:?}"));
                    }
                    Some(Ok(_)) => {}
                    Some(Err(error)) => break Err(anyhow!("receive AssemblyAI event: {error}")),
                    None => break Err(anyhow!("AssemblyAI stream ended before termination")),
                }
            }
        }
    };
    let _ = writer.close().await;
    let _ = result_tx.send(outcome);
}

/// Transcribe a retained WAV after streaming failed, selecting sync or async by duration.
pub async fn transcribe_recorded_fallback(
    client: &reqwest::Client,
    api_key: &str,
    wav_bytes: &[u8],
    duration_seconds: Option<f64>,
    keyterms: &[String],
) -> anyhow::Result<TranscriptionResult> {
    if duration_seconds.is_some_and(|seconds| seconds <= SYNC_MAX_DURATION_SECONDS) {
        match transcribe_sync(client, api_key, wav_bytes, keyterms).await {
            Ok(result) => return Ok(result),
            Err(error) => tracing::warn!("AssemblyAI sync fallback failed; trying async: {error}"),
        }
    }
    transcribe_async(client, api_key, wav_bytes, keyterms).await
}

/// Transcribe a short WAV through AssemblyAI Sync STT.
async fn transcribe_sync(
    client: &reqwest::Client,
    api_key: &str,
    wav_bytes: &[u8],
    keyterms: &[String],
) -> anyhow::Result<TranscriptionResult> {
    let audio = reqwest::multipart::Part::bytes(wav_bytes.to_vec())
        .file_name("recording.wav")
        .mime_str("audio/wav")?;
    let config = json!({
        "language_code": "en",
        "prompt": "English single-speaker dictation.",
        "keyterms_prompt": keyterms,
    });
    let form = reqwest::multipart::Form::new().part("audio", audio).part(
        "config",
        reqwest::multipart::Part::text(config.to_string()).mime_str("application/json")?,
    );
    let response = client
        .post(SYNC_ENDPOINT)
        .header(AUTHORIZATION, api_key)
        .header("X-AAI-Model", MODEL)
        .multipart(form)
        .send()
        .await
        .context("submit AssemblyAI sync transcription")?;
    let status = response.status();
    let body: Value = response
        .json()
        .await
        .context("parse AssemblyAI sync response")?;
    if !status.is_success() {
        return Err(anyhow!(
            "AssemblyAI sync returned {status}: {}",
            api_error(&body)
        ));
    }
    let text = body
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let mut metadata = HashMap::new();
    metadata.insert("provider".to_string(), json!("assemblyai"));
    metadata.insert("mode".to_string(), json!("sync_fallback"));
    metadata.insert("model".to_string(), json!(MODEL));
    copy_metadata(
        &body,
        &mut metadata,
        &[
            "session_id",
            "confidence",
            "audio_duration_ms",
            "request_time_ms",
        ],
    );
    Ok(TranscriptionResult { text, metadata })
}

/// Upload and poll a long WAV through AssemblyAI's pre-recorded API.
async fn transcribe_async(
    client: &reqwest::Client,
    api_key: &str,
    wav_bytes: &[u8],
    keyterms: &[String],
) -> anyhow::Result<TranscriptionResult> {
    let upload_response = client
        .post(format!("{REST_ENDPOINT}/v2/upload"))
        .header(AUTHORIZATION, api_key)
        .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
        .body(wav_bytes.to_vec())
        .send()
        .await
        .context("upload audio to AssemblyAI")?;
    let upload_status = upload_response.status();
    let upload_body: Value = upload_response
        .json()
        .await
        .context("parse AssemblyAI upload response")?;
    if !upload_status.is_success() {
        return Err(anyhow!(
            "AssemblyAI upload returned {upload_status}: {}",
            api_error(&upload_body)
        ));
    }
    let upload_url = upload_body
        .get("upload_url")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("AssemblyAI upload response omitted upload_url"))?;
    let submit_body = json!({
        "audio_url": upload_url,
        "speech_models": [MODEL, "universal-2"],
        "language_code": "en",
        "prompt": "English single-speaker dictation.",
        "keyterms_prompt": keyterms,
    });
    let submit_response = client
        .post(format!("{REST_ENDPOINT}/v2/transcript"))
        .header(AUTHORIZATION, api_key)
        .json(&submit_body)
        .send()
        .await
        .context("submit AssemblyAI async transcription")?;
    let submit_status = submit_response.status();
    let submit_json: Value = submit_response
        .json()
        .await
        .context("parse AssemblyAI submit response")?;
    if !submit_status.is_success() {
        return Err(anyhow!(
            "AssemblyAI submit returned {submit_status}: {}",
            api_error(&submit_json)
        ));
    }
    let transcript_id = submit_json
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("AssemblyAI submit response omitted transcript id"))?
        .to_string();
    let deadline = tokio::time::Instant::now() + ASYNC_POLL_TIMEOUT;
    loop {
        if tokio::time::Instant::now() >= deadline {
            return Err(anyhow!(
                "timed out polling AssemblyAI transcript {transcript_id}"
            ));
        }
        let response = client
            .get(format!("{REST_ENDPOINT}/v2/transcript/{transcript_id}"))
            .header(AUTHORIZATION, api_key)
            .send()
            .await
            .context("poll AssemblyAI transcript")?;
        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let delay = retry_after(&response).unwrap_or(Duration::from_secs(1));
            tokio::time::sleep(delay).await;
            continue;
        }
        let status = response.status();
        let body: Value = response
            .json()
            .await
            .context("parse AssemblyAI poll response")?;
        if !status.is_success() {
            return Err(anyhow!(
                "AssemblyAI poll returned {status}: {}",
                api_error(&body)
            ));
        }
        match body.get("status").and_then(Value::as_str) {
            Some("completed") => {
                let text = body
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let mut metadata = HashMap::new();
                metadata.insert("provider".to_string(), json!("assemblyai"));
                metadata.insert("mode".to_string(), json!("async_fallback"));
                metadata.insert("model".to_string(), json!(MODEL));
                metadata.insert("transcript_id".to_string(), json!(transcript_id));
                copy_metadata(
                    &body,
                    &mut metadata,
                    &["speech_model_used", "confidence", "audio_duration"],
                );
                return Ok(TranscriptionResult { text, metadata });
            }
            Some("error") => {
                return Err(anyhow!(
                    "AssemblyAI transcript failed: {}",
                    api_error(&body)
                ))
            }
            _ => tokio::time::sleep(ASYNC_POLL_INTERVAL).await,
        }
    }
}

/// Return AssemblyAI's useful error text without losing nonstandard responses.
fn api_error(body: &Value) -> String {
    body.get("error")
        .or_else(|| body.get("message"))
        .or_else(|| body.get("detail"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| body.to_string())
}

/// Copy selected response values into transcription history metadata.
fn copy_metadata(body: &Value, metadata: &mut HashMap<String, Value>, keys: &[&str]) {
    for key in keys {
        if let Some(value) = body.get(*key) {
            metadata.insert((*key).to_string(), value.clone());
        }
    }
}

/// Parse a standard Retry-After seconds header.
fn retry_after(response: &reqwest::Response) -> Option<Duration> {
    let seconds = response
        .headers()
        .get(RETRY_AFTER)?
        .to_str()
        .ok()?
        .parse()
        .ok()?;
    Some(Duration::from_secs(seconds))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_url_sets_low_latency_english_and_keyterms() {
        let url = streaming_url(16_000, &["Scythe".to_string()]).unwrap();
        let parsed = reqwest::Url::parse(&url).unwrap();
        let query = parsed.query_pairs().collect::<HashMap<_, _>>();
        assert_eq!(query.get("speech_model").map(|v| v.as_ref()), Some(MODEL));
        assert_eq!(query.get("mode").map(|v| v.as_ref()), Some("min_latency"));
        assert_eq!(
            query.get("language_codes").map(|v| v.as_ref()),
            Some("[\"en\"]")
        );
        assert_eq!(
            query.get("include_partial_turns").map(|v| v.as_ref()),
            Some("false")
        );
        assert_eq!(
            query.get("keyterms_prompt").map(|v| v.as_ref()),
            Some("[\"Scythe\"]")
        );
    }

    #[test]
    fn api_error_accepts_problem_details() {
        assert_eq!(api_error(&json!({"detail": "bad audio"})), "bad audio");
    }
}
