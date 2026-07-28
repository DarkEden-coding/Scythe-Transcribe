/** Capture microphone audio while retaining a 16 kHz mono WAV fallback. */

const TARGET_SAMPLE_RATE = 16000;
const ASSEMBLYAI_TERMINATION_TIMEOUT_MS = 15_000;

type AssemblyAiStreamingConfig = {
  keyterms: string[];
};

export type AssemblyAiStreamingResult = {
  completed: boolean;
  transcript: string;
  sessionId: string;
  audioDurationSec: number | null;
  finalizedTurns: number;
};

export type MicRecordingResult = {
  audio: Blob;
  assemblyaiStreaming: AssemblyAiStreamingResult | null;
};

function mergeChunks(chunks: Float32Array[]): Float32Array {
  let total = 0;
  for (const chunk of chunks) total += chunk.length;
  const output = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
}

function resampleLinear(
  data: Float32Array,
  fromRate: number,
  toRate: number,
): Float32Array {
  if (fromRate === toRate) return data;
  const ratio = fromRate / toRate;
  const outputLength = Math.max(1, Math.floor(data.length / ratio));
  const output = new Float32Array(outputLength);
  for (let index = 0; index < outputLength; index++) {
    const sourcePosition = index * ratio;
    const firstIndex = Math.floor(sourcePosition);
    const secondIndex = Math.min(firstIndex + 1, data.length - 1);
    const fraction = sourcePosition - firstIndex;
    output[index] =
      data[firstIndex]! * (1 - fraction) + data[secondIndex]! * fraction;
  }
  return output;
}

function floatTo16BitPcm(samples: Float32Array): Int16Array {
  const output = new Int16Array(samples.length);
  for (let index = 0; index < samples.length; index++) {
    const sample = Math.max(-1, Math.min(1, samples[index]!));
    output[index] = Math.round(sample < 0 ? sample * 0x8000 : sample * 0x7fff);
  }
  return output;
}

function writeWavPcm16(pcm: Int16Array, sampleRate: number): ArrayBuffer {
  const channelCount = 1;
  const bitsPerSample = 16;
  const blockAlign = (channelCount * bitsPerSample) / 8;
  const byteRate = sampleRate * blockAlign;
  const dataSize = pcm.byteLength;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index++) {
      view.setUint8(offset + index, value.charCodeAt(index)!);
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channelCount, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);
  new Int16Array(buffer, 44, pcm.length).set(pcm);
  return buffer;
}

export type MicSession = {
  /** Stop capture, terminate any live stream, and return retained audio plus its result. */
  stop: () => Promise<MicRecordingResult>;
};

async function getPreferredBrowserAudioConstraints(
  preferredDevice: string,
): Promise<MediaStreamConstraints["audio"]> {
  const preferred = preferredDevice.trim().toLowerCase();
  if (
    !preferred ||
    preferred === "__system_default__" ||
    !navigator.mediaDevices?.enumerateDevices
  ) {
    return true;
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  const inputs = devices.filter((device) => device.kind === "audioinput");
  const isBuiltIn = preferred === "__builtin_microphone__";
  const selected = inputs.find((device) => {
    const label = device.label.toLowerCase();
    if (isBuiltIn) {
      return (
        label.includes("built-in microphone") ||
        label.includes("built in microphone") ||
        label.includes("macbook pro microphone") ||
        label.includes("macbook air microphone") ||
        label.includes("imac microphone")
      );
    }
    return label === preferred || label.includes(preferred);
  });

  return selected?.deviceId ? { deviceId: { exact: selected.deviceId } } : true;
}

/** Open an authenticated AssemblyAI stream using a server-minted temporary token. */
async function openAssemblyAiStream(
  config: AssemblyAiStreamingConfig,
): Promise<{
  socket: WebSocket;
  result: Promise<AssemblyAiStreamingResult | null>;
}> {
  const tokenResponse = await fetch("/api/assemblyai/token");
  if (!tokenResponse.ok) throw new Error(await tokenResponse.text());
  const tokenBody = (await tokenResponse.json()) as { token?: string };
  if (!tokenBody.token) throw new Error("AssemblyAI token response omitted token.");

  const params = new URLSearchParams({
    token: tokenBody.token,
    sample_rate: String(TARGET_SAMPLE_RATE),
    encoding: "pcm_s16le",
    speech_model: "universal-3-5-pro",
    mode: "min_latency",
    language_codes: JSON.stringify(["en"]),
    include_partial_turns: "false",
  });
  if (config.keyterms.length > 0) {
    params.set("keyterms_prompt", JSON.stringify(config.keyterms.slice(0, 100)));
  }

  const socket = new WebSocket(`wss://streaming.assemblyai.com/v3/ws?${params}`);
  const finalizedTurns = new Map<number, string>();
  let sessionId = "";
  let resolveResult!: (result: AssemblyAiStreamingResult) => void;
  let rejectResult!: (error: Error) => void;
  let settled = false;
  const result = new Promise<AssemblyAiStreamingResult>((resolve, reject) => {
    resolveResult = resolve;
    rejectResult = reject;
  });

  socket.addEventListener("message", (event) => {
    if (typeof event.data !== "string") return;
    const message = JSON.parse(event.data) as Record<string, unknown>;
    if (message.type === "Begin") {
      sessionId = typeof message.id === "string" ? message.id : "";
      return;
    }
    if (message.type === "Turn" && message.end_of_turn === true) {
      const transcript = typeof message.transcript === "string" ? message.transcript.trim() : "";
      const order = typeof message.turn_order === "number" ? message.turn_order : finalizedTurns.size;
      if (transcript) finalizedTurns.set(order, transcript);
      return;
    }
    if (message.type === "Termination") {
      settled = true;
      const transcript = [...finalizedTurns.entries()]
        .sort(([left], [right]) => left - right)
        .map(([, text]) => text)
        .join(" ");
      resolveResult({
        completed: true,
        transcript,
        sessionId,
        audioDurationSec:
          typeof message.audio_duration_seconds === "number"
            ? message.audio_duration_seconds
            : null,
        finalizedTurns: finalizedTurns.size,
      });
    }
  });
  socket.addEventListener("error", () => {
    if (!settled) rejectResult(new Error("AssemblyAI streaming connection failed."));
  });
  socket.addEventListener("close", (event) => {
    if (!settled) {
      rejectResult(
        new Error(`AssemblyAI stream closed before termination (${event.code}).`),
      );
    }
  });

  await new Promise<void>((resolve, reject) => {
    const onOpen = () => {
      cleanup();
      resolve();
    };
    const onFailure = () => {
      cleanup();
      reject(new Error("Unable to open AssemblyAI streaming connection."));
    };
    const cleanup = () => {
      socket.removeEventListener("open", onOpen);
      socket.removeEventListener("error", onFailure);
    };
    socket.addEventListener("open", onOpen);
    socket.addEventListener("error", onFailure);
  });
  return { socket, result: result.catch(() => null) };
}

/**
 * Start recording from the selected microphone.
 *
 * When AssemblyAI configuration is present, PCM is streamed while the same samples are retained
 * for the recorded fallback. ScriptProcessorNode remains in use for broad embedded-browser support.
 */
export async function startMicRecording(
  preferredDevice = "",
  assemblyaiConfig?: AssemblyAiStreamingConfig,
): Promise<MicSession> {
  const audio = await getPreferredBrowserAudioConstraints(preferredDevice);
  const stream = await navigator.mediaDevices.getUserMedia({ audio });
  const audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
  const assemblyai = assemblyaiConfig
    ? await openAssemblyAiStream(assemblyaiConfig).catch(() => null)
    : null;
  const inputRate = audioContext.sampleRate;
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const chunks: Float32Array[] = [];

  processor.onaudioprocess = (event) => {
    const input = new Float32Array(event.inputBuffer.getChannelData(0));
    chunks.push(input);
    if (assemblyai?.socket.readyState === WebSocket.OPEN) {
      const resampled = resampleLinear(input, inputRate, TARGET_SAMPLE_RATE);
      const pcm = floatTo16BitPcm(resampled);
      const buffer = new ArrayBuffer(pcm.byteLength);
      new Int16Array(buffer).set(pcm);
      assemblyai.socket.send(buffer);
    }
  };

  const gain = audioContext.createGain();
  gain.gain.value = 0;
  source.connect(processor);
  processor.connect(gain);
  gain.connect(audioContext.destination);

  const cleanup = () => {
    processor.onaudioprocess = null;
    processor.disconnect();
    gain.disconnect();
    source.disconnect();
    stream.getTracks().forEach((track) => track.stop());
    void audioContext.close();
  };

  return {
    stop: async () => {
      cleanup();
      const merged = mergeChunks(chunks);
      const resampled = resampleLinear(merged, inputRate, TARGET_SAMPLE_RATE);
      const pcm = floatTo16BitPcm(resampled);
      const wav = writeWavPcm16(pcm, TARGET_SAMPLE_RATE);
      let streamingResult: AssemblyAiStreamingResult | null = null;
      if (assemblyai) {
        try {
          if (assemblyai.socket.readyState === WebSocket.OPEN) {
            assemblyai.socket.send(JSON.stringify({ type: "Terminate" }));
          }
          streamingResult = await Promise.race([
            assemblyai.result,
            new Promise<never>((_, reject) =>
              window.setTimeout(
                () => reject(new Error("AssemblyAI termination timed out.")),
                ASSEMBLYAI_TERMINATION_TIMEOUT_MS,
              ),
            ),
          ]);
        } catch {
          // The retained WAV is submitted by the backend when no completed stream is supplied.
        } finally {
          if (assemblyai.socket.readyState < WebSocket.CLOSING) assemblyai.socket.close();
        }
      }
      return {
        audio: new Blob([wav], { type: "audio/wav" }),
        assemblyaiStreaming: streamingResult,
      };
    },
  };
}
