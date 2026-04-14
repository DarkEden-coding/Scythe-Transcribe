/** Capture microphone to 16 kHz mono WAV (matches Python AudioRecorder defaults). */

const TARGET_SAMPLE_RATE = 16000;

function mergeChunks(chunks: Float32Array[]): Float32Array {
  let total = 0;
  for (const c of chunks) total += c.length;
  const out = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

function resampleLinear(
  data: Float32Array,
  fromRate: number,
  toRate: number,
): Float32Array {
  if (fromRate === toRate) {
    return data;
  }
  const ratio = fromRate / toRate;
  const outLen = Math.max(1, Math.floor(data.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcPos = i * ratio;
    const i0 = Math.floor(srcPos);
    const i1 = Math.min(i0 + 1, data.length - 1);
    const t = srcPos - i0;
    out[i] = data[i0]! * (1 - t) + data[i1]! * t;
  }
  return out;
}

function floatTo16BitPcm(samples: Float32Array): Int16Array {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]!));
    out[i] = Math.round(s < 0 ? s * 0x8000 : s * 0x7fff);
  }
  return out;
}

function writeWavPcm16(pcm: Int16Array, sampleRate: number): ArrayBuffer {
  const numChannels = 1;
  const bitsPerSample = 16;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const byteRate = sampleRate * blockAlign;
  const dataSize = pcm.byteLength;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeStr = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) {
      view.setUint8(off + i, s.charCodeAt(i)!);
    }
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  const pcmView = new Int16Array(buffer, 44, pcm.length);
  pcmView.set(pcm);
  return buffer;
}

export type MicSession = {
  /** Stop capture and return WAV bytes as a Blob. */
  stop: () => Promise<Blob>;
};

/**
 * Start recording from the default microphone.
 *
 * Uses ScriptProcessorNode (deprecated but widely supported) to collect PCM.
 */
export async function startMicRecording(): Promise<MicSession> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const audioCtx = new AudioContext();
  const inputRate = audioCtx.sampleRate;
  const source = audioCtx.createMediaStreamSource(stream);
  const bufferSize = 4096;
  const processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
  const chunks: Float32Array[] = [];

  processor.onaudioprocess = (ev) => {
    const input = ev.inputBuffer.getChannelData(0);
    chunks.push(new Float32Array(input));
  };

  const gain = audioCtx.createGain();
  gain.gain.value = 0;
  source.connect(processor);
  processor.connect(gain);
  gain.connect(audioCtx.destination);

  const cleanup = () => {
    processor.disconnect();
    gain.disconnect();
    source.disconnect();
    stream.getTracks().forEach((t) => {
      t.stop();
    });
    void audioCtx.close();
  };

  return {
    stop: async () => {
      const merged = mergeChunks(chunks);
      cleanup();
      const resampled = resampleLinear(merged, inputRate, TARGET_SAMPLE_RATE);
      const pcm = floatTo16BitPcm(resampled);
      const buf = writeWavPcm16(pcm, TARGET_SAMPLE_RATE);
      return new Blob([buf], { type: "audio/wav" });
    },
  };
}
