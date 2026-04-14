import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent, PointerEvent } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { startMicRecording } from "./audio";
import { OpenRouterModelPicker, type OrModel } from "./OpenRouterModelPicker";

type TabId = "general" | "keys" | "transcribe" | "postprocess" | "stats" | "output";
type RuntimeIconState = "idle" | "recording" | "processing";
type RuntimeIconStatus = {
  base_state: RuntimeIconState;
  override_state: RuntimeIconState | null;
  display_state: RuntimeIconState;
};
type HotkeyDiagnostics = {
  state?: string;
  error?: string | null;
  accessibility_trusted?: boolean;
  input_monitoring_trusted?: boolean;
  microphone_authorized?: boolean;
  microphone_authorization?: string;
  capture_state?: RuntimeIconState | string;
  configured_combo?: string;
  combo_parts?: string[];
  pressed_tokens?: string[];
  combo_active?: boolean;
  last_event?: string | null;
  last_token?: string | null;
  last_key?: string | null;
  last_event_at?: number | null;
  last_combo_matched_at?: number | null;
  last_recording_started_at?: number | null;
  last_recording_stopped_at?: number | null;
  last_stream_error?: string | null;
  last_stream_error_at?: number | null;
  last_transcribe_error?: string | null;
  last_transcribe_error_at?: number | null;
  event_count?: number;
  listener_backend?: string;
};
type RuntimeState = {
  icon_state: RuntimeIconState;
  capture_state: RuntimeIconState;
  capturing_audio: boolean;
  processing_audio?: boolean;
  os_icon: RuntimeIconStatus;
  hotkey?: HotkeyDiagnostics;
};

const TABS: { id: TabId; label: string }[] = [
  { id: "general", label: "General" },
  { id: "keys", label: "API keys" },
  { id: "transcribe", label: "Transcribe" },
  { id: "postprocess", label: "Post-process" },
  { id: "stats", label: "Stats" },
  { id: "output", label: "Output" },
];

const GROQ_STT_DEFAULTS = [
  "whisper-large-v3",
  "whisper-large-v3-turbo",
  "distil-whisper-large-v3-en",
] as const;

/** Groq ``reasoning_effort`` (model-dependent); empty = API default. */
const GROQ_POST_REASONING_EFFORTS = ["", "none", "default", "low", "medium", "high"] as const;

/** OpenRouter ``reasoning.effort``; empty = omit. */
const OR_POST_REASONING_EFFORTS = ["", "xhigh", "high", "medium", "low", "minimal", "none"] as const;

type AppPreferences = {
  audio_input_device: string;
  transcription_provider: string;
  transcription_model_groq: string;
  transcription_model_openrouter: string;
  postprocess_enabled: boolean;
  postprocess_prompt: string;
  postprocess_provider: string;
  postprocess_model: string;
  postprocess_groq_reasoning_effort: string;
  postprocess_openrouter_reasoning_effort: string;
  openrouter_models_cache_hint: string;
  keyword_replacement_spec: string;
  openrouter_transcription_instruction: string;
  hotkey_toggle_recording: string;
};

type KeysPublic = {
  groq_configured: boolean;
  openrouter_configured: boolean;
};

type AudioInputDevice = {
  id: string;
  name: string;
  is_default: boolean;
  is_builtin_candidate: boolean;
};

type AudioInputDevicesResponse = {
  builtin_id: string;
  system_default_id: string;
  devices: AudioInputDevice[];
};

type AccessibilityIdentity = {
  app_bundle?: string | null;
  executable?: string | null;
  pid?: number;
};

type TranscriptionHistoryEntry = {
  id: string;
  createdAt: number;
  transcript: string;
  processed: string;
  silenceDetected: boolean;
  transcriptChars: number;
  audioDurationSec: number | null;
  transcribeMs: number;
  postprocessMs: number | null;
  prePostprocessMs: number | null;
  postprocessPrepMs: number | null;
  postprocessApiMs: number | null;
  postprocessChunks: number | null;
  hotkeyPostApiToPasteMs: number | null;
  hotkeyPasteChordMs: number | null;
  totalMs: number;
  asrMetadata: Record<string, unknown> | null;
};

function numOrNull(v: unknown): number | null {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function objectOrNull(v: unknown): Record<string, unknown> | null {
  if (v && typeof v === "object" && !Array.isArray(v)) {
    return v as Record<string, unknown>;
  }
  return null;
}

function mapHistoryEntry(raw: Record<string, unknown>): TranscriptionHistoryEntry {
  const createdRaw = raw.created_at ?? raw.createdAt;
  const createdAt =
    typeof createdRaw === "number"
      ? createdRaw
      : typeof createdRaw === "string"
        ? Number(createdRaw)
        : Date.now();
  const idRaw = raw.id;
  const id =
    typeof idRaw === "string" && idRaw.length > 0 ? idRaw : `legacy-${createdAt}`;
  const ppRaw = raw.postprocess_ms ?? raw.postprocessMs;
  const asrMetadata = objectOrNull(raw.asr_metadata ?? raw.asrMetadata);
  return {
    id,
    createdAt: Number.isFinite(createdAt) ? createdAt : Date.now(),
    transcript: String(raw.transcript ?? ""),
    processed: String(raw.processed ?? ""),
    silenceDetected: Boolean(raw.silence_detected ?? raw.silenceDetected ?? false),
    transcriptChars: Number(raw.transcript_chars ?? raw.transcriptChars ?? 0),
    audioDurationSec:
      numOrNull(raw.audio_duration_sec ?? raw.audioDurationSec) ??
      numOrNull(asrMetadata?.duration),
    transcribeMs: Number(raw.transcribe_ms ?? raw.transcribeMs ?? 0),
    postprocessMs:
      ppRaw === null || ppRaw === undefined ? null : Number(ppRaw),
    prePostprocessMs: numOrNull(raw.pre_postprocess_ms ?? raw.prePostprocessMs),
    postprocessPrepMs: numOrNull(raw.postprocess_prep_ms ?? raw.postprocessPrepMs),
    postprocessApiMs: numOrNull(raw.postprocess_api_ms ?? raw.postprocessApiMs),
    postprocessChunks: numOrNull(raw.postprocess_chunks ?? raw.postprocessChunks),
    hotkeyPostApiToPasteMs: numOrNull(
      raw.hotkey_post_api_to_paste_ms ?? raw.hotkeyPostApiToPasteMs,
    ),
    hotkeyPasteChordMs: numOrNull(raw.hotkey_paste_chord_ms ?? raw.hotkeyPasteChordMs),
    totalMs: Number(raw.total_ms ?? raw.totalMs ?? 0),
    asrMetadata,
  };
}

function formatDurationMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) {
    return "—";
  }
  if (ms < 1000) {
    return `${Math.round(ms)} ms`;
  }
  return `${(ms / 1000).toFixed(2)} s`;
}

function historyTimingSecondaryLine(e: TranscriptionHistoryEntry): string | null {
  const parts: string[] = [];
  if (e.transcriptChars > 0) {
    parts.push(`${e.transcriptChars.toLocaleString()} chars`);
  }
  if (e.postprocessMs != null && Number.isFinite(e.postprocessMs)) {
    if (e.prePostprocessMs != null) {
      parts.push(`before LLM ${formatDurationMs(e.prePostprocessMs)}`);
    }
    if (e.postprocessPrepMs != null) {
      parts.push(`chunk prep ${formatDurationMs(e.postprocessPrepMs)}`);
    }
    if (e.postprocessApiMs != null) {
      parts.push(`API wall ${formatDurationMs(e.postprocessApiMs)}`);
    }
    if (e.postprocessChunks != null && e.postprocessChunks >= 1) {
      parts.push(
        e.postprocessChunks === 1 ? "1 chunk" : `${e.postprocessChunks} chunks`,
      );
    }
  }
  if (parts.length === 0) return null;
  return parts.join(" · ");
}

function historyTimingHotkeyLine(e: TranscriptionHistoryEntry): string | null {
  if (e.hotkeyPostApiToPasteMs == null && e.hotkeyPasteChordMs == null) return null;
  const p: string[] = [];
  if (e.hotkeyPostApiToPasteMs != null) {
    p.push(`after API → paste ${formatDurationMs(e.hotkeyPostApiToPasteMs)}`);
  }
  if (e.hotkeyPasteChordMs != null) {
    p.push(`paste chord ${formatDurationMs(e.hotkeyPasteChordMs)}`);
  }
  return p.length ? `Hotkey: ${p.join(" · ")}` : null;
}

type StatsSummary = {
  transcriptionCount: number;
  timedEntryCount: number;
  totalAudioSec: number;
  totalChars: number;
  totalWords: number;
  totalTranscribeMs: number;
  totalCostUsd: number | null;
  costEntryCount: number;
  firstAt: number | null;
  lastAt: number | null;
  averageWordsPerEntry: number;
  averageSpeechWpm: number | null;
  apiThroughputWpm: number | null;
  averageApiSeconds: number | null;
  longestAudioSec: number | null;
  recent: {
    id: string;
    label: string;
    words: number;
    audioSec: number | null;
    wpm: number | null;
  }[];
};

function countWords(text: string): number {
  const matches = text.trim().match(/[\p{L}\p{N}]+(?:['’.-][\p{L}\p{N}]+)*/gu);
  return matches?.length ?? 0;
}

function formatNumber(n: number): string {
  return Number.isFinite(n) ? Math.round(n).toLocaleString() : "—";
}

function formatDecimal(n: number | null, digits = 1): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatAudioDuration(seconds: number | null): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds <= 0) return "—";
  const minutes = seconds / 60;
  if (minutes < 60) return `${formatDecimal(minutes, minutes < 10 ? 1 : 0)} min`;
  const hours = minutes / 60;
  return `${formatDecimal(hours, hours < 10 ? 2 : 1)} hr`;
}

function formatAudioDurationDetail(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0 min · 0 hr";
  const minutes = seconds / 60;
  const hours = minutes / 60;
  return `${formatDecimal(minutes, minutes < 10 ? 1 : 0)} min · ${formatDecimal(hours, 2)} hr`;
}

function formatCostUsd(cost: number | null): string {
  if (cost == null || !Number.isFinite(cost)) return "Unavailable";
  if (cost === 0) return "$0.00";
  if (cost < 0.01) return `$${cost.toFixed(4)}`;
  if (cost < 1) return `$${cost.toFixed(3)}`;
  return cost.toLocaleString(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  });
}

function costNumber(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") {
    const n = Number(v.replace(/[$,]/g, "").trim());
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function costFromObject(obj: Record<string, unknown> | null): number | null {
  if (!obj) return null;
  for (const key of ["total_cost", "totalCost", "cost", "total_price", "totalPrice"]) {
    const n = costNumber(obj[key]);
    if (n != null) return n;
  }
  const usage = objectOrNull(obj.usage);
  if (usage) {
    for (const key of ["total_cost", "totalCost", "cost", "total_price", "totalPrice"]) {
      const n = costNumber(usage[key]);
      if (n != null) return n;
    }
  }
  return null;
}

function computeStats(entries: TranscriptionHistoryEntry[]): StatsSummary {
  let totalAudioSec = 0;
  let totalChars = 0;
  let totalWords = 0;
  let timedWords = 0;
  let timedEntryCount = 0;
  let totalTranscribeMs = 0;
  let totalCostUsd = 0;
  let costEntryCount = 0;
  let firstAt: number | null = null;
  let lastAt: number | null = null;
  let longestAudioSec: number | null = null;

  const recent = entries.slice(0, 8).map((entry) => {
    const text = entry.silenceDetected ? "" : entry.transcript;
    const words = countWords(text);
    const audioSec = entry.audioDurationSec;
    const wpm =
      audioSec != null && audioSec > 0 && words > 0 ? words / (audioSec / 60) : null;
    return {
      id: entry.id,
      label: new Date(entry.createdAt).toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
      }),
      words,
      audioSec,
      wpm,
    };
  });

  for (const entry of entries) {
    const text = entry.silenceDetected ? "" : entry.transcript;
    const chars = entry.transcriptChars || text.length;
    const words = countWords(text);
    totalChars += chars;
    totalWords += words;
    if (entry.audioDurationSec != null && entry.audioDurationSec > 0) {
      totalAudioSec += entry.audioDurationSec;
      timedWords += words;
      timedEntryCount += 1;
      longestAudioSec =
        longestAudioSec == null
          ? entry.audioDurationSec
          : Math.max(longestAudioSec, entry.audioDurationSec);
    }
    if (Number.isFinite(entry.transcribeMs) && entry.transcribeMs > 0) {
      totalTranscribeMs += entry.transcribeMs;
    }
    const cost = costFromObject(entry.asrMetadata);
    if (cost != null) {
      totalCostUsd += cost;
      costEntryCount += 1;
    }
    firstAt = firstAt == null ? entry.createdAt : Math.min(firstAt, entry.createdAt);
    lastAt = lastAt == null ? entry.createdAt : Math.max(lastAt, entry.createdAt);
  }

  const transcriptionCount = entries.length;
  return {
    transcriptionCount,
    timedEntryCount,
    totalAudioSec,
    totalChars,
    totalWords,
    totalTranscribeMs,
    totalCostUsd: costEntryCount > 0 ? totalCostUsd : null,
    costEntryCount,
    firstAt,
    lastAt,
    averageWordsPerEntry: transcriptionCount > 0 ? totalWords / transcriptionCount : 0,
    averageSpeechWpm:
      totalAudioSec > 0 && timedWords > 0 ? timedWords / (totalAudioSec / 60) : null,
    apiThroughputWpm:
      totalTranscribeMs > 0 && totalWords > 0 ? totalWords / (totalTranscribeMs / 60000) : null,
    averageApiSeconds:
      transcriptionCount > 0 && totalTranscribeMs > 0
        ? totalTranscribeMs / transcriptionCount / 1000
        : null,
    longestAudioSec,
    recent,
  };
}

const defaultPrefs = (): AppPreferences => ({
  audio_input_device: "__builtin_microphone__",
  transcription_provider: "groq",
  transcription_model_groq: "whisper-large-v3-turbo",
  transcription_model_openrouter: "",
  postprocess_enabled: false,
  postprocess_prompt: "Summarize the transcript in bullet points.",
  postprocess_provider: "openrouter",
  postprocess_model: "openai/gpt-4o-mini",
  postprocess_groq_reasoning_effort: "",
  postprocess_openrouter_reasoning_effort: "",
  openrouter_models_cache_hint: "",
  keyword_replacement_spec: "",
  openrouter_transcription_instruction: "",
  hotkey_toggle_recording: "ctrl+shift+space",
});

const MODIFIER_KEYS = new Set([
  "ctrl",
  "alt",
  "shift",
  "meta",
  "control",
  "os",
  "super",
]);

/** Normalizes `e.key` for the non-modifier slot (Win/OS/Super → meta for consistency). */
function normalizeKeySlot(e: KeyboardEvent): string {
  // Use the physical key code for space and function keys so that macOS modifier
  // behavior doesn't corrupt the token:
  // - Option+Space produces e.key = "\u00a0" instead of " ".
  // - Option+Fn may produce a non-standard e.key for some function keys.
  // e.code is "Space" / "F1"..."F20" regardless of held modifiers.
  if (e.code === "Space" || e.key === " " || e.key === "\u00a0") return "space";
  const fnMatch = /^F(\d{1,2})$/.exec(e.code ?? "");
  if (fnMatch) return "f" + fnMatch[1];
  const lower = e.key.toLowerCase();
  if (lower === "os" || lower === "super") return "meta";
  return lower;
}

function normalizeHotkeyFromEvent(e: KeyboardEvent): string {
  const parts: string[] = [];
  if (e.ctrlKey) parts.push("ctrl");
  if (e.altKey) parts.push("alt");
  if (e.shiftKey) parts.push("shift");
  if (e.metaKey) parts.push("meta");

  const keySlot = normalizeKeySlot(e);
  const isDuplicateModifierSlot =
    (keySlot === "meta" && e.metaKey) ||
    (keySlot === "control" && e.ctrlKey) ||
    (keySlot === "shift" && e.shiftKey) ||
    (keySlot === "alt" && e.altKey);
  if (!isDuplicateModifierSlot) {
    parts.push(keySlot);
  }
  return parts.join("+");
}

function isOnlyModifiersCombo(combo: string): boolean {
  return combo.split("+").every((p) => MODIFIER_KEYS.has(p));
}

function isApplePlatform(): boolean {
  return /Mac|iPhone|iPad|iPod/i.test(
    typeof navigator !== "undefined" ? navigator.platform : "",
  );
}

function hotkeyConflictWarning(combo: string): string | null {
  const parts = combo
    .split("+")
    .map((p) => p.trim().toLowerCase())
    .filter(Boolean);
  if (!parts.length || !isApplePlatform()) return null;
  const joined = parts.join("+");
  if (joined === "meta+space") {
    return "Cmd+Space is Spotlight on macOS by default, so the OS usually consumes it before Scythe can use it. Disable/remap Spotlight or choose another shortcut.";
  }
  if (joined === "ctrl+space") {
    return "Ctrl+Space is commonly used by macOS Input Source switching, which can steal the shortcut before Scythe sees it.";
  }
  if (joined === "meta+tab") {
    return "Cmd+Tab is the macOS app switcher and is effectively unavailable as a Scythe hotkey.";
  }
  return null;
}

function formatHotkeyLabel(combo: string): string {
  const t = combo.trim();
  if (!t) return "None";
  const metaName = isApplePlatform() ? "Cmd" : "Win";
  return t
    .split("+")
    .map((p) => {
      if (p === "ctrl") return "Ctrl";
      if (p === "meta" || p === "os" || p === "super") return metaName;
      if (p === "alt") return "Alt";
      if (p === "shift") return "Shift";
      if (p === "space") return "Space";
      if (p.length === 1) return p.toUpperCase();
      return p.charAt(0).toUpperCase() + p.slice(1);
    })
    .join("+");
}

function runtimeIconSrc(state: RuntimeIconState): string {
  if (state === "recording") return "/icon-red.webp";
  if (state === "processing") return "/icon-yellow.webp";
  return "/icon-blue.webp";
}

function runtimeIconColor(state: RuntimeIconState): string {
  if (state === "recording") return "#e53935";
  if (state === "processing") return "#ffb300";
  return "#42a5f5";
}

function parseRuntimeIconOverride(value: string): RuntimeIconState | null {
  if (value === "idle" || value === "recording" || value === "processing") {
    return value;
  }
  return null;
}

function formatDiagnosticTime(seconds?: number | null): string {
  if (seconds == null || !Number.isFinite(seconds)) return "Never";
  return new Date(seconds * 1000).toLocaleTimeString();
}

function formatDiagnosticList(values?: string[]): string {
  return values && values.length > 0 ? values.join("+") : "None";
}

function formatDiagnosticBool(value?: boolean): string {
  if (value === undefined) return "Unknown";
  return value ? "Yes" : "No";
}

function preventModifiedButtonActivation(e: ReactKeyboardEvent<HTMLButtonElement>) {
  if (
    (e.key === " " || e.key === "Enter" || e.code === "Space" || e.code === "Enter") &&
    (e.ctrlKey || e.altKey || e.shiftKey || e.metaKey)
  ) {
    e.preventDefault();
    e.stopPropagation();
  }
}

/** Matches `text_replacements.parse_replacement_spec` (first arrow only). */
const KEYWORD_LINE =
  /^(.*?)\s*(?:->|=>|→|⇒|\t)\s*(.*)$/s;

type KeywordRow = { id: string; from: string; to: string };

function parseKeywordPairs(spec: string): KeywordRow[] {
  const out: KeywordRow[] = [];
  for (const rawLine of spec.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const m = line.match(KEYWORD_LINE);
    if (!m || m[1] === undefined || m[2] === undefined) continue;
    const from = m[1].trim();
    const to = m[2].trim();
    if (!from) continue;
    out.push({ id: crypto.randomUUID(), from, to });
  }
  return out;
}

function serializeKeywordPairs(rows: KeywordRow[]): string {
  const lines: string[] = [];
  for (const r of rows) {
    const from = r.from.trim();
    if (!from) continue;
    lines.push(`${from} -> ${r.to.trim()}`);
  }
  return lines.join("\n");
}

async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, {
    ...init,
    headers: {
      ...(init?.headers ?? {}),
    },
  });
  if (!r.ok) {
    const text = await r.text();
    throw new Error(text || r.statusText);
  }
  return r.json() as Promise<T>;
}

export function App() {
  const [prefs, setPrefs] = useState<AppPreferences>(defaultPrefs);
  const [keys, setKeys] = useState<KeysPublic>({
    groq_configured: false,
    openrouter_configured: false,
  });
  const [groqKeyInput, setGroqKeyInput] = useState("");
  const [orKeyInput, setOrKeyInput] = useState("");
  const [orModels, setOrModels] = useState<OrModel[]>([]);
  const [groqChatModels, setGroqChatModels] = useState<string[]>([]);
  const [audioInputDevices, setAudioInputDevices] = useState<AudioInputDevice[]>([]);
  const [audioInputIds, setAudioInputIds] = useState({
    builtin: "__builtin_microphone__",
    systemDefault: "__system_default__",
  });
  const [transcriptionHistory, setTranscriptionHistory] = useState<
    TranscriptionHistoryEntry[]
  >([]);
  const [status, setStatus] = useState("Idle");
  const [statusColor, setStatusColor] = useState("#666666");
  const [busy, setBusy] = useState(false);
  const [recording, setRecording] = useState(false);
  const micRef = useRef<{ stop: () => Promise<Blob> } | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>("keys");
  const [bgPos, setBgPos] = useState({ x: 50, y: 40 });
  const [capturingToggleRecording, setCapturingToggleRecording] = useState(false);
  const [keywordRows, setKeywordRows] = useState<KeywordRow[]>([]);
  const [startupEnabled, setStartupEnabled] = useState(false);
  const [startupLoading, setStartupLoading] = useState(false);
  const [accessibilityTrusted, setAccessibilityTrusted] = useState(true);
  const [inputMonitoringTrusted, setInputMonitoringTrusted] = useState(true);
  const [microphoneAuthorized, setMicrophoneAuthorized] = useState(true);
  const [accessibilitySupported, setAccessibilitySupported] = useState(false);
  const [accessibilityIdentity, setAccessibilityIdentity] =
    useState<AccessibilityIdentity | null>(null);
  const [runtimeState, setRuntimeState] = useState<RuntimeState | null>(null);
  const [groqTranscribeCustomOpen, setGroqTranscribeCustomOpen] = useState(false);
  const [orTranscribeManualOpen, setOrTranscribeManualOpen] = useState(false);
  const [postCustomModelOpen, setPostCustomModelOpen] = useState(false);
  const prefsRef = useRef(prefs);
  const frontendSessionIdRef = useRef(crypto.randomUUID());
  prefsRef.current = prefs;

  const onShellPointerMove = useCallback((e: PointerEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    const r = el.getBoundingClientRect();
    const x = ((e.clientX - r.left) / Math.max(r.width, 1)) * 100;
    const y = ((e.clientY - r.top) / Math.max(r.height, 1)) * 100;
    setBgPos({ x, y });
  }, []);

  const setPref = useCallback(<K extends keyof AppPreferences>(k: K, v: AppPreferences[K]) => {
    setPrefs((p) => ({ ...p, [k]: v }));
  }, []);

  const refreshRuntimeState = useCallback(async () => {
    try {
      const runtime = await apiJson<RuntimeState>("/api/runtime-state");
      setRuntimeState(runtime);
    } catch {
      setRuntimeState(null);
    }
  }, []);

  const applyRuntimeIconStatus = useCallback((osIcon: RuntimeIconStatus) => {
    setRuntimeState((prev) =>
      prev
        ? {
            ...prev,
            icon_state: osIcon.display_state,
            os_icon: osIcon,
          }
        : prev,
    );
  }, []);

  const refreshAccessibility = useCallback(async () => {
    try {
      const ax = await apiJson<{
        supported: boolean;
        trusted: boolean;
        input_monitoring_trusted?: boolean;
        microphone_authorized?: boolean;
        identity?: AccessibilityIdentity;
      }>("/api/accessibility");
      setAccessibilitySupported(ax.supported);
      setAccessibilityTrusted(ax.trusted);
      setInputMonitoringTrusted(ax.input_monitoring_trusted ?? true);
      setMicrophoneAuthorized(ax.microphone_authorized ?? true);
      setAccessibilityIdentity(ax.identity ?? null);
    } catch {
      /* accessibility endpoint may not be supported on this platform */
    }
  }, []);

  const refreshAudioInputDevices = useCallback(async () => {
    const res = await apiJson<AudioInputDevicesResponse>("/api/audio-input-devices");
    setAudioInputDevices(res.devices ?? []);
    setAudioInputIds({
      builtin: res.builtin_id || "__builtin_microphone__",
      systemDefault: res.system_default_id || "__system_default__",
    });
  }, []);

  const commitKeywordRows = useCallback(
    (nextOrFn: KeywordRow[] | ((prev: KeywordRow[]) => KeywordRow[])) => {
      setKeywordRows((prev) => (typeof nextOrFn === "function" ? nextOrFn(prev) : nextOrFn));
    },
    [],
  );

  useEffect(() => {
    const sessionId = frontendSessionIdRef.current;
    const heartbeatPath = `/api/frontend-session/${encodeURIComponent(sessionId)}/heartbeat`;
    const closePath = `/api/frontend-session/${encodeURIComponent(sessionId)}/close`;
    const heartbeat = () => {
      void fetch(heartbeatPath, { method: "POST", keepalive: true }).catch(() => {});
    };
    const closeSession = () => {
      if (navigator.sendBeacon) {
        navigator.sendBeacon(closePath, new Blob([], { type: "application/octet-stream" }));
        return;
      }
      void fetch(closePath, { method: "POST", keepalive: true }).catch(() => {});
    };
    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        heartbeat();
      }
    };
    heartbeat();
    const id = window.setInterval(heartbeat, 20_000);
    window.addEventListener("pagehide", closeSession);
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      window.clearInterval(id);
      window.removeEventListener("pagehide", closeSession);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      closeSession();
    };
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    setPref("keyword_replacement_spec", serializeKeywordPairs(keywordRows));
  }, [keywordRows, hydrated, setPref]);

  useEffect(() => {
    if ((activeTab !== "output" && activeTab !== "stats") || !hydrated) return;
    const refresh = async () => {
      try {
        const hist = await apiJson<{ entries: Record<string, unknown>[] }>(
          "/api/transcription-history",
        );
        setTranscriptionHistory((hist.entries ?? []).map(mapHistoryEntry));
      } catch {
        /* offline */
      }
    };
    void refresh();
    const id = setInterval(refresh, 2000);
    return () => clearInterval(id);
  }, [activeTab, hydrated]);

  useEffect(() => {
    void (async () => {
      try {
        const [p, k, audioInputs] = await Promise.all([
          apiJson<Record<string, unknown>>("/api/preferences"),
          apiJson<KeysPublic>("/api/keys"),
          apiJson<AudioInputDevicesResponse>("/api/audio-input-devices"),
        ]);
        const merged = { ...defaultPrefs(), ...p } as AppPreferences;
        setPrefs(merged);
        setAudioInputDevices(audioInputs.devices ?? []);
        setAudioInputIds({
          builtin: audioInputs.builtin_id || "__builtin_microphone__",
          systemDefault: audioInputs.system_default_id || "__system_default__",
        });
        setKeywordRows(parseKeywordPairs(merged.keyword_replacement_spec));
        setKeys(k);
        try {
          const hist = await apiJson<{ entries: Record<string, unknown>[] }>(
            "/api/transcription-history",
          );
          setTranscriptionHistory((hist.entries ?? []).map(mapHistoryEntry));
        } catch {
          /* keep empty history if endpoint unavailable */
        }
        const or = await apiJson<{ models: OrModel[] }>("/api/openrouter/models");
        setOrModels(or.models ?? []);
        if (k.groq_configured) {
          const gm = await apiJson<{ models: string[] }>("/api/groq/chat-models");
          setGroqChatModels(gm.models ?? []);
        }
        try {
          const su = await apiJson<{ enabled: boolean }>("/api/startup");
          setStartupEnabled(su.enabled);
        } catch {
          /* startup endpoint may not be supported on this platform */
        }
        await refreshAccessibility();
      } catch (e) {
        setStatus(`Load failed: ${e instanceof Error ? e.message : String(e)}`);
        setStatusColor("#c62828");
      } finally {
        setHydrated(true);
      }
    })();
  }, [refreshAccessibility]);

  useEffect(() => {
    const refreshWhenVisible = () => {
      if (document.visibilityState === "visible") {
        void refreshAccessibility();
      }
    };
    window.addEventListener("focus", refreshAccessibility);
    document.addEventListener("visibilitychange", refreshWhenVisible);
    return () => {
      window.removeEventListener("focus", refreshAccessibility);
      document.removeEventListener("visibilitychange", refreshWhenVisible);
    };
  }, [refreshAccessibility]);

  useEffect(() => {
    if (!hydrated) return;
    let cancelled = false;
    const refresh = async () => {
      try {
        const runtime = await apiJson<RuntimeState>("/api/runtime-state");
        if (!cancelled) {
          setRuntimeState(runtime);
        }
      } catch {
        if (!cancelled) {
          setRuntimeState(null);
        }
      }
    };
    void refresh();
    const id = window.setInterval(refresh, 500);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [hydrated]);

  useEffect(() => {
    if (!hydrated) return;
    const t = window.setTimeout(() => {
      void (async () => {
        try {
          await apiJson("/api/preferences", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(prefsRef.current),
          });
        } catch {
          /* ignore autosave errors */
        }
      })();
    }, 500);
    return () => window.clearTimeout(t);
  }, [prefs, hydrated]);

  const saveKeys = async () => {
    const body: { groq?: string | null; openrouter?: string | null } = {};
    if (groqKeyInput.trim()) body.groq = groqKeyInput.trim();
    if (orKeyInput.trim()) body.openrouter = orKeyInput.trim();
    if (!body.groq && !body.openrouter) {
      setStatus("Enter a key to save.");
      setStatusColor("#ff9800");
      return;
    }
    try {
      const k = await apiJson<KeysPublic>("/api/keys", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setKeys(k);
      setGroqKeyInput("");
      setOrKeyInput("");
      setStatus("Keys saved.");
      setStatusColor("#66bb6a");
      if (k.groq_configured) {
        const gm = await apiJson<{ models: string[] }>("/api/groq/chat-models");
        setGroqChatModels(gm.models ?? []);
      }
    } catch (e) {
      setStatus(e instanceof Error ? e.message : String(e));
      setStatusColor("#c62828");
    }
  };

  const toggleStartup = async (enabled: boolean) => {
    setStartupLoading(true);
    try {
      const res = await apiJson<{ enabled: boolean }>("/api/startup", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      });
      setStartupEnabled(res.enabled);
    } catch (e) {
      setStatus(e instanceof Error ? e.message : String(e));
      setStatusColor("#c62828");
    } finally {
      setStartupLoading(false);
    }
  };

  const setRuntimeIconOverride = async (overrideState: RuntimeIconState | null) => {
    try {
      const res = await apiJson<{ os_icon: RuntimeIconStatus }>("/api/runtime-icon/override", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ override_state: overrideState }),
      });
      applyRuntimeIconStatus(res.os_icon);
      await refreshRuntimeState();
    } catch (e) {
      setStatus(e instanceof Error ? e.message : String(e));
      setStatusColor("#c62828");
    }
  };

  const refreshOrModels = async () => {
    try {
      setStatus("Loading OpenRouter models…");
      setStatusColor("#ff9800");
      const res = await apiJson<{ models: OrModel[] }>("/api/openrouter/models/refresh", {
        method: "POST",
      });
      setOrModels(res.models ?? []);
      setStatus(`Loaded ${res.models?.length ?? 0} models.`);
      setStatusColor("#66bb6a");
    } catch (e) {
      setStatus(e instanceof Error ? e.message : String(e));
      setStatusColor("#c62828");
    }
  };

  const onRecordClick = useCallback(async () => {
    if (busy) return;
    if (!recording) {
      try {
        const session = await startMicRecording(prefsRef.current.audio_input_device);
        micRef.current = session;
        setRecording(true);
        setStatus("Recording…");
        setStatusColor("#e53935");
      } catch (e) {
        setStatus(e instanceof Error ? e.message : String(e));
        setStatusColor("#c62828");
      }
      return;
    }
    const session = micRef.current;
    micRef.current = null;
    setRecording(false);
    if (!session) return;
    setBusy(true);
    setStatus("Transcribing…");
    setStatusColor("#ff9800");
    try {
      const blob = await session.stop();
      const p = prefsRef.current;
      const meta = {
        transcription_provider: p.transcription_provider,
        transcription_model_groq: p.transcription_model_groq,
        transcription_model_openrouter: p.transcription_model_openrouter,
        openrouter_transcription_instruction: p.openrouter_transcription_instruction,
        keyword_replacement_spec: p.keyword_replacement_spec,
        postprocess_enabled: p.postprocess_enabled,
        postprocess_prompt: p.postprocess_prompt,
        postprocess_provider: p.postprocess_provider,
        postprocess_model: p.postprocess_model,
        postprocess_groq_reasoning_effort: p.postprocess_groq_reasoning_effort,
        postprocess_openrouter_reasoning_effort: p.postprocess_openrouter_reasoning_effort,
      };
      const fd = new FormData();
      fd.set("meta", JSON.stringify(meta));
      fd.set("audio", blob, "recording.wav");
      const r = await fetch("/api/transcribe", { method: "POST", body: fd });
      if (!r.ok) {
        throw new Error(await r.text());
      }
      const raw = (await r.json()) as Record<string, unknown>;
      const entry = mapHistoryEntry(raw);
      setTranscriptionHistory((prev) => [
        entry,
        ...prev.filter((e) => e.id !== entry.id),
      ]);
      setStatus("Idle");
      setStatusColor("#666666");
    } catch (e) {
      setStatus(e instanceof Error ? e.message : String(e));
      setStatusColor("#c62828");
    } finally {
      setBusy(false);
    }
  }, [busy, recording]);

  useEffect(() => {
    if (!capturingToggleRecording) return;
    const onKey = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.key === "Escape") {
        setCapturingToggleRecording(false);
        return;
      }
      const combo = normalizeHotkeyFromEvent(e);
      if (isOnlyModifiersCombo(combo)) return;
      setPref("hotkey_toggle_recording", combo);
      setCapturingToggleRecording(false);
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [capturingToggleRecording, setPref]);

  const isGroq = prefs.transcription_provider === "groq";
  const postGroq = prefs.postprocess_provider === "groq";
  const actualIconState: RuntimeIconState = busy
    ? "processing"
    : recording
      ? "recording"
      : runtimeState?.capture_state ?? "idle";
  const statusIconSrc = runtimeIconSrc(actualIconState);
  const statusIconStyle = { color: runtimeIconColor(actualIconState) } as CSSProperties;
  const osIconOverride = runtimeState?.os_icon?.override_state ?? null;
  const hotkeyDiagnostics = runtimeState?.hotkey;
  const hotkeyConflict = hotkeyConflictWarning(prefs.hotkey_toggle_recording);
  const osIconStatus = runtimeState?.os_icon;
  const stats = useMemo(
    () => computeStats(transcriptionHistory),
    [transcriptionHistory],
  );
  const audioInputSelection = prefs.audio_input_device || audioInputIds.builtin;
  const selectedAudioInputKnown =
    audioInputSelection === audioInputIds.builtin ||
    audioInputSelection === audioInputIds.systemDefault ||
    audioInputDevices.some((d) => d.id === audioInputSelection);
  const hasBuiltInAudioInput = audioInputDevices.some((d) => d.is_builtin_candidate);
  const accessibilityGrantTarget =
    accessibilityIdentity?.app_bundle ?? accessibilityIdentity?.executable ?? "";
  const needsPrivacyPermission =
    accessibilitySupported &&
    (!accessibilityTrusted || !inputMonitoringTrusted || !microphoneAuthorized);

  const groqModelInDefaults = GROQ_STT_DEFAULTS.includes(
    prefs.transcription_model_groq as (typeof GROQ_STT_DEFAULTS)[number],
  );
  const groqDropdownValue = groqModelInDefaults
    ? prefs.transcription_model_groq
    : GROQ_STT_DEFAULTS[1];
  const groqCustomValue = groqModelInDefaults ? "" : prefs.transcription_model_groq;

  const bgStyle = {
    "--px": `${bgPos.x}%`,
    "--py": `${bgPos.y}%`,
  } as CSSProperties;

  return (
    <div className="app-shell" onPointerMove={onShellPointerMove}>
      <div className="app-bg" style={bgStyle} aria-hidden />
      <div className="app-inner">
        <header className="app-header">
          <h1 className="app-title">Scythe-Transcribe</h1>
          <div className="app-header-spacer" aria-hidden />
          <div className="status-pill" style={{ color: statusColor }}>
            <img
              key={statusIconSrc}
              className="status-icon"
              src={statusIconSrc}
              style={statusIconStyle}
              alt=""
              aria-hidden
            />
            <span>{status}</span>
          </div>
        </header>

        <div className="app-card">
          <div
            className="tablist"
            role="tablist"
            aria-label="Settings sections"
          >
            {TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                role="tab"
                id={`tab-${t.id}`}
                aria-selected={activeTab === t.id}
                aria-controls={`panel-${t.id}`}
                className="tab"
                onClick={() => setActiveTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>

          {activeTab === "general" && (
          <div
            id="panel-general"
            role="tabpanel"
            aria-labelledby="tab-general"
          >
            <h2 className="section-title">Microphone</h2>
            <p className="muted">
              Choose the input used by the global hotkey recorder. Built-in microphone avoids
              Bluetooth headset mode when macOS can find it.
            </p>
            <div className="field-row field-row--shortcut">
              <label className="flex-240">
                Audio input
                <select
                  className="input-field"
                  value={audioInputSelection}
                  onChange={(e) => setPref("audio_input_device", e.target.value)}
                >
                  <option value={audioInputIds.builtin}>Built-in microphone</option>
                  <option value={audioInputIds.systemDefault}>System default</option>
                  {audioInputDevices.map((device) => (
                    <option key={device.id} value={device.id}>
                      {device.name}
                      {device.is_default ? " (system default)" : ""}
                    </option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                className="btn-outline"
                onClick={() => {
                  void (async () => {
                    try {
                      await refreshAudioInputDevices();
                    } catch (e) {
                      setStatus(e instanceof Error ? e.message : String(e));
                      setStatusColor("#c62828");
                    }
                  })();
                }}
              >
                Refresh inputs
              </button>
            </div>
            {!selectedAudioInputKnown && (
              <p className="muted" style={{ color: "#ffb300", marginTop: "var(--space-2)" }}>
                Selected input is not currently available. Recording will use the system default.
              </p>
            )}
            {audioInputSelection === audioInputIds.builtin && !hasBuiltInAudioInput && (
              <p className="muted" style={{ color: "#ffb300", marginTop: "var(--space-2)" }}>
                Built-in microphone was not found. Recording will use the system default.
              </p>
            )}

            <h2 className="section-title section-title--spaced">Keyboard shortcuts</h2>
            <p className="muted">
              The desktop process captures this shortcut globally (hold to record, release to
              transcribe, post-process if enabled, then paste at the text cursor). Keep the settings
              server running from the tray app. OS-level shortcuts can still win first: on macOS,
              Cmd+Space is Spotlight by default and Ctrl+Space is often Input Source switching.
              Browser focus does not bypass those OS shortcuts. Press Esc to cancel capture.
            </p>
            <div className="field-row field-row--shortcut">
              <label className="flex-240">
                Hold to dictate (paste)
                <input
                  type="text"
                  className="input-field"
                  readOnly
                  value={
                    capturingToggleRecording
                      ? "Press keys…"
                      : formatHotkeyLabel(prefs.hotkey_toggle_recording)
                  }
                  aria-live="polite"
                />
              </label>
              <button
                type="button"
                className={capturingToggleRecording ? "btn-primary" : "btn-outline"}
                onClick={() => setCapturingToggleRecording((c) => !c)}
              >
                {capturingToggleRecording ? "Cancel capture" : "Set shortcut"}
              </button>
              <button
                type="button"
                className="btn-outline"
                disabled={!prefs.hotkey_toggle_recording.trim()}
                onClick={() => setPref("hotkey_toggle_recording", "")}
              >
                Clear
              </button>
              <button
                type="button"
                className="btn-outline"
                onClick={() =>
                  setPref("hotkey_toggle_recording", defaultPrefs().hotkey_toggle_recording)
                }
              >
                Reset to default
              </button>
            </div>
            {hotkeyConflict && (
              <p className="muted" style={{ color: "#ffb300", marginTop: "var(--space-2)" }}>
                {hotkeyConflict}
              </p>
            )}

            {needsPrivacyPermission && (
              <>
                <h2 className="section-title section-title--spaced">macOS privacy permissions</h2>
                <p className="muted">
                  Scythe needs Input Monitoring to see global hotkeys, Accessibility to paste the
                  transcript back into other apps, and Microphone access so the hold-to-record hotkey
                  can capture audio. Use Ask macOS first so the system can show the
                  native permission prompts. If a permission was denied earlier, or an old build with
                  the same name left a stale entry, use Open Settings and enable{" "}
                  <code>Scythe Transcribe</code> — prefer the packaged <code>.app</code> so
                  permissions are tied to this app&apos;s bundle identifier.
                </p>
                <p className="muted">
                  Input Monitoring granted: {inputMonitoringTrusted ? "Yes" : "No"} ·
                  Accessibility granted: {accessibilityTrusted ? "Yes" : "No"} · Microphone granted:{" "}
                  {microphoneAuthorized ? "Yes" : "No"}
                </p>
                {accessibilityGrantTarget && (
                  <p className="muted">
                    Current app: <code>{accessibilityGrantTarget}</code>
                  </p>
                )}
                <div className="panel-keys-actions">
                  {!inputMonitoringTrusted && (
                    <>
                      <button
                        type="button"
                        className="btn-primary"
                        onClick={() => {
                          void (async () => {
                            await apiJson("/api/input-monitoring/request-prompt", {
                              method: "POST",
                            });
                            window.setTimeout(() => void refreshAccessibility(), 1500);
                          })();
                        }}
                      >
                        Ask macOS (Input Monitoring)
                      </button>
                      <button
                        type="button"
                        className="btn-outline"
                        onClick={() => {
                          void (async () => {
                            await apiJson("/api/input-monitoring/open-settings", {
                              method: "POST",
                            });
                            window.setTimeout(() => void refreshAccessibility(), 1000);
                          })();
                        }}
                      >
                        Open Input Monitoring settings
                      </button>
                    </>
                  )}
                  {!accessibilityTrusted && (
                    <>
                      <button
                        type="button"
                        className="btn-primary"
                        onClick={() => {
                          void (async () => {
                            await apiJson("/api/accessibility/request-prompt", { method: "POST" });
                            window.setTimeout(() => void refreshAccessibility(), 1500);
                          })();
                        }}
                      >
                        Ask macOS (Accessibility)
                      </button>
                      <button
                        type="button"
                        className="btn-outline"
                        onClick={() => {
                          void (async () => {
                            await apiJson("/api/accessibility/open-settings", { method: "POST" });
                            window.setTimeout(() => void refreshAccessibility(), 1000);
                          })();
                        }}
                      >
                        Open Accessibility settings
                      </button>
                    </>
                  )}
                  {!microphoneAuthorized && (
                    <>
                      <button
                        type="button"
                        className="btn-primary"
                        onClick={() => {
                          void (async () => {
                            await apiJson("/api/microphone/request-access", { method: "POST" });
                            window.setTimeout(() => void refreshAccessibility(), 1500);
                          })();
                        }}
                      >
                        Ask macOS (Microphone)
                      </button>
                      <button
                        type="button"
                        className="btn-outline"
                        onClick={() => {
                          void (async () => {
                            await apiJson("/api/microphone/open-settings", { method: "POST" });
                            window.setTimeout(() => void refreshAccessibility(), 1000);
                          })();
                        }}
                      >
                        Open Microphone settings
                      </button>
                    </>
                  )}
                </div>
              </>
            )}

            <h2 className="section-title section-title--spaced">Startup</h2>
            <div className="field-row field-row--center field-row--meta">
              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={startupEnabled}
                  disabled={startupLoading}
                  onChange={(e) => void toggleStartup(e.target.checked)}
                />
                Launch at login
              </label>
              <span className="muted">
                Automatically start Scythe-Transcribe when you log in.
              </span>
            </div>

            <h2 className="section-title section-title--spaced">Diagnostics</h2>
            <div className="diagnostic-row">
              <div className="diagnostic-status">
                <span className="field-inline-label">Backend audio capture</span>
                <span
                  className={
                    runtimeState?.capturing_audio
                      ? "diagnostic-pill diagnostic-pill--active"
                      : "diagnostic-pill"
                  }
                >
                  <span className="diagnostic-dot" aria-hidden />
                  {runtimeState === null
                    ? "Runtime status unavailable"
                    : runtimeState.capturing_audio
                      ? "Capturing audio"
                      : "Not capturing"}
                </span>
              </div>
              <label className="field-inline field-inline--runtime-icon">
                <span className="field-inline-label">OS icon override</span>
                <select
                  className="input-field input-field--toolbar"
                  value={osIconOverride ?? ""}
                  onChange={(e) =>
                    void setRuntimeIconOverride(parseRuntimeIconOverride(e.target.value))
                  }
                >
                  <option value="">Follow backend</option>
                  <option value="idle">Idle blue</option>
                  <option value="recording">Recording red</option>
                  <option value="processing">Processing yellow</option>
                </select>
              </label>
            </div>

            <div className="diagnostic-grid">
              <div className="diagnostic-item">
                <span className="field-inline-label">Hotkey listener</span>
                <strong>{hotkeyDiagnostics?.state ?? "Unknown"}</strong>
              </div>
              <div className="diagnostic-item">
                <span className="field-inline-label">Listener backend</span>
                <strong>{hotkeyDiagnostics?.listener_backend || "Unknown"}</strong>
              </div>
              <div className="diagnostic-item">
                <span className="field-inline-label">Accessibility trusted</span>
                <strong>{formatDiagnosticBool(hotkeyDiagnostics?.accessibility_trusted)}</strong>
              </div>
              <div className="diagnostic-item">
                <span className="field-inline-label">Input Monitoring trusted</span>
                <strong>{formatDiagnosticBool(hotkeyDiagnostics?.input_monitoring_trusted)}</strong>
              </div>
              <div className="diagnostic-item">
                <span className="field-inline-label">Microphone authorized</span>
                <strong>{formatDiagnosticBool(hotkeyDiagnostics?.microphone_authorized)}</strong>
              </div>
              <div className="diagnostic-item">
                <span className="field-inline-label">Microphone (AVFoundation)</span>
                <strong>{hotkeyDiagnostics?.microphone_authorization ?? "unknown"}</strong>
              </div>
              <div className="diagnostic-item">
                <span className="field-inline-label">Capture state</span>
                <strong>{runtimeState?.capture_state ?? "unknown"}</strong>
              </div>
              <div className="diagnostic-item">
                <span className="field-inline-label">OS icon</span>
                <strong>
                  {osIconStatus
                    ? `${osIconStatus.display_state}${
                        osIconStatus.override_state ? " override" : " follows backend"
                      }`
                    : "unknown"}
                </strong>
              </div>
              <div className="diagnostic-item">
                <span className="field-inline-label">Configured shortcut</span>
                <strong>{hotkeyDiagnostics?.configured_combo || "None"}</strong>
              </div>
              <div className="diagnostic-item">
                <span className="field-inline-label">Backend parsed shortcut</span>
                <strong>{formatDiagnosticList(hotkeyDiagnostics?.combo_parts)}</strong>
              </div>
              <div className="diagnostic-item">
                <span className="field-inline-label">Pressed tokens</span>
                <strong>{formatDiagnosticList(hotkeyDiagnostics?.pressed_tokens)}</strong>
              </div>
              <div className="diagnostic-item">
                <span className="field-inline-label">Combo active</span>
                <strong>{formatDiagnosticBool(hotkeyDiagnostics?.combo_active)}</strong>
              </div>
              <div className="diagnostic-item">
                <span className="field-inline-label">Last key event</span>
                <strong>
                  {hotkeyDiagnostics?.last_event
                    ? `${hotkeyDiagnostics.last_event}: ${hotkeyDiagnostics.last_token ?? "unknown"}`
                    : "None"}
                </strong>
              </div>
              <div className="diagnostic-item">
                <span className="field-inline-label">Last event time</span>
                <strong>{formatDiagnosticTime(hotkeyDiagnostics?.last_event_at)}</strong>
              </div>
              <div className="diagnostic-item">
                <span className="field-inline-label">Event count</span>
                <strong>{hotkeyDiagnostics?.event_count ?? 0}</strong>
              </div>
              <div className="diagnostic-item">
                <span className="field-inline-label">Last combo match</span>
                <strong>{formatDiagnosticTime(hotkeyDiagnostics?.last_combo_matched_at)}</strong>
              </div>
              <div className="diagnostic-item diagnostic-item--wide">
                <span className="field-inline-label">Last stream error</span>
                <strong>{hotkeyDiagnostics?.last_stream_error || "None"}</strong>
              </div>
              <div className="diagnostic-item diagnostic-item--wide">
                <span className="field-inline-label">Last transcription error</span>
                <strong>{hotkeyDiagnostics?.last_transcribe_error || "None"}</strong>
              </div>
            </div>
          </div>
          )}

          {activeTab === "keys" && (
          <div
            id="panel-keys"
            role="tabpanel"
            aria-labelledby="tab-keys"
          >
            <h2 className="section-title">API keys (stored in local file)</h2>
            <div className="panel-keys-fields">
              <label className="panel-keys-label">
                Groq API key
                <input
                  type="password"
                  className="input-field"
                  value={groqKeyInput}
                  onChange={(e) => setGroqKeyInput(e.target.value)}
                  placeholder={keys.groq_configured ? "(saved)" : ""}
                  autoComplete="off"
                />
              </label>
              <label className="panel-keys-label">
                OpenRouter API key
                <input
                  type="password"
                  className="input-field"
                  value={orKeyInput}
                  onChange={(e) => setOrKeyInput(e.target.value)}
                  placeholder={keys.openrouter_configured ? "(saved)" : ""}
                  autoComplete="off"
                />
              </label>
            </div>
            <div className="panel-keys-actions">
              <button type="button" className="btn-primary" onClick={() => void saveKeys()}>
                Save keys
              </button>
              <button type="button" className="btn-outline" onClick={() => void refreshOrModels()}>
                Refresh OpenRouter models
              </button>
            </div>
            <p className="muted panel-keys-footnote">
              Groq: {keys.groq_configured ? "key saved" : "no key"} · OpenRouter:{" "}
              {keys.openrouter_configured ? "key saved" : "no key"}
            </p>
          </div>
          )}

          {activeTab === "transcribe" && (
          <div
            id="panel-transcribe"
            role="tabpanel"
            aria-labelledby="tab-transcribe"
          >
            <div className="stack-gap">
              <div>
                <h2 className="section-title">Transcription</h2>
                <div className="provider-model-toolbar">
                  <label className="field-inline field-inline--provider">
                    <span className="field-inline-label">Provider</span>
                    <select
                      className="input-field input-field--toolbar"
                      value={prefs.transcription_provider}
                      onChange={(e) => setPref("transcription_provider", e.target.value)}
                    >
                      <option value="groq">Groq</option>
                      <option value="openrouter">OpenRouter</option>
                    </select>
                  </label>
                  <div className="provider-model-toolbar-main">
                    {isGroq ? (
                      <label className="field-inline">
                        <span className="field-inline-label">Groq model</span>
                        <select
                          className="input-field input-field--toolbar"
                          value={groqDropdownValue}
                          onChange={(e) => setPref("transcription_model_groq", e.target.value)}
                        >
                          {GROQ_STT_DEFAULTS.map((m) => (
                            <option key={m} value={m}>
                              {m}
                            </option>
                          ))}
                        </select>
                      </label>
                    ) : (
                      <OpenRouterModelPicker
                        idBase="or-transcribe"
                        label="Model"
                        models={orModels}
                        mode="audio"
                        value={prefs.transcription_model_openrouter}
                        onChange={(id) => setPref("transcription_model_openrouter", id)}
                      />
                    )}
                  </div>
                </div>
                <div className="provider-model-extra">
                  {isGroq ? (
                    groqModelInDefaults && !groqTranscribeCustomOpen ? (
                      <button
                        type="button"
                        className="btn-link-caret"
                        onClick={() => setGroqTranscribeCustomOpen(true)}
                      >
                        Custom Groq model ID…
                      </button>
                    ) : (
                      <div className="custom-id-expand">
                        <div className="custom-id-expand-head">
                          <span className="field-inline-label" id="transcribe-groq-custom-lbl">
                            Custom Groq model ID
                          </span>
                          {groqModelInDefaults ? (
                            <button
                              type="button"
                              className="btn-link-quiet"
                              onClick={() => setGroqTranscribeCustomOpen(false)}
                            >
                              Cancel
                            </button>
                          ) : (
                            <button
                              type="button"
                              className="btn-link-quiet"
                              onClick={() =>
                                setPref("transcription_model_groq", groqDropdownValue)
                              }
                            >
                              Use catalog model
                            </button>
                          )}
                        </div>
                        <input
                          id="transcribe-groq-custom"
                          className="input-field"
                          aria-labelledby="transcribe-groq-custom-lbl"
                          value={groqCustomValue}
                          onChange={(e) => {
                            const v = e.target.value.trim();
                            setPref("transcription_model_groq", v || groqDropdownValue);
                          }}
                          placeholder="Overrides catalog when set"
                          autoComplete="off"
                        />
                      </div>
                    )
                  ) : !orTranscribeManualOpen ? (
                    <button
                      type="button"
                      className="btn-link-caret"
                      onClick={() => setOrTranscribeManualOpen(true)}
                    >
                      Enter model ID manually…
                    </button>
                  ) : (
                    <div className="custom-id-expand">
                      <div className="custom-id-expand-head">
                        <span className="field-inline-label" id="transcribe-or-manual-lbl">
                          Custom model ID
                        </span>
                        <button
                          type="button"
                          className="btn-link-quiet"
                          onClick={() => setOrTranscribeManualOpen(false)}
                        >
                          Hide
                        </button>
                      </div>
                      <input
                        id="transcribe-or-manual"
                        className="input-field"
                        aria-labelledby="transcribe-or-manual-lbl"
                        value={prefs.transcription_model_openrouter}
                        onChange={(e) =>
                          setPref("transcription_model_openrouter", e.target.value)
                        }
                        placeholder="e.g. vendor/model-id"
                        autoComplete="off"
                      />
                    </div>
                  )}
                </div>
                {!isGroq ? (
                  <label className="field-inline" style={{ marginTop: "var(--space-4)" }}>
                    <span className="field-inline-label">Transcription instruction</span>
                    <textarea
                      className="input-field"
                      style={{ minHeight: 72 }}
                      value={prefs.openrouter_transcription_instruction}
                      onChange={(e) =>
                        setPref("openrouter_transcription_instruction", e.target.value)
                      }
                    />
                  </label>
                ) : null}
              </div>

              <div>
                <h2 className="section-title">Keyword dictionary</h2>
                <p className="muted">
                  Replace mistaken words or phrases in the raw transcript (before any LLM step). Longer
                  phrases are applied first. With Groq transcription, these terms are also sent as a
                  Whisper prompt so recognition can follow your vocabulary.
                </p>
                {keywordRows.length === 0 ? (
                  <p className="muted keyword-dict-empty">No rules yet. Add a replacement below.</p>
                ) : (
                  <ul className="keyword-list" aria-label="Keyword replacements">
                    {keywordRows.map((row, index) => (
                      <li key={row.id} className="keyword-row">
                        <label className="keyword-field">
                          <span className="keyword-field-label">Find</span>
                          <input
                            className="input-field"
                            value={row.from}
                            onChange={(e) => {
                              const v = e.target.value;
                              commitKeywordRows((prev) =>
                                prev.map((r) => (r.id === row.id ? { ...r, from: v } : r)),
                              );
                            }}
                            placeholder="e.g. teh"
                            autoComplete="off"
                            spellCheck={true}
                          />
                        </label>
                        <span className="keyword-arrow" aria-hidden>
                          →
                        </span>
                        <label className="keyword-field">
                          <span className="keyword-field-label">Replace with</span>
                          <input
                            className="input-field"
                            value={row.to}
                            onChange={(e) => {
                              const v = e.target.value;
                              commitKeywordRows((prev) =>
                                prev.map((r) => (r.id === row.id ? { ...r, to: v } : r)),
                              );
                            }}
                            placeholder="e.g. the"
                            autoComplete="off"
                            spellCheck={true}
                          />
                        </label>
                        <button
                          type="button"
                          className="btn-remove-row"
                          onClick={() =>
                            commitKeywordRows((prev) => prev.filter((r) => r.id !== row.id))
                          }
                          aria-label={`Remove replacement ${index + 1}`}
                        >
                          Remove
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
                <div className="keyword-dict-actions">
                  <button
                    type="button"
                    className="btn-outline"
                    onClick={() =>
                      commitKeywordRows((prev) => [
                        ...prev,
                        { id: crypto.randomUUID(), from: "", to: "" },
                      ])
                    }
                  >
                    Add rule
                  </button>
                </div>
                <p className="muted">
                  Lines starting with # in a saved file are treated as comments by the engine; this
                  editor only manages replacement rules.
                </p>
              </div>

              <div>
                <button
                  type="button"
                  className="btn-primary"
                  disabled={busy}
                  onClick={() => void onRecordClick()}
                >
                  <img className="action-button-icon" src={statusIconSrc} alt="" aria-hidden />
                  {recording ? "Stop & transcribe" : "Start recording"}
                </button>
              </div>
            </div>
          </div>
          )}

          {activeTab === "postprocess" && (
          <div
            id="panel-postprocess"
            role="tabpanel"
            aria-labelledby="tab-postprocess"
          >
            <h2 className="section-title">LLM post-processing</h2>
            <div className="stack-gap">
              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={prefs.postprocess_enabled}
                  onChange={(e) => setPref("postprocess_enabled", e.target.checked)}
                />
                Auto-process transcript with prompt
              </label>
              <label className="field-inline">
                <span className="postprocess-prompt-label">Post-process prompt</span>
                <textarea
                  className="input-field"
                  style={{ minHeight: 80 }}
                  value={prefs.postprocess_prompt}
                  onChange={(e) => setPref("postprocess_prompt", e.target.value)}
                />
              </label>
              <div>
                <div className="provider-model-toolbar">
                  <label className="field-inline field-inline--provider">
                    <span className="field-inline-label">Provider</span>
                    <select
                      className="input-field input-field--toolbar"
                      value={prefs.postprocess_provider}
                      onChange={(e) => setPref("postprocess_provider", e.target.value)}
                    >
                      <option value="groq">Groq</option>
                      <option value="openrouter">OpenRouter</option>
                    </select>
                  </label>
                  <div className="provider-model-toolbar-main">
                    {postGroq ? (
                      <label className="field-inline">
                        <span className="field-inline-label">Groq model</span>
                        <select
                          className="input-field input-field--toolbar"
                          value={prefs.postprocess_model}
                          onChange={(e) => setPref("postprocess_model", e.target.value)}
                        >
                          <option value="">(select)</option>
                          {groqChatModels.map((m) => (
                            <option key={m} value={m}>
                              {m}
                            </option>
                          ))}
                        </select>
                      </label>
                    ) : (
                      <OpenRouterModelPicker
                        idBase="or-postprocess"
                        label="Model"
                        models={orModels}
                        mode="all"
                        value={prefs.postprocess_model}
                        onChange={(id) => setPref("postprocess_model", id)}
                      />
                    )}
                  </div>
                </div>
                <div className="provider-model-extra">
                  {!postCustomModelOpen ? (
                    <button
                      type="button"
                      className="btn-link-caret"
                      onClick={() => setPostCustomModelOpen(true)}
                    >
                      Custom model ID override…
                    </button>
                  ) : (
                    <div className="custom-id-expand">
                      <div className="custom-id-expand-head">
                        <span className="field-inline-label" id="post-custom-model-lbl">
                          Custom model ID
                        </span>
                        <button
                          type="button"
                          className="btn-link-quiet"
                          onClick={() => setPostCustomModelOpen(false)}
                        >
                          Hide
                        </button>
                      </div>
                      <input
                        id="post-custom-model"
                        className="input-field"
                        aria-labelledby="post-custom-model-lbl"
                        value={prefs.postprocess_model}
                        onChange={(e) => setPref("postprocess_model", e.target.value)}
                        placeholder="Overrides the selection above"
                        autoComplete="off"
                      />
                    </div>
                  )}
                </div>
              </div>
              <div className="postprocess-reasoning-row">
                {postGroq ? (
                  <label className="field-inline">
                    <span className="field-inline-label">Groq reasoning effort</span>
                    <select
                      className="input-field input-field--toolbar"
                      value={prefs.postprocess_groq_reasoning_effort}
                      onChange={(e) => setPref("postprocess_groq_reasoning_effort", e.target.value)}
                    >
                      <option value="">Default (omit)</option>
                      {GROQ_POST_REASONING_EFFORTS.filter((x) => x !== "").map((x) => (
                        <option key={x} value={x}>
                          {x}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : (
                  <label className="field-inline">
                    <span className="field-inline-label">OpenRouter reasoning effort</span>
                    <select
                      className="input-field input-field--toolbar"
                      value={prefs.postprocess_openrouter_reasoning_effort}
                      onChange={(e) =>
                        setPref("postprocess_openrouter_reasoning_effort", e.target.value)
                      }
                    >
                      <option value="">Default (omit)</option>
                      {OR_POST_REASONING_EFFORTS.filter((x) => x !== "").map((x) => (
                        <option key={x} value={x}>
                          {x}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
              </div>
              <p className="muted">
                Reasoning options depend on the post-process model; unsupported values may be
                ignored or rejected by the provider.
              </p>
            </div>
          </div>
          )}

          {activeTab === "stats" && (
          <div
            id="panel-stats"
            role="tabpanel"
            aria-labelledby="tab-stats"
          >
            <h2 className="section-title">Transcription stats</h2>
            <p className="muted output-hint">
              Totals come from local transcription history. Cost appears only when the provider
              returned a usable cost field.
            </p>
            {stats.transcriptionCount === 0 ? (
              <p className="muted">No transcriptions yet. Record and transcribe from the Transcribe tab.</p>
            ) : (
              <div className="stats-panel">
                <div className="stats-hero-grid">
                  <div className="stats-card stats-card--wide">
                    <span className="stats-label">Audio transcribed</span>
                    <strong>{formatAudioDuration(stats.totalAudioSec)}</strong>
                    <span className="stats-detail">{formatAudioDurationDetail(stats.totalAudioSec)}</span>
                  </div>
                  <div className="stats-card">
                    <span className="stats-label">Transcriptions</span>
                    <strong>{formatNumber(stats.transcriptionCount)}</strong>
                    <span className="stats-detail">
                      {stats.firstAt != null && stats.lastAt != null
                        ? `${new Date(stats.firstAt).toLocaleDateString()} → ${new Date(
                            stats.lastAt,
                          ).toLocaleDateString()}`
                        : "Local history"}
                    </span>
                  </div>
                  <div className="stats-card">
                    <span className="stats-label">Words</span>
                    <strong>{formatNumber(stats.totalWords)}</strong>
                    <span className="stats-detail">
                      {formatDecimal(stats.averageWordsPerEntry, 1)} avg each
                    </span>
                  </div>
                  <div className="stats-card">
                    <span className="stats-label">Characters</span>
                    <strong>{formatNumber(stats.totalChars)}</strong>
                    <span className="stats-detail">Raw transcript text</span>
                  </div>
                  <div className="stats-card stats-card--accent">
                    <span className="stats-label">Avg speech pace</span>
                    <strong>{formatDecimal(stats.averageSpeechWpm, 1)}</strong>
                    <span className="stats-detail">words per audio minute</span>
                  </div>
                  <div className="stats-card">
                    <span className="stats-label">API throughput</span>
                    <strong>{formatDecimal(stats.apiThroughputWpm, 0)}</strong>
                    <span className="stats-detail">words per processing minute</span>
                  </div>
                  <div className="stats-card">
                    <span className="stats-label">Total cost</span>
                    <strong>{formatCostUsd(stats.totalCostUsd)}</strong>
                    <span className="stats-detail">
                      {stats.costEntryCount > 0
                        ? `${stats.costEntryCount.toLocaleString()} entries with cost`
                        : "Provider did not return cost"}
                    </span>
                  </div>
                </div>

                <div className="stats-insight-grid">
                  <div className="stats-insight">
                    <span className="stats-label">Average API wait</span>
                    <strong>
                      {stats.averageApiSeconds == null
                        ? "—"
                        : `${formatDecimal(stats.averageApiSeconds, 2)} s`}
                    </strong>
                  </div>
                  <div className="stats-insight">
                    <span className="stats-label">Longest audio</span>
                    <strong>{formatAudioDuration(stats.longestAudioSec)}</strong>
                  </div>
                  <div className="stats-insight">
                    <span className="stats-label">Known audio coverage</span>
                    <strong>
                      {stats.timedEntryCount > 0
                        ? `${stats.timedEntryCount.toLocaleString()} / ${stats.transcriptionCount.toLocaleString()} entries`
                        : "Missing from older entries"}
                    </strong>
                  </div>
                </div>

                <div className="stats-recent">
                  <h3>Recent word volume</h3>
                  <ul className="stats-bars" aria-label="Recent transcription word counts">
                    {stats.recent.map((item) => {
                      const maxWords = Math.max(1, ...stats.recent.map((x) => x.words));
                      const width = Math.max(4, Math.round((item.words / maxWords) * 100));
                      return (
                        <li key={item.id} className="stats-bar-row">
                          <span className="stats-bar-label">{item.label}</span>
                          <span className="stats-bar-track" aria-hidden>
                            <span className="stats-bar-fill" style={{ width: `${width}%` }} />
                          </span>
                          <span className="stats-bar-value">
                            {item.words.toLocaleString()} words
                            {item.wpm != null ? ` · ${formatDecimal(item.wpm, 0)} wpm` : ""}
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              </div>
            )}
          </div>
          )}

          {activeTab === "output" && (
          <div
            id="panel-output"
            role="tabpanel"
            aria-labelledby="tab-output"
          >
            <h2 className="section-title">Transcription history</h2>
            <p className="muted output-hint">
              Timings below include pipeline breakdown; the Output tab refreshes every 2s so hotkey
              paste metrics appear after dictation.
            </p>
            {transcriptionHistory.length === 0 ? (
              <p className="muted">No transcriptions yet. Record and transcribe from the Transcribe tab.</p>
            ) : (
              <ul className="history-list" aria-label="Transcription history">
                {transcriptionHistory.map((entry) => {
                  const timingExtra = historyTimingSecondaryLine(entry);
                  const timingHotkey = historyTimingHotkeyLine(entry);
                  return (
                  <li key={entry.id} className="history-item">
                    <time className="history-time" dateTime={new Date(entry.createdAt).toISOString()}>
                      {new Date(entry.createdAt).toLocaleString()}
                    </time>
                    <p className="history-metrics" aria-label="Timing">
                      Transcribe {formatDurationMs(entry.transcribeMs)}
                      {entry.postprocessMs != null && Number.isFinite(entry.postprocessMs)
                        ? ` · Post-process ${formatDurationMs(entry.postprocessMs)}`
                        : ""}
                      {` · Total ${formatDurationMs(entry.totalMs)}`}
                    </p>
                    {timingExtra ? (
                      <p className="history-timing-detail" aria-label="Pipeline timing detail">
                        {timingExtra}
                      </p>
                    ) : null}
                    {timingHotkey ? (
                      <p className="history-timing-detail" aria-label="Hotkey timing">
                        {timingHotkey}
                      </p>
                    ) : null}
                    <div className="history-item-body">
                      <label className="history-block">
                        Transcript
                        <textarea
                          className="input-field history-textarea"
                          readOnly
                          value={entry.transcript}
                          rows={6}
                        />
                      </label>
                      <label className="history-block">
                        LLM processed output
                        <textarea
                          className="input-field history-textarea"
                          readOnly
                          value={entry.processed}
                          rows={6}
                          placeholder="(none)"
                        />
                      </label>
                    </div>
                  </li>
                  );
                })}
              </ul>
            )}
          </div>
          )}
        </div>
      </div>
    </div>
  );
}
