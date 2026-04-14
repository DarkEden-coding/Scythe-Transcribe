import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";

export type OrModel = {
  model_id: string;
  name: string;
  supports_audio_input: boolean;
  supports_text: boolean;
  pricing_prompt?: string;
  pricing_completion?: string;
};

type OpenRouterModelPickerProps = {
  models: OrModel[];
  value: string;
  onChange: (modelId: string) => void;
  mode: "audio" | "all";
  label: string;
  idBase: string;
};

function priceSummary(m: OrModel): string {
  const a = (m.pricing_prompt ?? "").trim();
  const b = (m.pricing_completion ?? "").trim();
  if (a && b) return `in ${a} · out ${b}`;
  if (a) return `in ${a}`;
  if (b) return `out ${b}`;
  return "";
}

export function OpenRouterModelPicker({
  models,
  value,
  onChange,
  mode,
  label,
  idBase,
}: OpenRouterModelPickerProps) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const reactId = useId();
  const searchId = `${idBase}-search-${reactId}`;
  const triggerId = `${idBase}-trigger-${reactId}`;
  const listId = `${idBase}-listbox-${reactId}`;

  const baseList = useMemo(() => {
    if (mode === "audio") {
      return models.filter((m) => m.supports_audio_input);
    }
    return models;
  }, [models, mode]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return baseList;
    return baseList.filter((m) => {
      const id = m.model_id.toLowerCase();
      const name = (m.name ?? "").toLowerCase();
      const price = priceSummary(m).toLowerCase();
      return id.includes(q) || name.includes(q) || price.includes(q);
    });
  }, [baseList, query]);

  const selected = useMemo(
    () => baseList.find((m) => m.model_id === value) ?? null,
    [baseList, value],
  );

  const onPick = useCallback(
    (id: string) => {
      onChange(id);
      setQuery("");
      setOpen(false);
    },
    [onChange],
  );

  useEffect(() => {
    if (!open) return;
    const t = window.setTimeout(() => searchRef.current?.focus(), 0);
    return () => window.clearTimeout(t);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        setOpen(false);
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      const el = wrapRef.current;
      if (el && !el.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const triggerSummary = selected
    ? selected.name || selected.model_id
    : value.trim()
      ? value.trim()
      : "Select a model…";

  return (
    <div className="or-picker or-picker--inline" ref={wrapRef}>
      {label ? (
        <label className="or-picker-label" htmlFor={triggerId}>
          {label}
        </label>
      ) : null}
      <div className="or-picker-anchor">
        <button
          id={triggerId}
          type="button"
          className="or-picker-trigger"
          aria-expanded={open}
          aria-haspopup="listbox"
          aria-controls={listId}
          onClick={() => setOpen((o) => !o)}
        >
          <span className="or-picker-trigger-text">
            <span className="or-picker-trigger-name">{triggerSummary}</span>
            {selected ? (
              <span className="or-picker-trigger-id">{selected.model_id}</span>
            ) : value.trim() && !selected ? (
              <span className="or-picker-trigger-id">{value.trim()}</span>
            ) : null}
          </span>
          <span className="or-picker-trigger-chevron" aria-hidden>
            ▾
          </span>
        </button>
        {open ? (
          <div className="or-picker-popout" role="presentation">
            <input
              ref={searchRef}
              id={searchId}
              type="search"
              className="input-field or-picker-search"
              placeholder="Search name, id, or price…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              autoComplete="off"
              aria-controls={listId}
            />
            <div
              id={listId}
              className="or-picker-list or-picker-list--popout"
              role="listbox"
              aria-label={label || "OpenRouter models"}
            >
              {filtered.length === 0 ? (
                <div className="or-picker-empty">
                  {baseList.length === 0
                    ? mode === "audio"
                      ? "No audio-capable models in the catalog. Try refreshing the model list from API keys."
                      : "No models loaded. Open the API keys tab and use Refresh OpenRouter models."
                    : "No models match your search."}
                </div>
              ) : (
                filtered.map((m) => {
                  const active = m.model_id === value;
                  const p = priceSummary(m);
                  return (
                    <button
                      key={m.model_id}
                      type="button"
                      role="option"
                      aria-selected={active}
                      className={`or-picker-row${active ? " or-picker-row-active" : ""}`}
                      onClick={() => onPick(m.model_id)}
                    >
                      <span className="or-picker-row-main">
                        <span className="or-picker-row-name">{m.name || m.model_id}</span>
                        <span className="or-picker-row-id">{m.model_id}</span>
                      </span>
                      {p ? <span className="or-picker-row-price">{p}</span> : null}
                    </button>
                  );
                })
              )}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
