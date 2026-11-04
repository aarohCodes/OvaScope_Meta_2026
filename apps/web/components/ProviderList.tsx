"use client";

import type { Provider } from "@/lib/types";

const KIND: Record<string, string> = {
  representation: "Image representation",
  segmentation: "Automatic segmentation",
  classifier: "Trained classifier",
  multimodal: "Multimodal / language (optional)",
  artifact_store: "Artifact storage",
};

export function ProviderList({ providers, detailed = false }: { providers: Provider[]; detailed?: boolean }) {
  return (
    <ul className="divide-y divide-line" data-testid="provider-list">
      {providers.map((p) => (
        <li key={p.kind + p.provider_id} className="flex flex-col gap-1 py-2.5 sm:flex-row sm:items-start sm:gap-4" data-provider={p.kind} data-available={p.available}>
          <div className="flex w-56 shrink-0 items-center gap-2">
            <span
              aria-hidden
              className={`h-2 w-2 rounded-full ${p.available ? "bg-accent-mark" : p.kind === "multimodal" ? "bg-muted" : "bg-warn"}`}
            />
            <span className="text-sm text-ink">{KIND[p.kind] ?? p.kind}</span>
          </div>
          <div className="min-w-0 flex-1 text-sm">
            <div className="flex flex-wrap items-center gap-2">
              <span className={p.available ? "text-accent" : "text-ink-2"}>{p.available ? "Available" : "Unavailable"}</span>
              {p.meta_model && <span className="chip border-accent-mark/40 text-accent">Meta model</span>}
              {p.model_id && <span className="font-mono text-xs text-ink-2 break-all">{p.model_id}</span>}
            </div>
            {p.model_version && detailed && <div className="font-mono text-xs text-muted break-all">revision {p.model_version}</div>}
            {p.license && detailed && <div className="text-xs text-muted">licence {p.license} · execution {p.execution}</div>}
            {!p.available && p.reason_unavailable && <div className="text-xs text-ink-2">{p.reason_unavailable}</div>}
            {!p.available && p.action_required && <div className="font-mono text-xs text-muted">{p.action_required}</div>}
          </div>
        </li>
      ))}
    </ul>
  );
}
