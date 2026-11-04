"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, ApiError } from "@/lib/api";
import type { Narrative } from "@/lib/types";

/** Optional narrative text from the Meta Model API. Never sends images; consent per call. */
export function NarrativePanel({ studyId, hasResult }: { studyId: string; hasResult: boolean }) {
  const [showPreview, setShowPreview] = useState(false);
  const [consent, setConsent] = useState(false);
  const preview = useQuery({
    queryKey: ["narrative-preview", studyId],
    queryFn: () => api.narrativePreview(studyId),
    enabled: hasResult,
    retry: false,
  });
  const gen = useMutation({ mutationFn: () => api.createNarrative(studyId) });
  const provider = preview.data?.provider;
  const n: Narrative | undefined = gen.data;

  return (
    <section className="panel space-y-3 p-4 text-sm no-print" aria-label="Narrative summary" data-testid="narrative-panel">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">Narrative summary (optional)</h2>
        <span className={`chip ${provider?.available ? "border-accent-mark/60 text-accent" : ""}`}>
          {provider?.available ? `Meta ${provider.model_id}` : "Template only"}
        </span>
      </div>
      {!hasResult && <p className="text-xs text-muted">Available after an analysis result exists.</p>}
      {provider && !provider.available && (
        <p className="text-xs text-ink-2" data-testid="narrative-unavailable">
          Meta Model API: {provider.reason_unavailable} <span className="font-mono text-muted">{provider.action_required}</span>
        </p>
      )}
      {provider?.available && (
        <>
          <p className="text-xs text-ink-2">
            Rephrases verified structured fields into a short paragraph. It cannot change scores, masks or status. Images, study IDs and hashes are never sent.
          </p>
          <button type="button" className="rounded border border-line-strong px-3 py-1 text-xs hover:bg-raised" onClick={() => setShowPreview((v) => !v)}>
            {showPreview ? "Hide" : "Preview"} exactly what would be sent
          </button>
          {showPreview && (
            <pre className="max-h-56 overflow-auto rounded-md border border-line bg-bg p-2 font-mono text-[11px] text-ink-2" data-testid="narrative-preview">
              {JSON.stringify(preview.data?.would_send, null, 2)}
            </pre>
          )}
          <label className="flex items-start gap-2 text-xs text-ink-2">
            <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} data-testid="narrative-consent" />
            I consent to sending these structured fields (no image) to the Meta Model API for this one request.
          </label>
          <button
            type="button"
            disabled={!consent || gen.isPending}
            onClick={() => gen.mutate()}
            className="rounded bg-accent-deep px-3 py-1.5 text-xs font-medium text-white hover:bg-accent-mark disabled:opacity-40"
          >
            {gen.isPending ? "Generating…" : "Generate narrative"}
          </button>
        </>
      )}
      {gen.isError && (
        <p role="alert" className="text-xs text-danger">
          {gen.error instanceof ApiError ? `${gen.error.code}: ${gen.error.message}` : "Narrative request failed."} No text was substituted.
        </p>
      )}
      {n && (
        <div className="rounded-md border border-line bg-raised p-3" data-testid="narrative-output" data-source={n.source}>
          <p className="text-xs text-muted">
            Source: {n.source === "meta-model-api" ? `${n.model_id} (Meta Model API), output validated` : "deterministic template"}
            {n.fallback_reason ? ` · model output rejected: ${n.fallback_reason}` : ""} · est. cost ${n.estimated_cost_usd.toFixed(4)}
          </p>
          <p className="mt-2 text-ink">{n.text}</p>
          {n.limitations.length > 0 && (
            <ul className="mt-2 list-disc pl-5 text-xs text-ink-2">
              {n.limitations.map((l) => (
                <li key={l}>{l}</li>
              ))}
            </ul>
          )}
          <p className="mt-2 text-xs text-muted">Included in the exported report. Scores, masks and status are unchanged.</p>
        </div>
      )}
    </section>
  );
}
