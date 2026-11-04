"use client";

import type { AnalysisResult, ArtifactRef, QualityCheck } from "@/lib/types";
import {
  calibrationExplanation,
  formatScore,
  resultStatusCopy,
  scoreHeading,
  segmentationCopy,
} from "@/lib/wording";

const TONE: Record<string, string> = {
  accent: "border-accent-mark/60 bg-accent-deep/15",
  warning: "border-warn/60 bg-warn/10",
  error: "border-danger/60 bg-danger/10",
  neutral: "border-line-strong bg-raised",
};

export function StatusBanner({ result }: { result: AnalysisResult }) {
  const c = resultStatusCopy(result.status);
  return (
    <section
      role="status"
      data-testid="result-status"
      data-status={result.status}
      className={`rounded-xl border px-4 py-3 ${TONE[c.tone]}`}
    >
      <h2 className={`text-sm font-semibold ${c.tone === "error" ? "text-danger" : c.tone === "warning" ? "text-warn-ink" : "text-ink"}`}>
        {c.title}
      </h2>
      <p className="mt-1 text-sm text-ink-2">{c.body}</p>
      {result.error && (
        <p className="mt-2 font-mono text-xs text-danger" data-testid="result-error">
          {result.error.code}: {result.error.message}
        </p>
      )}
      {result.abstention_reason_or_null && <p className="mt-2 text-sm text-warn-ink">{result.abstention_reason_or_null}</p>}
      {result.warnings.length > 0 && (
        <ul className="mt-2 list-disc pl-5 text-sm text-warn-ink">
          {result.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function MissingRequirements({ result }: { result: AnalysisResult }) {
  if (!result.missing_requirements.length) return null;
  return (
    <section className="panel p-4" data-testid="missing-requirements" aria-label="Missing requirements">
      <h2 className="text-sm font-semibold">What is missing</h2>
      <ul className="mt-2 space-y-2 text-sm">
        {result.missing_requirements.map((m) => (
          <li key={m.component + m.requirement} className="rounded-md border border-line bg-raised px-3 py-2">
            <span className="font-medium text-ink">{m.component}</span>
            <span className="text-ink-2">: {m.requirement}</span>
            <div className="mt-1 font-mono text-xs text-muted">{m.action}</div>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Row({ k, v, mono }: { k: string; v: React.ReactNode; mono?: boolean }) {
  return (
    <div className="grid grid-cols-[150px_1fr] gap-2 py-1 text-sm">
      <dt className="text-muted">{k}</dt>
      <dd className={`min-w-0 break-words text-ink-2 ${mono ? "font-mono text-xs" : ""}`}>{v ?? "—"}</dd>
    </div>
  );
}

export function SegmentationPanel({ result }: { result: AnalysisResult }) {
  const s = result.segmentation;
  return (
    <section className="panel p-4" aria-label="Segmentation" data-testid="segmentation-panel" data-seg-status={s?.status ?? "none"}>
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">Segmentation mask</h2>
        <span className="chip">{s?.status.replaceAll("_", " ") ?? "not run"}</span>
      </div>
      <p className="mt-1 text-sm text-ink-2">{segmentationCopy(s)}</p>
      {s && s.status !== "not_run" && (
        <dl className="mt-2">
          <Row k="Mask source" v={s.status === "succeeded" ? `${s.provider_id} · ${s.model_id}` : "no mask"} />
          <Row k="Prompt" v={s.prompt_source ? `${s.prompt_source}${s.automatic ? "" : " (manual)"}` : null} />
          <Row k="Prompt box (px)" v={s.prompt_box_xyxy?.join(", ")} mono />
          <Row k="Localiser max score" v={s.localizer_max_score?.toFixed(2)} />
          <Row k="SAM IoU self-estimate" v={s.sam_predicted_iou != null ? `${s.sam_predicted_iou.toFixed(2)} (SAM's own estimate; not validated)` : null} />
          <Row k="Mask area" v={s.mask_area_fraction != null ? `${(s.mask_area_fraction * 100).toFixed(1)}% of frame` : null} />
          {s.message && <Row k="Note" v={s.message} />}
        </dl>
      )}
      {result.segmentation_artifact_or_null && <Limitation a={result.segmentation_artifact_or_null} />}
    </section>
  );
}

export function ClassificationPanel({ result }: { result: AnalysisResult }) {
  const c = result.classification;
  if (!c) {
    return (
      <section className="panel p-4" aria-label="Classifier output" data-testid="classification-panel" data-has-scores="false">
        <h2 className="text-sm font-semibold">Classifier output</h2>
        <p className="mt-1 text-sm text-ink-2">No classifier output. {result.status === "model_unavailable" ? "No trained model is configured." : "See status above."}</p>
      </section>
    );
  }
  const abstained = result.abstention_status === "abstained";
  const max = Math.max(...c.scores.map((s) => s.score), 1e-6);
  return (
    <section className="panel p-4" aria-label="Classifier output" data-testid="classification-panel" data-has-scores="true">
      <h2 className="text-sm font-semibold">Classifier output</h2>
      <p className="mt-1 text-xs text-muted">
        Target: {result.classification_target} (eight MMOTU dataset categories) · pathway <span className="font-mono">{c.pathway}</span>
      </p>
      <p className="mt-1 text-xs text-ink-2">{c.pathway_reason}</p>
      {abstained ? (
        <p className="mt-3 rounded-md border border-warn/50 bg-warn/10 px-3 py-2 text-sm text-warn-ink">
          Abstained: no category is reported. Scores are listed for research inspection only.
        </p>
      ) : (
        <p className="mt-3 text-sm">
          <span className="text-muted">Highest-scoring MMOTU category: </span>
          <span className="font-semibold text-ink" data-testid="top-category">
            {c.highest_scoring_display}
          </span>
        </p>
      )}
      <h3 className="mt-3 text-xs font-medium uppercase tracking-wide text-muted" data-testid="score-heading">
        {scoreHeading(result.calibration_status)}
      </h3>
      <ul className="mt-2 space-y-1.5" aria-label="Category scores">
        {c.scores.map((s) => (
          <li key={s.label} className="grid grid-cols-[minmax(0,1fr)_120px_44px] items-center gap-2 text-sm">
            <span className="truncate text-ink-2" title={s.display}>
              {s.display}
            </span>
            <span className="h-2 rounded-full bg-line" aria-hidden>
              <span className="block h-2 rounded-full bg-accent-mark" style={{ width: `${(s.score / max) * 100}%` }} />
            </span>
            <span className="tabular text-right text-ink">{formatScore(s.score)}</span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs text-muted">{calibrationExplanation(result.calibration_status)}</p>
    </section>
  );
}

export function AttributionPanel({ result }: { result: AnalysisResult }) {
  const a = result.attribution_artifact_or_null;
  return (
    <section className="panel p-4" aria-label="Attribution map" data-testid="attribution-panel">
      <h2 className="text-sm font-semibold">Attribution map</h2>
      {a ? (
        <>
          <p className="mt-1 text-sm text-ink-2">{a.description}. Teal: pushes toward the highest-scoring category; violet: away from it.</p>
          <Limitation a={a} />
        </>
      ) : (
        <p className="mt-1 text-sm text-ink-2">No attribution map for this result.</p>
      )}
    </section>
  );
}

function Limitation({ a }: { a: ArtifactRef }) {
  return (
    <p className="mt-2 rounded-md border border-line bg-raised px-3 py-2 text-xs text-ink-2">
      <span className="font-medium text-ink">Source:</span> {a.source}. <span className="font-medium text-ink">Limitation:</span> {a.limitations}
    </p>
  );
}

const STATUS_STYLE: Record<QualityCheck["status"], string> = {
  pass: "text-good",
  info: "text-ink-2",
  warn: "text-warn-ink",
  fail: "text-danger",
  not_run: "text-muted",
};

export function QualityPanel({ result }: { result: AnalysisResult }) {
  const groups: QualityCheck["category"][] = ["technical", "image_quality", "content", "distribution_shift"];
  const names = { technical: "Technical validity", image_quality: "Image-quality heuristics", content: "Content", distribution_shift: "Distribution-shift heuristics" };
  return (
    <section className="panel p-4" aria-label="Quality checks" data-testid="quality-panel">
      <h2 className="text-sm font-semibold">Input checks</h2>
      <p className="mt-1 text-xs text-muted">Heuristics only. None of these establishes clinical image adequacy; none detects unfamiliar disease.</p>
      {groups.map((g) => {
        const items = result.quality_checks.filter((q) => q.category === g);
        if (!items.length) return null;
        return (
          <div key={g} className="mt-3">
            <h3 className="text-xs font-medium uppercase tracking-wide text-muted">{names[g]}</h3>
            <ul className="mt-1 space-y-1">
              {items.map((q) => (
                <li key={q.id} className="flex gap-2 text-sm">
                  <span className={`w-12 shrink-0 text-xs font-semibold uppercase ${STATUS_STYLE[q.status]}`}>{q.status}</span>
                  <span className="text-ink-2">{q.message}</span>
                </li>
              ))}
            </ul>
          </div>
        );
      })}
    </section>
  );
}

export function ProvenancePanel({ result }: { result: AnalysisResult }) {
  const ev = result.evaluation_reference as Record<string, unknown> | null;
  return (
    <section className="panel p-4" aria-label="Provenance" data-testid="provenance-panel">
      <h2 className="text-sm font-semibold">Model and data provenance</h2>
      <dl className="mt-2">
        <Row k="Result schema" v={result.schema_version} mono />
        <Row k="Input SHA-256" v={result.input_sha256} mono />
        <Row k="Model" v={result.model_id ?? "none (no model configured)"} />
        <Row k="Model version" v={result.model_version} mono />
        <Row k="Checkpoint SHA-256" v={result.checkpoint_sha256} mono />
        <Row k="Segmentation" v={result.segmentation_provider} />
        <Row k="Calibration" v={result.calibration_status.replaceAll("_", " ")} />
        {ev && (
          <>
            <Row k="Evaluation" v={`${String(ev.experiment_id)} · ${String(ev.evaluation_unit)}-level · patient-independent: ${String(ev.patient_independent)}`} />
            <Row k="Metrics SHA-256" v={String(ev.metrics_sha256)} mono />
            <Row k="External validation" v={String(ev.external_validation)} />
          </>
        )}
        <Row k="Created" v={new Date(result.created_at).toLocaleString()} />
        <Row
          k="Processing (ms)"
          v={Object.entries(result.processing_times)
            .map(([k, v]) => `${k} ${Math.round(v)}`)
            .join(" · ")}
        />
      </dl>
    </section>
  );
}
