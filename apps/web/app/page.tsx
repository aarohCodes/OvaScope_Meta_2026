"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { ProviderList } from "@/components/ProviderList";

export default function Overview() {
  const caps = useQuery({ queryKey: ["capabilities"], queryFn: api.capabilities });
  const ev = useQuery({ queryKey: ["evaluation"], queryFn: api.evaluation });
  const evaluation = ev.data?.evaluation;
  const audit = ev.data?.data_audit;
  const live = caps.data?.mode === "RESEARCH_INFERENCE";

  return (
    <div className="space-y-6">
      <section className="grid gap-6 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <div className="space-y-4">
          <h1 className="text-3xl font-semibold tracking-tight text-ink">OvaScope</h1>
          <p className="text-lg text-ink-2">Ovarian ultrasound analysis with inspectable evidence.</p>
          <p className="max-w-2xl text-sm leading-6 text-ink-2">
            A research prototype for ultrasound researchers evaluating an investigational model on already-acquired ovarian/adnexal images. It asks:
            can image-based models distinguish a dataset’s verified ovarian-mass categories, and does Meta-assisted lesion localisation (DINOv2 + SAM 2.1)
            change performance compared with whole-image analysis? Segmentation, classifier output and attribution are shown separately, each with its source.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link href="/analyze" className="rounded-md bg-accent-deep px-4 py-2 text-sm font-medium text-white hover:bg-accent-mark" data-testid="cta-analyze">
              Analyze image
            </Link>
            <Link href="/evaluation" className="rounded-md border border-line-strong px-4 py-2 text-sm text-ink hover:bg-raised">
              Held-out evaluation
            </Link>
            <Link href="/cards" className="rounded-md border border-line-strong px-4 py-2 text-sm text-ink hover:bg-raised">
              Model &amp; data cards
            </Link>
          </div>
        </div>
        <div className="panel space-y-2 p-4 text-sm">
          <h2 className="text-sm font-semibold text-ink">Intended and prohibited use</h2>
          <p className="text-ink-2">
            <span className="text-accent">Intended:</span> research analysis of already-acquired images containing suspected ovarian/adnexal lesions, by researchers or
            clinicians evaluating an investigational model.
          </p>
          <p className="text-ink-2">
            <span className="text-warn-ink">Not intended:</span> population screening; diagnosis or exclusion of cancer; treatment or surgical decisions; replacing
            clinical examination, pathology or specialist assessment.
          </p>
          <p className="text-muted">No clinical validation. No regulatory clearance. Not shown to detect cancer earlier than clinicians.</p>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <div className="panel p-4" aria-label="Model readiness">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Actual model readiness</h2>
            {caps.data && (
              <span className={`chip ${live ? "border-accent-mark/60 text-accent" : ""}`} data-testid="overview-mode">
                {live ? "RESEARCH_INFERENCE" : "NO_MODEL"}
              </span>
            )}
          </div>
          {caps.isError && <p className="mt-2 text-sm text-danger">The API is not reachable. Start it with `python tasks.py api`.</p>}
          {caps.data && (
            <>
              <ProviderList providers={caps.data.providers} />
              {caps.data.active_bundle ? (
                <p className="mt-2 text-xs text-muted">
                  Active bundle <span className="font-mono">{caps.data.active_bundle.bundle_id}</span> · checkpoint{" "}
                  <span className="font-mono">{caps.data.active_bundle.checkpoint_sha256.slice(0, 16)}…</span> · primary pathway{" "}
                  <span className="font-mono">{caps.data.active_bundle.primary_pathway}</span>
                </p>
              ) : (
                <p className="mt-2 text-xs text-ink-2">No model bundle is active: uploads and the viewer work, but no scores are produced.</p>
              )}
            </>
          )}
        </div>
        <div className="panel space-y-2 p-4 text-sm" aria-label="Data readiness">
          <h2 className="text-sm font-semibold">Data readiness</h2>
          {audit ? (
            <>
              <p className="text-ink-2">
                {audit.dataset.name}: {audit.n_images} images, 8 categories. Patients: not released. Split unit:{" "}
                <strong className="text-ink">{audit.split.unit}</strong>; patient-independent: <strong className="text-warn-ink">{String(audit.split.patient_independent)}</strong>.
              </p>
              <p className="text-ink-2">Licence: {audit.dataset.license}. Redistribution: {audit.dataset.permitted_redistribution}.</p>
              <p className="text-ink-2">Stage labels: none, so the early-stage analysis is blocked. No borderline category.</p>
            </>
          ) : (
            <p className="text-ink-2">No dataset audit artifact found. Run `python tasks.py data-audit`.</p>
          )}
          <p className="text-ink-2">
            Evaluation:{" "}
            {evaluation?.available ? (
              <>
                held-out results for <span className="font-mono">{evaluation.experiment_id}</span>.{" "}
                <span className="text-warn-ink">{evaluation.scope?.external_validation}</span>
              </>
            ) : (
              "not available."
            )}
          </p>
          <p className="text-xs text-muted">
            Docs: <span className="font-mono">docs/DATASET_READINESS.md</span>, <span className="font-mono">docs/MODEL_CARD.md</span>,{" "}
            <span className="font-mono">docs/CLAIMS_LEDGER.md</span>
          </p>
        </div>
      </section>
    </div>
  );
}
