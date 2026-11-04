"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { categoryDisplay } from "@/lib/wording";

type AnyRec = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

function Field({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-1 border-b border-line/60 py-2 text-sm sm:grid-cols-[200px_1fr]">
      <dt className="text-muted">{k}</dt>
      <dd className="min-w-0 break-words text-ink-2">{children}</dd>
    </div>
  );
}

export default function CardsPage() {
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const ev = useQuery({ queryKey: ["evaluation"], queryFn: api.evaluation });
  const b = (models.data?.active_bundle ?? null) as AnyRec | null;
  const weights = (models.data?.third_party_weights ?? []) as AnyRec[];
  const notUsed = (models.data?.not_used ?? {}) as Record<string, string>;
  const audit = ev.data?.data_audit as AnyRec | null;
  const evaluation = ev.data?.evaluation;

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight">Model and data cards</h1>

      <section className="panel p-4" aria-label="Model card" data-testid="model-card">
        <h2 className="text-base font-semibold">Model card</h2>
        <dl className="mt-2">
          <Field k="Model">{b ? `${b.model_id} · ${b.model_version}` : <span className="text-warn-ink">No trained model bundle is active ({String(models.data?.bundle_error ?? "")})</span>}</Field>
          {b && (
            <>
              <Field k="Checkpoint SHA-256">
                <span className="font-mono text-xs">{b.checkpoint_sha256}</span>
              </Field>
              <Field k="Bundle">
                <span className="font-mono text-xs">{b.bundle_id}</span> (created {new Date(b.created_at).toLocaleString()})
              </Field>
            </>
          )}
          <Field k="Intended use">
            Research analysis of already-acquired ovarian/adnexal ultrasound stills by researchers or clinicians evaluating an investigational model.
          </Field>
          <Field k="Prohibited uses">
            Population screening; diagnosis or exclusion of cancer; treatment, surgical or follow-up decisions; replacing clinical examination, pathology or specialist
            assessment; use on patients outside a research protocol; any claim of clinical validation or regulatory clearance.
          </Field>
          <Field k="Classification target">
            {b ? b.target_description : "mmotu8: the eight MMOTU OTU_2d categories (image level)."}
            <ul className="mt-1 grid gap-x-4 sm:grid-cols-2">
              {(b?.class_mapping ?? []).map((c: AnyRec) => (
                <li key={c.label} className="text-xs">
                  {c.index}: {c.display} <span className="text-muted">(source code name “{c.source_code_name}”)</span>
                </li>
              ))}
            </ul>
          </Field>
          <Field k="Architecture">
            Frozen DINOv2 ViT-S/14 features → logistic-regression heads (no fine-tuning). Automatic ROI: DINOv2 patch-token localiser → box prompt → SAM 2.1 mask → DINOv2
            features of the ROI crop. Heads stored as NumPy arrays (no pickle), integrity-checked at load.
          </Field>
          {b && (
            <>
              <Field k="Pathways">
                Primary: <span className="font-mono">{b.primary_pathway}</span>; fallback: <span className="font-mono">{b.fallback_pathway}</span>. {b.pathway_rule}
              </Field>
              <Field k="Calibration">
                {b.calibration.status}: {b.calibration.description}
              </Field>
              <Field k="Abstention">
                {b.abstention.rule}. Thresholds frozen on development data (target coverage {b.abstention.target_dev_coverage}).
              </Field>
              <Field k="Training data">
                {b.training_data.dataset} · manifest <span className="font-mono text-xs">{String(b.training_data.manifest_sha256).slice(0, 16)}…</span> · split{" "}
                <span className="font-mono text-xs">{String(b.training_data.split_sha256).slice(0, 16)}…</span> · licence: {b.training_data.license}
              </Field>
              <Field k="Evaluation reference">
                {b.evaluation_reference.experiment_id} · {b.evaluation_reference.evaluation_unit}-level · patient-independent {String(b.evaluation_reference.patient_independent)} ·{" "}
                {b.evaluation_reference.external_validation} · metrics <span className="font-mono text-xs">{String(b.evaluation_reference.metrics_sha256).slice(0, 16)}…</span>
              </Field>
            </>
          )}
          <Field k="Known limitations">
            <ul className="list-disc space-y-1 pl-5">
              <li>Single centre, single dataset; external validation not performed.</li>
              <li>No patient identifiers: image-level evaluation only; near-duplicate images span development and test, so held-out metrics are likely optimistic.</li>
              <li>Burned-in annotations (calipers, text labels such as “CY”), device text and colour Doppler remain in images and are possible shortcuts.</li>
              <li>Only {evaluation?.scope?.test_class_distribution?.high_grade_serous ?? "few"} held-out “High grade serous” images; no borderline tumours, no other invasive histotypes, no stage labels.</li>
              <li>Scores are not calibrated probabilities. Attribution maps show model sensitivity, not disease location.</li>
              <li>Generic SAM is not trained on ultrasound; masks can be wrong even when the localiser box is reasonable.</li>
            </ul>
          </Field>
          <Field k="Third-party weights">
            <ul className="space-y-1">
              {weights.map((w) => (
                <li key={w.key} className="text-xs">
                  <span className="font-mono">{w.repo_id}</span>@{String(w.revision).slice(0, 12)} · {w.license} · {w.publisher} · {w.role}
                </li>
              ))}
              {Object.entries(notUsed).map(([k, v]) => (
                <li key={k} className="text-xs text-muted">
                  Not used: <span className="font-mono">{k}</span>: {v}
                </li>
              ))}
              <li className="text-xs text-muted">Comparator (not deployed): torchvision ResNet-50 IMAGENET1K_V2.</li>
            </ul>
          </Field>
        </dl>
      </section>

      <section className="panel p-4" aria-label="Data card" data-testid="data-card">
        <h2 className="text-base font-semibold">Data card</h2>
        {!audit ? (
          <p className="mt-2 text-sm text-ink-2">No dataset audit artifact found.</p>
        ) : (
          <dl className="mt-2">
            <Field k="Dataset">{audit.dataset.name}</Field>
            <Field k="Official source">{audit.dataset.official_source}</Field>
            <Field k="Transport">{audit.dataset.transport}</Field>
            <Field k="Retrieved">{audit.dataset.retrieved_at}</Field>
            <Field k="Licence">{audit.dataset.license}</Field>
            <Field k="Access conditions">{audit.dataset.access_conditions}</Field>
            <Field k="Training use">{audit.dataset.permitted_training_use}</Field>
            <Field k="Redistribution">{audit.dataset.permitted_redistribution}</Field>
            <Field k="De-identification">{audit.dataset.de_identified}</Field>
            <Field k="External API transmission">{audit.dataset.external_api_transmission_authorized ? "authorised" : "not authorised (images never leave this machine)"}</Field>
            <Field k="Images / patients">
              {audit.n_images} images · patients: {audit.n_patients ?? "not released"}. {audit.n_patients_note}
            </Field>
            <Field k="Label policy">
              {audit.label_policy_version}: labels come only from the official *_cls.txt files via an explicit index→category mapping, cross-checked against semantic-mask
              colours. Reference standard: expert annotation referring to pathology reports; per-image reports not released.
            </Field>
            <Field k="Split">{audit.dataset.split_policy}</Field>
            <Field k="Class distribution">
              <table className="text-xs tabular">
                <thead>
                  <tr className="text-muted">
                    <th className="pr-4 text-left font-normal">Category</th>
                    <th className="pr-4 text-right font-normal">Development</th>
                    <th className="text-right font-normal">Test</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(audit.class_distribution.development).map((k: string) => (
                    <tr key={k}>
                      <td className="pr-4">{categoryDisplay(k)}</td>
                      <td className="pr-4 text-right">{audit.class_distribution.development[k]}</td>
                      <td className="text-right">{audit.class_distribution.test[k]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Field>
            <Field k="Stage / histology fields">Stage: none. Histology beyond the category: none. Early-stage analysis: {audit.early_stage_analysis}.</Field>
            <Field k="Manifest">
              <span className="font-mono text-xs">{audit.manifest_sha256}</span>
            </Field>
          </dl>
        )}
      </section>
    </div>
  );
}
