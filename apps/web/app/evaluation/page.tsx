"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, evalCaseImageUrl } from "@/lib/api";
import type { Diff, Evaluation, FailureCase, ModelEval } from "@/lib/types";
import { categoryDisplay, MODEL_NAMES } from "@/lib/wording";
import { ConfusionMatrix, DataTable, DotCIChart, LineChart } from "@/components/charts";

const ORDER = ["majority", "handcrafted", "resnet50_imagenet", "dinov2_whole", "dinov2_roi_coarse", "dinov2_roi_sam", "oracle_dinov2_roi_gt"];
const f3 = (v: number | null | undefined) => (v == null ? "n/a" : v.toFixed(3));

export default function EvaluationPage() {
  const q = useQuery({ queryKey: ["evaluation"], queryFn: api.evaluation });
  if (q.isLoading) return <p className="text-sm text-muted">Loading evaluation artifacts…</p>;
  if (q.isError || !q.data) return <p className="text-sm text-danger">Evaluation artifacts could not be loaded.</p>;
  const ev = q.data.evaluation;
  const audit = q.data.data_audit;
  if (!ev.available) {
    return (
      <div className="mx-auto max-w-3xl space-y-4" data-testid="evaluation-unavailable">
        <h1 className="text-2xl font-semibold tracking-tight">Evaluation</h1>
        <p className="panel p-4 text-sm text-ink-2">{ev.reason}</p>
        <p className="text-sm text-muted">No metrics are shown until a reproducible evaluation artifact exists. No sample or placeholder metrics are displayed.</p>
        {audit && <AuditSection audit={audit} />}
      </div>
    );
  }
  return <EvaluationBody ev={ev} audit={audit} mirror={q.data.mirror_verification} />;
}

function EvaluationBody({ ev, audit, mirror }: { ev: Evaluation; audit: Record<string, any> | null; mirror: Record<string, unknown> | null }) { // eslint-disable-line @typescript-eslint/no-explicit-any
  const sc = ev.scope!;
  const models = ev.models!;
  const primary = ev.primary_pathway!;
  const pm = models[primary];
  const rows = ORDER.filter((k) => models[k]);
  return (
    <div className="space-y-6" data-testid="evaluation-page" data-experiment={ev.experiment_id}>
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Held-out evaluation</h1>
          <p className="mt-1 text-sm text-ink-2">
            Experiment <span className="font-mono">{ev.experiment_id}</span> · generated {new Date(ev.generated_at!).toLocaleString()} · metrics SHA-256{" "}
            <span className="font-mono text-xs">{ev.metrics_sha256?.slice(0, 16)}…</span>
          </p>
        </div>
        <span className="chip border-warn/60 text-warn-ink">{sc.external_validation}</span>
      </header>

      <section className="panel grid gap-4 p-4 md:grid-cols-2" aria-label="Evaluation scope" data-testid="eval-scope">
        <div className="space-y-1.5 text-sm text-ink-2">
          <h2 className="text-sm font-semibold text-ink">Scope</h2>
          <p>
            <span className="text-muted">Dataset:</span> {sc.dataset}
          </p>
          <p>
            <span className="text-muted">Test split:</span> {sc.test_split}
          </p>
          <p>
            <span className="text-muted">Evaluation unit:</span> <strong className="text-ink">{sc.evaluation_unit}</strong> · <span className="text-muted">patient-independent:</span>{" "}
            <strong className="text-warn-ink">{String(sc.patient_independent)}</strong>
          </p>
          <p className="text-warn-ink">{sc.statement}</p>
          <p>{sc.cohort_note}</p>
          <p className="text-muted">{sc.ci_note}</p>
          <p className="text-muted">Early-stage analysis: {ev.early_stage?.reason}</p>
        </div>
        <div>
          <h2 className="text-sm font-semibold text-ink">Test class distribution (n = {sc.n_test}; bars relative to the largest class)</h2>
          <ul className="mt-2 space-y-1">
            {Object.entries(sc.test_class_distribution).map(([k, n]) => (
              <li key={k} className="grid grid-cols-[minmax(0,1fr)_160px_32px] items-center gap-2 text-sm">
                <span className="truncate text-ink-2">{categoryDisplay(k)}</span>
                <span className="h-2 rounded-full bg-line" aria-hidden>
                  <span className="block h-2 rounded-full bg-accent-mark" style={{ width: `${(n / Math.max(...Object.values(sc.test_class_distribution))) * 100}%` }} />
                </span>
                <span className="tabular text-right">{n}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="panel space-y-4 p-4" aria-label="Model comparison">
        <h2 className="text-sm font-semibold">Primary target: eight MMOTU categories (image level)</h2>
        <p className="text-xs text-muted">
          Primary deployed pathway: <span className="font-mono">{primary}</span>. {ev.primary_rule}
        </p>
        <div className="grid gap-6 lg:grid-cols-2">
          <DotCIChart
            title="Balanced accuracy (95% CI)"
            reference={{ x: 1 / 8, label: "chance 0.125" }}
            rows={rows.map((k) => ({ id: k, label: MODEL_NAMES[k] ?? k, point: models[k].mmotu8.balanced_accuracy_ci.point, ci: models[k].mmotu8.balanced_accuracy_ci.ci95, muted: models[k].role === "oracle", note: models[k].role === "oracle" ? "uses reference masks at test time (upper bound)" : undefined }))}
          />
          <DotCIChart
            title="Macro one-vs-rest AUROC (95% CI)"
            reference={{ x: 0.5, label: "0.5" }}
            rows={rows.filter((k) => k !== "majority").map((k) => ({ id: k, label: MODEL_NAMES[k] ?? k, point: models[k].mmotu8.macro_auroc_ci.point, ci: models[k].mmotu8.macro_auroc_ci.ci95, muted: models[k].role === "oracle" }))}
          />
        </div>
        <ModelTable models={models} rows={rows} />
      </section>

      <PrimaryDetail m={pm} name={primary} />
      <HgsSection models={models} rows={rows} primary={primary} />
      <AblationSection ev={ev} />
      <SegmentationSection seg={ev.segmentation!} />
      <SensitivitySection s={ev.sensitivity_analyses!} />
      <FailureSection fc={ev.failure_cases!} />
      {audit && <AuditSection audit={audit} mirror={mirror} />}
    </div>
  );
}

function ModelTable({ models, rows }: { models: Record<string, ModelEval>; rows: string[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm tabular" data-testid="model-table">
        <thead>
          <tr className="text-left text-xs text-muted">
            {["Model", "Role", "Bal. acc.", "Macro AUROC", "Accuracy", "Top-label ECE", "Coverage at frozen abstention", "Retained accuracy", "Fallbacks"].map((h) => (
              <th key={h} className="border-b border-line px-2 py-1.5 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((k) => {
            const m = models[k];
            return (
              <tr key={k} className={m.role === "oracle" ? "text-muted" : "text-ink-2"}>
                <td className="border-b border-line/60 px-2 py-1">{MODEL_NAMES[k] ?? k}</td>
                <td className="border-b border-line/60 px-2 py-1">{m.role}</td>
                <td className="border-b border-line/60 px-2 py-1">{f3(m.mmotu8.balanced_accuracy)}</td>
                <td className="border-b border-line/60 px-2 py-1">{f3(m.mmotu8.macro_auroc_ovr)}</td>
                <td className="border-b border-line/60 px-2 py-1">{f3(m.mmotu8.accuracy)}</td>
                <td className="border-b border-line/60 px-2 py-1">{f3(m.mmotu8.calibration.top_label_ece_10bin)}</td>
                <td className="border-b border-line/60 px-2 py-1">{f3(m.abstention.coverage as number)}</td>
                <td className="border-b border-line/60 px-2 py-1">{f3(m.abstention.accuracy_retained as number)}</td>
                <td className="border-b border-line/60 px-2 py-1">{m.test_fallback_to_whole_image}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-1 text-xs text-muted">Fallbacks: test images where the ROI pathway had no automatic region and the whole-image head was used (counted in that row’s metrics).</p>
    </div>
  );
}

function PrimaryDetail({ m, name }: { m: ModelEval; name: string }) {
  const classes = Object.keys(m.per_class_curves ?? {});
  const [cls, setCls] = useState(classes[classes.length - 1] ?? "");
  const curve = m.per_class_curves?.[cls];
  const cal = m.mmotu8.calibration;
  return (
    <section className="panel space-y-4 p-4" aria-label="Primary pathway detail">
      <h2 className="text-sm font-semibold">
        Primary pathway detail: {MODEL_NAMES[name] ?? name}
      </h2>
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <ConfusionMatrix labels={m.mmotu8.confusion_matrix.labels} matrix={m.mmotu8.confusion_matrix.rows_true_cols_pred} display={categoryDisplay} />
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-xs tabular" data-testid="per-class-table">
            <thead>
              <tr className="text-left text-muted">
                {["Category", "n", "Prevalence", "Sens.", "Spec.", "PPV", "NPV", "AUROC", "AUPRC"].map((h) => (
                  <th key={h} className="border-b border-line px-1.5 py-1 font-medium">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {m.mmotu8.per_class.map((p) => (
                <tr key={p.label} className="text-ink-2">
                  <td className="border-b border-line/60 px-1.5 py-1">{categoryDisplay(p.label).split(" (")[0]}</td>
                  <td className="border-b border-line/60 px-1.5 py-1">{p.n}</td>
                  <td className="border-b border-line/60 px-1.5 py-1">{f3(p.prevalence)}</td>
                  <td className="border-b border-line/60 px-1.5 py-1">{f3(p.sensitivity)}</td>
                  <td className="border-b border-line/60 px-1.5 py-1">{f3(p.specificity)}</td>
                  <td className="border-b border-line/60 px-1.5 py-1">{f3(p.ppv)}</td>
                  <td className="border-b border-line/60 px-1.5 py-1">{f3(p.npv)}</td>
                  <td className="border-b border-line/60 px-1.5 py-1">{f3(p.auroc_ovr)}</td>
                  <td className="border-b border-line/60 px-1.5 py-1">{f3(p.auprc_ovr)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1 text-xs text-muted">Sensitivity/specificity/PPV/NPV use the highest-scoring category, one-vs-rest, within this enriched cohort.</p>
        </div>
      </div>
      <div>
        <label className="text-xs text-muted">
          One-vs-rest curves for{" "}
          <select value={cls} onChange={(e) => setCls(e.target.value)} className="rounded border border-line-strong bg-raised px-2 py-1 text-xs text-ink" aria-label="Category for ROC and PR curves">
            {classes.map((c) => (
              <option key={c} value={c}>
                {categoryDisplay(c)}
              </option>
            ))}
          </select>
        </label>
        {curve && (
          <div className="mt-3 grid gap-6 md:grid-cols-2 xl:grid-cols-4">
            {curve.roc_curve && (
              <LineChart title={`ROC · AUROC ${f3(curve.auroc)}`} xLabel="False-positive rate" yLabel="Sensitivity" series={[{ id: "roc", label: "ROC", points: curve.roc_curve }]} reference={{ kind: "diagonal" }} stepped />
            )}
            {curve.pr_curve && (
              <LineChart title={`Precision–recall · AUPRC ${f3(curve.auprc)}`} xLabel="Recall" yLabel="Precision" series={[{ id: "pr", label: "PR", points: curve.pr_curve }]} reference={{ kind: "horizontal", y: curve.prevalence, label: `prevalence ${curve.prevalence.toFixed(3)}` }} />
            )}
            <LineChart
              title={`Reliability (top-label ECE ${f3(cal.top_label_ece_10bin)})`}
              xLabel="Mean top score"
              yLabel="Accuracy"
              series={[{ id: "rel", label: "bins", points: cal.reliability_bins.map((b) => [b.mean_score, b.accuracy] as [number, number]) }]}
              reference={{ kind: "diagonal" }}
            />
            <LineChart
              title="Risk–coverage (selective prediction)"
              xLabel="Coverage"
              yLabel="Accuracy"
              series={[{ id: "rc", label: "accuracy", points: m.risk_coverage.map((p) => [p.coverage, p.accuracy] as [number, number]) }]}
              marker={{ x: m.abstention.coverage as number, y: m.abstention.accuracy_retained as number, label: "frozen threshold" }}
            />
          </div>
        )}
        <p className="mt-2 text-xs text-muted">
          Scores are temperature-scaled on development data; heads were trained with class-balanced weights, so these are not prevalence-calibrated probabilities.
        </p>
      </div>
    </section>
  );
}

function HgsSection({ models, rows, primary }: { models: Record<string, ModelEval>; rows: string[]; primary: string }) {
  const h = models[primary].hgs_vs_other;
  return (
    <section className="panel space-y-4 p-4" aria-label="Secondary target">
      <h2 className="text-sm font-semibold">Secondary analysis: MMOTU “High grade serous” vs all other categories</h2>
      <p className="rounded-md border border-warn/50 bg-warn/10 px-3 py-2 text-sm text-warn-ink">
        Only {h.n_positive} positive test images (prevalence {h.prevalence.toFixed(3)}), possibly from a few patients. This is not a general malignancy detector:
        MMOTU has no borderline tumours, no other invasive histotypes, and no stage labels.
      </p>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.2fr)]">
        {h.roc_curve && (
          <LineChart title={`ROC (primary pathway) · AUROC ${f3(h.auroc)}`} xLabel="False-positive rate" yLabel="Sensitivity" series={[{ id: "h", label: "ROC", points: h.roc_curve }]} reference={{ kind: "diagonal" }} stepped marker={h.sensitivity != null && h.specificity != null ? { x: 1 - h.specificity, y: h.sensitivity, label: "frozen threshold" } : null} />
        )}
        {h.pr_curve && (
          <LineChart title={`Precision–recall · AUPRC ${f3(h.auprc)}`} xLabel="Recall" yLabel="Precision" series={[{ id: "p", label: "PR", points: h.pr_curve }]} reference={{ kind: "horizontal", y: h.prevalence, label: `prevalence ${h.prevalence.toFixed(3)}` }} />
        )}
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-xs tabular">
            <thead>
              <tr className="text-left text-muted">
                {["Model", "AUROC (95% CI)", "Sens.", "Spec.", "PPV", "NPV"].map((x) => (
                  <th key={x} className="border-b border-line px-1.5 py-1 font-medium">
                    {x}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((k) => {
                const v = models[k].hgs_vs_other;
                const ci = v.auroc_ci.ci95;
                return (
                  <tr key={k} className={models[k].role === "oracle" ? "text-muted" : "text-ink-2"}>
                    <td className="border-b border-line/60 px-1.5 py-1">{MODEL_NAMES[k] ?? k}</td>
                    <td className="border-b border-line/60 px-1.5 py-1">
                      {f3(v.auroc)}
                      {ci && ` (${ci[0].toFixed(2)}–${ci[1].toFixed(2)})`}
                    </td>
                    <td className="border-b border-line/60 px-1.5 py-1">{f3(v.sensitivity)}</td>
                    <td className="border-b border-line/60 px-1.5 py-1">{f3(v.specificity)}</td>
                    <td className="border-b border-line/60 px-1.5 py-1">{f3(v.ppv)}</td>
                    <td className="border-b border-line/60 px-1.5 py-1">{f3(v.npv)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="mt-1 text-xs text-muted">Operating thresholds: {h.threshold_source}. PPV/NPV apply only to this enriched cohort.</p>
        </div>
      </div>
    </section>
  );
}

function verdict(d: Diff): string {
  if (d.difference == null || !d.ci95) return "Not estimable";
  if (d.ci95[0] > 0) return "A higher (CI excludes 0)";
  if (d.ci95[1] < 0) return "B higher (CI excludes 0)";
  return "No clear difference (CI includes 0)";
}

function AblationSection({ ev }: { ev: Evaluation }) {
  const metric = [
    ["balanced_accuracy", "Balanced accuracy"],
    ["macro_auroc", "Macro AUROC"],
    ["hgs_auroc", "HGS AUROC"],
  ] as const;
  return (
    <section className="panel space-y-3 p-4" aria-label="Meta ablations" data-testid="ablations">
      <h2 className="text-sm font-semibold">What changed when this Meta component was removed?</h2>
      <p className="text-xs text-muted">Paired image-level bootstrap on the held-out set (A − B). Negative or null results are reported as measured.</p>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-xs tabular">
          <thead>
            <tr className="text-left text-muted">
              {["Question", "A", "B", "Metric", "A − B", "95% CI", "Reading"].map((x) => (
                <th key={x} className="border-b border-line px-1.5 py-1 font-medium">
                  {x}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ev.ablations!.flatMap((a) =>
              metric.map(([k, label], i) => {
                const d = a[k];
                return (
                  <tr key={a.question + k} className="text-ink-2">
                    {i === 0 && (
                      <td rowSpan={3} className="max-w-[280px] border-b border-line/60 px-1.5 py-1 align-top text-ink">
                        {a.question}
                      </td>
                    )}
                    {i === 0 && (
                      <td rowSpan={3} className="border-b border-line/60 px-1.5 py-1 align-top">
                        {MODEL_NAMES[a.a]}
                      </td>
                    )}
                    {i === 0 && (
                      <td rowSpan={3} className="border-b border-line/60 px-1.5 py-1 align-top">
                        {MODEL_NAMES[a.b]}
                      </td>
                    )}
                    <td className="border-b border-line/60 px-1.5 py-1">{label}</td>
                    <td className="border-b border-line/60 px-1.5 py-1">{d.difference == null ? "n/a" : (d.difference >= 0 ? "+" : "") + d.difference.toFixed(3)}</td>
                    <td className="border-b border-line/60 px-1.5 py-1">{d.ci95 ? `${d.ci95[0].toFixed(3)} to ${d.ci95[1].toFixed(3)}` : "n/a"}</td>
                    <td className="border-b border-line/60 px-1.5 py-1">{verdict(d)}</td>
                  </tr>
                );
              }),
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function SegmentationSection({ seg }: { seg: Record<string, any> }) { // eslint-disable-line @typescript-eslint/no-explicit-any
  const blocks: Array<[string, any]> = [ // eslint-disable-line @typescript-eslint/no-explicit-any
    ["All test images", seg.all_test_images],
    ["Lesion categories only", seg.lesion_categories_only],
  ];
  const d = seg.sam_minus_coarse_mean_dice;
  return (
    <section className="panel space-y-3 p-4" aria-label="Segmentation evaluation" data-testid="segmentation-eval">
      <h2 className="text-sm font-semibold">Automatic localisation: Dice / IoU against reference masks</h2>
      <p className="text-xs text-muted">
        {seg.reference}. {seg.failure_counting}. SAM model: <span className="font-mono">{seg.sam_model}</span>.
      </p>
      <DotCIChart
        title="Mean Dice, all test images (no interval; see table for median and IQR)"
        rows={[
          { id: "c", label: "DINOv2 localiser only", point: seg.all_test_images.coarse.dice.mean, ci: [seg.all_test_images.coarse.dice.p25, seg.all_test_images.coarse.dice.p75] },
          { id: "s", label: "Localiser box → SAM 2.1", point: seg.all_test_images.sam_auto.dice.mean, ci: [seg.all_test_images.sam_auto.dice.p25, seg.all_test_images.sam_auto.dice.p75] },
          { id: "o", label: "Oracle box → SAM 2.1", point: seg.all_test_images.sam_oracle.dice.mean, ci: [seg.all_test_images.sam_oracle.dice.p25, seg.all_test_images.sam_oracle.dice.p75], muted: true, note: "reference-mask box: upper bound" },
        ]}
      />
      <p className="text-xs text-muted">Whiskers in this chart show the interquartile range of per-image Dice, not a confidence interval.</p>
      <DataTable
        columns={["Subset", "n", "Candidate rate", "Localiser Dice mean/median", "SAM auto Dice mean/median", "SAM oracle-prompt Dice mean/median"]}
        rows={blocks.map(([label, b]) => [
          label,
          b.n,
          f3(b.candidate_rate),
          `${f3(b.coarse.dice.mean)} / ${f3(b.coarse.dice.median)}`,
          `${f3(b.sam_auto.dice.mean)} / ${f3(b.sam_auto.dice.median)}`,
          `${f3(b.sam_oracle.dice.mean)} / ${f3(b.sam_oracle.dice.median)}`,
        ])}
      />
      <p className="text-sm text-ink-2">
        SAM minus localiser mean Dice: <strong className="text-ink">{d.difference >= 0 ? "+" : ""}{d.difference.toFixed(3)}</strong> (95% CI {d.ci95[0].toFixed(3)} to {d.ci95[1].toFixed(3)}, {d.method}). {seg.oracle_prompt_note}
      </p>
    </section>
  );
}

function SensitivitySection({ s }: { s: Record<string, any> }) { // eslint-disable-line @typescript-eslint/no-explicit-any
  const groups = [s.near_duplicate, s.colour_content];
  return (
    <section className="panel space-y-3 p-4" aria-label="Sensitivity analyses">
      <h2 className="text-sm font-semibold">Sensitivity analyses and subgroups (primary pathway)</h2>
      {groups.map((g) => (
        <div key={g.description}>
          <h3 className="text-xs font-medium text-muted">{g.description}</h3>
          <ul className="mt-1 space-y-1 text-sm text-ink-2">
            {Object.values(g)
              .filter((v): v is Record<string, any> => typeof v === "object") // eslint-disable-line @typescript-eslint/no-explicit-any
              .map((v) => (
                <li key={v.subset}>
                  <span className="text-ink">{v.subset}</span> (n={v.n}):{" "}
                  {v.suppressed ? <span className="text-warn-ink">suppressed: {v.reason}</span> : <>balanced accuracy {f3(v.balanced_accuracy)}, accuracy {f3(v.accuracy)}</>}
                  <span className="ml-2 text-xs text-muted">per-class n: {Object.values(v.per_class_n as Record<string, number>).join(" / ")}</span>
                </li>
              ))}
          </ul>
        </div>
      ))}
    </section>
  );
}

function CaseCard({ c }: { c: FailureCase }) {
  const [imgOk, setImgOk] = useState(true);
  return (
    <li className="rounded-md border border-line bg-raised p-2 text-xs text-ink-2">
      {imgOk && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={evalCaseImageUrl(c.image_id)} alt={`Held-out case ${c.image_id}`} className="mb-2 h-28 w-full rounded object-contain bg-black" onError={() => setImgOk(false)} />
      )}
      <div className="font-mono text-ink">{c.image_id}</div>
      <div>reference: {categoryDisplay(c.true)}</div>
      <div>highest-scoring: {categoryDisplay(c.highest_scoring)} ({c.top_score.toFixed(2)})</div>
      <div>HGS score {c.hgs_score.toFixed(2)}{c.used_whole_image_fallback ? " · whole-image fallback" : ""}</div>
    </li>
  );
}

function FailureSection({ fc }: { fc: NonNullable<Evaluation["failure_cases"]> }) {
  const groups: Array<[string, FailureCase[]]> = [
    ["Confident errors (not abstained)", fc.confident_errors],
    ["Abstained", fc.abstained],
    ["High grade serous below frozen threshold", fc.hgs_below_threshold],
  ];
  return (
    <section className="panel space-y-3 p-4" aria-label="Failure cases" data-testid="failure-cases">
      <h2 className="text-sm font-semibold">Failure cases</h2>
      <p className="text-xs text-muted">{fc.selection_rule} Thumbnails appear only when local serving of evaluation images is enabled.</p>
      {groups.map(([label, cases]) => (
        <div key={label}>
          <h3 className="text-xs font-medium text-muted">
            {label} ({cases.length})
          </h3>
          {cases.length === 0 ? (
            <p className="text-sm text-ink-2">None.</p>
          ) : (
            <ul className="mt-1 grid gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
              {cases.map((c) => (
                <CaseCard key={c.image_id + label} c={c} />
              ))}
            </ul>
          )}
        </div>
      ))}
    </section>
  );
}

function AuditSection({ audit, mirror }: { audit: Record<string, any>; mirror?: Record<string, unknown> | null }) { // eslint-disable-line @typescript-eslint/no-explicit-any
  const p = audit.shortcut_audit.metadata_probe;
  const v = audit.validation;
  return (
    <section className="panel space-y-3 p-4" aria-label="Dataset audit" data-testid="data-audit">
      <h2 className="text-sm font-semibold">Dataset audit and shortcut risk</h2>
      <div className="grid gap-4 md:grid-cols-2 text-sm text-ink-2">
        <div className="space-y-1">
          <p>
            Manifest <span className="font-mono text-xs">{audit.manifest_version}</span> · {audit.n_images} images · patients: not released
          </p>
          <p>
            Suspected near-duplicate pairs: {v.near_duplicates.n_pairs} ({v.near_duplicates.n_cross_split_pairs} across development/test). Heuristic; not proof of same patient.
          </p>
          <p>Split unit: {audit.split.unit}; patient-independent: {String(audit.split.patient_independent)}.</p>
          {mirror && (
            <p>
              Mirror check: {String(mirror.identical)} of {String(mirror.official_files_checked)} files fetched from the official source were byte-identical.
            </p>
          )}
        </div>
        <div className="space-y-1">
          <p className="text-ink">Metadata-only probe (frame size, aspect, colour fraction):</p>
          <p>
            8 categories: dev CV {f3(p.mmotu8.dev_grouped_cv_balanced_accuracy)} · test {f3(p.mmotu8.test_balanced_accuracy)} (chance {f3(p.chance_balanced_accuracy_mmotu8)})
          </p>
          <p>
            HGS vs other: dev CV {f3(p.hgs_vs_other.dev_grouped_cv_balanced_accuracy)} · test {f3(p.hgs_vs_other.test_balanced_accuracy)} (chance 0.500)
          </p>
          <p className="text-xs text-muted">{p.interpretation}</p>
        </div>
      </div>
      <p className="text-xs text-warn-ink">{audit.shortcut_audit.burned_in_content}</p>
    </section>
  );
}
