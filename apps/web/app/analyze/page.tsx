"use client";

import { Suspense, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError, artifactUrl, imageUrl, reportUrl } from "@/lib/api";
import type { AnalysisResult, Job } from "@/lib/types";
import { ImageViewer, type OverlayLayer } from "@/components/ImageViewer";
import { UploadDropzone } from "@/components/UploadDropzone";
import { JobProgress } from "@/components/JobProgress";
import {
  AttributionPanel,
  ClassificationPanel,
  MissingRequirements,
  ProvenancePanel,
  QualityPanel,
  SegmentationPanel,
  StatusBanner,
} from "@/components/ResultPanels";

export default function AnalyzePage() {
  return (
    <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
      <Analyze />
    </Suspense>
  );
}

function Analyze() {
  const params = useSearchParams();
  const studyId = params.get("study");
  return studyId ? <Workspace studyId={studyId} /> : <Upload />;
}

function Upload() {
  const router = useRouter();
  const caps = useQuery({ queryKey: ["capabilities"], queryFn: api.capabilities });
  const [error, setError] = useState<string | null>(null);
  const upload = useMutation({
    mutationFn: async (file: File) => {
      const study = await api.uploadStudy(file);
      await api.requestAnalysis(study.id, { idempotencyKey: crypto.randomUUID() });
      return study;
    },
    onSuccess: (study) => router.replace(`/analyze?study=${study.id}`),
    onError: (e) => setError(e instanceof ApiError ? `${e.code}: ${e.message}` : "Upload failed."),
  });
  const maxBytes = Number(caps.data?.limits.max_upload_bytes ?? 20 * 1024 * 1024);
  return (
    <div className="mx-auto max-w-3xl space-y-5">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Analyze an image</h1>
        <p className="mt-1 text-sm text-ink-2">
          Upload one already-acquired, de-identified ovarian/adnexal ultrasound still for research analysis.
        </p>
      </header>
      {caps.data?.mode === "NO_MODEL" && (
        <p className="rounded-lg border border-line-strong bg-raised px-4 py-3 text-sm text-ink-2" data-testid="no-model-notice">
          No trained classifier is configured. Upload, validation and the viewer work, but no score or category will be produced.
        </p>
      )}
      <UploadDropzone maxBytes={maxBytes} disabled={upload.isPending} onFile={(f) => upload.mutate(f)} />
      {upload.isPending && <p className="text-sm text-ink-2" aria-live="polite">Uploading and validating…</p>}
      {error && (
        <p role="alert" data-testid="upload-error" className="rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-danger">
          {error}
        </p>
      )}
      <section className="panel space-y-2 p-4 text-sm text-ink-2" aria-label="Privacy and data handling">
        <h2 className="text-sm font-semibold text-ink">Before you upload</h2>
        <p>
          Use de-identified research images only. Ultrasound frames often contain <strong className="text-warn-ink">burned-in text</strong> (names,
          dates, IDs, device settings). Removing file metadata does not remove burned-in text.
        </p>
        <p>
          Images stay on this server. External transmission is <strong>disabled</strong> (fail closed):{" "}
          {caps.data?.external_transmission.reason ?? "no hosted provider is authorised."}
        </p>
        <p>
          Studies and all derived artifacts are deleted automatically after {caps.data?.retention.hours ?? 24} hours, or immediately with
          “Delete study”. File names are not stored.
        </p>
      </section>
    </div>
  );
}

function Workspace({ studyId }: { studyId: string }) {
  const router = useRouter();
  const qc = useQueryClient();
  const study = useQuery({ queryKey: ["study", studyId], queryFn: () => api.study(studyId), retry: false });
  const jobId = study.data?.latest_job_id ?? null;
  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.job(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) => {
      const s = (q.state.data as Job | undefined)?.status;
      return s === "queued" || s === "running" ? 700 : false;
    },
  });
  const terminal = job.data && !["queued", "running"].includes(job.data.status);
  const results = useQuery({
    queryKey: ["results", studyId, job.data?.id, job.data?.status],
    queryFn: () => api.results(studyId),
    enabled: !!terminal,
  });
  const annotations = useQuery({ queryKey: ["annotations", studyId], queryFn: () => api.annotations(studyId), enabled: !!study.data });
  const result: AnalysisResult | undefined = results.data?.results[0];

  const [vis, setVis] = useState({ segmentation_mask: true, localizer_prior: false, attribution_map: false });
  const [opacity, setOpacity] = useState({ segmentation_mask: 0.85, localizer_prior: 0.6, attribution_map: 0.75 });
  const [showBox, setShowBox] = useState(true);
  const [annotating, setAnnotating] = useState(false);
  const [draft, setDraft] = useState<number[][]>([]);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [actionMsg, setActionMsg] = useState<string | null>(null);

  const overlays: OverlayLayer[] = useMemo(() => {
    if (!result) return [];
    const refs = [result.localizer_artifact_or_null, result.segmentation_artifact_or_null, result.attribution_artifact_or_null];
    return refs
      .filter((r): r is NonNullable<typeof r> => !!r)
      .map((r) => ({
        id: r.kind,
        label: r.description,
        src: artifactUrl(studyId, r.artifact_id),
        visible: vis[r.kind],
        opacity: opacity[r.kind],
      }));
  }, [result, studyId, vis, opacity]);

  const rerun = useMutation({
    mutationFn: () => api.requestAnalysis(studyId, { force: true, idempotencyKey: crypto.randomUUID() }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["study", studyId] }),
  });
  const cancel = useMutation({ mutationFn: (id: string) => api.cancelJob(id), onSuccess: () => qc.invalidateQueries({ queryKey: ["job"] }) });
  const del = useMutation({
    mutationFn: () => api.deleteStudy(studyId),
    onSuccess: (r) => {
      qc.removeQueries({ queryKey: ["study", studyId] });
      router.replace(`/analyze?deleted=${r.artifacts_removed}`);
    },
  });
  const saveAnn = useMutation({
    mutationFn: () => api.addAnnotation(studyId, draft),
    onSuccess: () => {
      setDraft([]);
      setAnnotating(false);
      setActionMsg("Researcher annotation saved. It is stored separately and is not used by the automatic pipeline.");
      qc.invalidateQueries({ queryKey: ["annotations", studyId] });
    },
  });

  if (study.isError) {
    return (
      <div className="mx-auto max-w-xl panel p-6 text-sm" data-testid="study-missing">
        <p className="text-ink">This study does not exist (it may have been deleted or expired).</p>
        <button className="mt-3 rounded bg-accent-deep px-3 py-1.5 text-white" onClick={() => router.replace("/analyze")}>
          Upload another image
        </button>
      </div>
    );
  }
  if (!study.data) return <p className="text-sm text-muted">Loading study…</p>;
  const s = study.data;

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_440px]">
      <div className="flex min-w-0 flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-lg font-semibold">Analysis workspace</h1>
          <span className="chip font-mono">study {s.id.slice(0, 8)}</span>
          <span className="chip tabular">
            {s.image_format} · {s.width}×{s.height}
          </span>
        </div>
        <div className="panel h-[62vh] min-h-[420px] overflow-hidden">
          <ImageViewer
            imageSrc={imageUrl(studyId)}
            natural={{ w: s.width, h: s.height }}
            overlays={overlays}
            promptBox={result?.segmentation?.prompt_box_xyxy ?? null}
            showPromptBox={showBox}
            polygons={annotations.data?.annotations.map((a) => a.polygon) ?? []}
            draftPolygon={draft}
            annotating={annotating}
            onImageClick={(x, y) => setDraft((d) => [...d, [x, y]])}
          />
        </div>
        <OverlayControls
          result={result}
          vis={vis}
          setVis={setVis}
          opacity={opacity}
          setOpacity={setOpacity}
          showBox={showBox}
          setShowBox={setShowBox}
        />
        <section className="panel p-4 text-sm no-print" aria-label="Researcher tools">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="mr-2 text-sm font-semibold">Researcher annotation (manual)</h2>
            <button
              type="button"
              aria-pressed={annotating}
              onClick={() => {
                setAnnotating((a) => !a);
                setDraft([]);
              }}
              className={`rounded border px-3 py-1 text-xs ${annotating ? "border-violet bg-violet/20" : "border-line-strong hover:bg-raised"}`}
            >
              {annotating ? "Stop drawing" : "Draw polygon"}
            </button>
            <button type="button" disabled={draft.length < 3 || saveAnn.isPending} onClick={() => saveAnn.mutate()} className="rounded border border-line-strong px-3 py-1 text-xs disabled:opacity-40">
              Save ({draft.length} pts)
            </button>
            <button type="button" disabled={!draft.length} onClick={() => setDraft((d) => d.slice(0, -1))} className="rounded border border-line-strong px-3 py-1 text-xs disabled:opacity-40">
              Undo point
            </button>
          </div>
          <p className="mt-2 text-xs text-muted">
            Manual research/annotation tool. Saved polygons are labelled as researcher annotations, exported in the report data, and are never fed to the automatic classifier.
            {annotations.data && ` ${annotations.data.annotations.length} saved.`}
          </p>
        </section>
      </div>

      <aside className="flex min-w-0 flex-col gap-3" aria-label="Results">
        {job.data && (!terminal || job.data.status !== "succeeded") && (
          <JobProgress job={job.data} onCancel={() => cancel.mutate(job.data!.id)} />
        )}
        {!jobId && <p className="panel p-4 text-sm text-ink-2">No analysis has been requested for this study.</p>}
        {result && (
          <>
            <StatusBanner result={result} />
            <MissingRequirements result={result} />
            <SegmentationPanel result={result} />
            <ClassificationPanel result={result} />
            <AttributionPanel result={result} />
            <QualityPanel result={result} />
            <ProvenancePanel result={result} />
          </>
        )}
        <section className="panel space-y-3 p-4 text-sm no-print" aria-label="Study actions">
          <h2 className="text-sm font-semibold">Export and manage</h2>
          <div className="flex flex-wrap gap-2">
            {(["html", "md", "json"] as const).map((f) => (
              <a key={f} href={reportUrl(studyId, f)} download className="rounded border border-line-strong px-3 py-1 text-xs hover:bg-raised" data-testid={`export-${f}`}>
                Report .{f}
              </a>
            ))}
            <button type="button" onClick={() => rerun.mutate()} disabled={rerun.isPending || !terminal} className="rounded border border-line-strong px-3 py-1 text-xs hover:bg-raised disabled:opacity-40">
              Re-run analysis
            </button>
          </div>
          {!confirmDelete ? (
            <button type="button" onClick={() => setConfirmDelete(true)} className="rounded border border-danger/60 px-3 py-1 text-xs text-danger hover:bg-danger/10" data-testid="delete-study">
              Delete study
            </button>
          ) : (
            <div className="rounded-md border border-danger/60 bg-danger/10 p-3" role="alertdialog" aria-label="Confirm deletion">
              <p className="text-xs text-ink">Delete this study, its results, overlays and annotations now? This cannot be undone.</p>
              <div className="mt-2 flex gap-2">
                <button type="button" onClick={() => del.mutate()} className="rounded bg-danger px-3 py-1 text-xs font-medium text-black" data-testid="confirm-delete">
                  Delete permanently
                </button>
                <button type="button" onClick={() => setConfirmDelete(false)} className="rounded border border-line-strong px-3 py-1 text-xs">
                  Keep
                </button>
              </div>
            </div>
          )}
          {actionMsg && <p className="text-xs text-ink-2" aria-live="polite">{actionMsg}</p>}
          <p className="text-xs text-muted">Retention: deleted automatically at {new Date(s.retention_expires_at).toLocaleString()}.</p>
        </section>
      </aside>
    </div>
  );
}

type Kind = "segmentation_mask" | "localizer_prior" | "attribution_map";

function OverlayControls(props: {
  result?: AnalysisResult;
  vis: Record<Kind, boolean>;
  setVis: (f: (v: Record<Kind, boolean>) => Record<Kind, boolean>) => void;
  opacity: Record<Kind, number>;
  setOpacity: (f: (v: Record<Kind, number>) => Record<Kind, number>) => void;
  showBox: boolean;
  setShowBox: (b: boolean) => void;
}) {
  const { result, vis, setVis, opacity, setOpacity, showBox, setShowBox } = props;
  const items: Array<{ kind: Kind; name: string; present: boolean; swatch: string }> = [
    { kind: "segmentation_mask", name: "Segmentation mask (SAM 2.1)", present: !!result?.segmentation_artifact_or_null, swatch: "bg-accent-mark" },
    { kind: "localizer_prior", name: "Localiser prior (DINOv2 patches)", present: !!result?.localizer_artifact_or_null, swatch: "bg-cyan-400" },
    { kind: "attribution_map", name: "Attribution map (model sensitivity)", present: !!result?.attribution_artifact_or_null, swatch: "bg-violet" },
  ];
  return (
    <section className="panel p-4 no-print" aria-label="Overlay controls" data-testid="overlay-controls">
      <h2 className="text-sm font-semibold">Overlays</h2>
      <div className="mt-2 grid gap-3 md:grid-cols-3">
        {items.map((it) => (
          <div key={it.kind} className={`rounded-md border border-line p-3 ${it.present ? "" : "opacity-50"}`}>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={vis[it.kind] && it.present}
                disabled={!it.present}
                onChange={(e) => setVis((v) => ({ ...v, [it.kind]: e.target.checked }))}
                data-testid={`toggle-${it.kind}`}
              />
              <span aria-hidden className={`h-2.5 w-2.5 rounded-sm ${it.swatch}`} />
              <span className="text-ink-2">{it.name}</span>
            </label>
            <label className="mt-2 flex items-center gap-2 text-xs text-muted">
              Opacity
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={opacity[it.kind]}
                disabled={!it.present}
                aria-label={`${it.name} opacity`}
                onChange={(e) => setOpacity((o) => ({ ...o, [it.kind]: Number(e.target.value) }))}
                data-testid={`opacity-${it.kind}`}
                className="flex-1"
              />
              <span className="tabular w-8 text-right">{Math.round(opacity[it.kind] * 100)}%</span>
            </label>
            {!it.present && <p className="mt-1 text-xs text-muted">Not produced for this result.</p>}
          </div>
        ))}
      </div>
      <label className="mt-3 flex items-center gap-2 text-xs text-ink-2">
        <input type="checkbox" checked={showBox} onChange={(e) => setShowBox(e.target.checked)} />
        Show automatic SAM prompt box (dashed)
      </label>
    </section>
  );
}
