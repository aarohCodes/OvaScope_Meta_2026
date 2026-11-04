"use client";

import type { Job } from "@/lib/types";
import { stageLabel } from "@/lib/wording";

const ORDER = ["validating", "representation", "segmentation", "classification", "attribution"];

export function JobProgress({ job, onCancel }: { job: Job; onCancel?: () => void }) {
  const active = job.status === "queued" || job.status === "running";
  const idx = job.stage ? ORDER.indexOf(job.stage) : -1;
  return (
    <section aria-label="Processing status" className="panel p-4" data-testid="job-progress" data-status={job.status} data-stage={job.stage ?? ""}>
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-ink">Processing</h2>
        <span className="chip tabular" aria-live="polite">
          {stageLabel(job.stage, job.status)}
        </span>
      </div>
      <ol className="mt-3 grid grid-cols-5 gap-1" aria-label="Pipeline stages">
        {ORDER.map((s, i) => {
          const done = job.status === "succeeded" || (idx >= 0 && i < idx);
          const current = active && i === idx;
          return (
            <li key={s} className="flex flex-col gap-1">
              <span
                className={`h-1.5 rounded-full ${done ? "bg-accent-mark" : current ? "bg-accent-mark/50 animate-pulse" : "bg-line"}`}
                aria-hidden
              />
              <span className={`text-[11px] ${current ? "text-ink" : "text-muted"}`}>{stageLabel(s, "running")}</span>
            </li>
          );
        })}
      </ol>
      <p className="mt-3 text-xs text-muted">
        Stage reported by the worker in real time. Attempt {job.attempts} of {job.max_attempts}.
        {job.status === "queued" && " Waiting for the worker."}
      </p>
      {job.error_code && (
        <p className="mt-2 text-xs text-danger">
          {job.error_code}: {job.error_message}
        </p>
      )}
      {active && onCancel && (
        <button type="button" onClick={onCancel} className="mt-3 rounded border border-line-strong px-3 py-1 text-xs hover:bg-raised">
          Cancel analysis
        </button>
      )}
    </section>
  );
}
