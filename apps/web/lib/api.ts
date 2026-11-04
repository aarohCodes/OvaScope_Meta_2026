// Browser-side API client. All calls go through the Next.js server proxy at
// /backend/*, which adds credentials server-side; no secret reaches the browser.
import type {
  AnalysisResult,
  ApiErrorBody,
  Capabilities,
  EvaluationResponse,
  Job,
  Narrative,
  Provider,
  Study,
} from "./types";

export const BACKEND = "/backend";

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public requestId?: string | null,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BACKEND}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError(0, "NETWORK_ERROR", "The OvaScope API could not be reached.");
  }
  if (!res.ok) {
    let body: ApiErrorBody | null = null;
    try {
      body = (await res.json()) as ApiErrorBody;
    } catch {
      /* non-JSON error */
    }
    throw new ApiError(
      res.status,
      body?.code ?? "HTTP_ERROR",
      body?.message ?? `Request failed (${res.status}).`,
      body?.request_id ?? res.headers.get("x-request-id"),
    );
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<{ status: string; mode: string; worker: { online: boolean } }>("/health"),
  capabilities: () => request<Capabilities>("/api/v1/capabilities"),
  uploadStudy: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<Study>("/api/v1/studies", { method: "POST", body: fd });
  },
  study: (id: string) => request<Study>(`/api/v1/studies/${id}`),
  requestAnalysis: (id: string, opts: { force?: boolean; idempotencyKey?: string } = {}) =>
    request<Job>(`/api/v1/studies/${id}/analysis${opts.force ? "?force=true" : ""}`, {
      method: "POST",
      headers: opts.idempotencyKey ? { "Idempotency-Key": opts.idempotencyKey } : undefined,
    }),
  job: (id: string) => request<Job>(`/api/v1/jobs/${id}`),
  narrativePreview: (id: string) =>
    request<{ provider: Provider; would_send: Record<string, unknown> }>(`/api/v1/studies/${id}/narrative/preview`),
  createNarrative: (id: string) =>
    request<Narrative>(`/api/v1/studies/${id}/narrative`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ consent_external_transmission: true }),
    }),
  cancelJob: (id: string) => request<Job>(`/api/v1/jobs/${id}/cancel`, { method: "POST" }),
  results: (id: string) => request<{ study_id: string; results: AnalysisResult[] }>(`/api/v1/studies/${id}/results`),
  deleteStudy: (id: string) =>
    request<{ deleted: boolean; artifacts_removed: number }>(`/api/v1/studies/${id}`, { method: "DELETE" }),
  evaluation: () => request<EvaluationResponse>("/api/v1/evaluation"),
  models: () => request<Record<string, unknown>>("/api/v1/models"),
  annotations: (id: string) =>
    request<{ annotations: Array<{ id: string; polygon: number[][]; note: string | null; created_at: string }> }>(
      `/api/v1/studies/${id}/annotations`,
    ),
  addAnnotation: (id: string, polygon: number[][], note?: string) =>
    request<{ id: string }>(`/api/v1/studies/${id}/annotations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon, note }),
    }),
};

export const imageUrl = (studyId: string) => `${BACKEND}/api/v1/studies/${studyId}/image`;
export const artifactUrl = (studyId: string, artifactId: string) =>
  `${BACKEND}/api/v1/studies/${studyId}/artifacts/${artifactId}`;
export const reportUrl = (studyId: string, format: "html" | "md" | "json") =>
  `${BACKEND}/api/v1/studies/${studyId}/report?format=${format}`;
export const evalCaseImageUrl = (imageId: string) => `${BACKEND}/api/v1/evaluation/cases/${imageId}/image`;
