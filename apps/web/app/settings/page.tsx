"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { ProviderList } from "@/components/ProviderList";

export default function SettingsPage() {
  const caps = useQuery({ queryKey: ["capabilities"], queryFn: api.capabilities });
  if (caps.isError) return <p className="text-sm text-danger">The API is not reachable.</p>;
  if (!caps.data) return <p className="text-sm text-muted">Loading…</p>;
  const c = caps.data;
  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">Settings and readiness</h1>
      <p className="text-sm text-ink-2">
        Settings are configured server-side through environment variables (see <span className="font-mono">.env.example</span>) and shown here read-only, so no browser
        can enable external transmission or raise limits.
      </p>
      <section className="panel p-4" aria-label="Provider readiness">
        <h2 className="text-sm font-semibold">Provider readiness</h2>
        <ProviderList providers={c.providers} detailed />
      </section>
      <section className="panel space-y-2 p-4 text-sm text-ink-2" aria-label="Inference mode">
        <h2 className="text-sm font-semibold text-ink">Inference mode</h2>
        <p>
          Current mode: <strong className="text-ink" data-testid="settings-mode">{c.mode}</strong> · deployment: <strong className="text-ink">{c.deployment}</strong>
          {c.deployment === "local" && " (local-only: no authentication is enforced; do not expose publicly)"}
        </p>
        <p>
          <span className="font-mono">OVASCOPE_INFERENCE_MODE=no_model</span> forces NO_MODEL. In <span className="font-mono">auto</span>, RESEARCH_INFERENCE requires an
          integrity-checked bundle and local DINOv2 weights.
        </p>
        {c.missing_requirements.length > 0 && (
          <ul className="list-disc pl-5">
            {c.missing_requirements.map((m) => (
              <li key={m.component}>
                {m.component}: {m.requirement} <span className="font-mono text-xs text-muted">{m.action}</span>
              </li>
            ))}
          </ul>
        )}
        <p>
          Worker: {c.worker.online ? "online" : "offline"}
          {c.worker.inline ? " (inline thread)" : ""} · concurrency {c.worker.concurrency}
        </p>
      </section>
      <section className="panel space-y-2 p-4 text-sm text-ink-2" aria-label="External transmission">
        <h2 className="text-sm font-semibold text-ink">External transmission policy</h2>
        <p data-testid="external-policy">
          {c.external_transmission.allowed ? "Allowed" : "Disabled"} ({c.external_transmission.policy}). {c.external_transmission.reason}
        </p>
      </section>
      <section className="panel space-y-2 p-4 text-sm text-ink-2" aria-label="Retention">
        <h2 className="text-sm font-semibold text-ink">Retention and deletion</h2>
        <p>Uploaded studies and every derived artifact are deleted automatically after {c.retention.hours} hours.</p>
        <p>{c.retention.deletion}.</p>
      </section>
      <section className="panel space-y-2 p-4 text-sm text-ink-2" aria-label="Budget and limits">
        <h2 className="text-sm font-semibold text-ink">Budget and limits</h2>
        <p>
          Hosted provider calls: {c.budget.hosted_calls_enabled ? "enabled" : "disabled"} · monthly cap ${c.budget.hosted_provider_monthly_cap_usd.toFixed(2)}
        </p>
        <ul className="grid gap-1 sm:grid-cols-2">
          {Object.entries(c.limits).map(([k, v]) => (
            <li key={k} className="tabular">
              <span className="text-muted">{k.replaceAll("_", " ")}:</span> {String(v)}
            </li>
          ))}
        </ul>
        <p>Accepted formats: {c.accepted_formats.join(", ")}. DICOM and video are not supported in this version.</p>
      </section>
    </div>
  );
}
