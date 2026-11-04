"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { RESEARCH_NOTICE } from "@/lib/wording";

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/analyze", label: "Analyze" },
  { href: "/evaluation", label: "Evaluation" },
  { href: "/cards", label: "Model & data cards" },
  { href: "/settings", label: "Settings" },
];

export function ResearchNotice() {
  return (
    <div
      role="note"
      aria-label="Research notice"
      data-testid="research-notice"
      className="sticky top-0 z-30 border-b border-warn/40 bg-[#2a210b] px-4 py-1.5 text-center text-[13px] font-medium text-warn-ink"
    >
      {RESEARCH_NOTICE}
    </div>
  );
}

export function ModePill() {
  const { data, isError } = useQuery({ queryKey: ["capabilities"], queryFn: api.capabilities, refetchInterval: 15_000 });
  if (isError) {
    return (
      <span className="chip border-danger/60 text-danger" data-testid="mode-pill">
        API unreachable
      </span>
    );
  }
  if (!data) return <span className="chip">Checking…</span>;
  const live = data.mode === "RESEARCH_INFERENCE";
  return (
    <span
      data-testid="mode-pill"
      className={`chip ${live ? "border-accent-mark/60 text-accent" : "border-line-strong text-ink-2"}`}
      title={live ? "A trained, integrity-checked model bundle is active" : "No trained classifier: no scores are produced"}
    >
      <span aria-hidden className={`h-2 w-2 rounded-full ${live ? "bg-accent-mark" : "bg-muted"}`} />
      {live ? "Research inference" : "No model"}
    </span>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  return (
    <>
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-10 focus:z-50 focus:rounded focus:bg-raised focus:px-3 focus:py-2">
        Skip to content
      </a>
      <ResearchNotice />
      <header className="border-b border-line bg-bg/95 no-print">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <Link href="/" className="flex items-baseline gap-2">
            <span className="text-lg font-semibold tracking-tight text-ink">OvaScope</span>
            <span className="hidden text-xs text-muted md:inline">research prototype</span>
          </Link>
          <nav aria-label="Primary" className="flex flex-wrap gap-1">
            {NAV.map((n) => {
              const active = n.href === "/" ? path === "/" : path.startsWith(n.href);
              return (
                <Link
                  key={n.href}
                  href={n.href}
                  aria-current={active ? "page" : undefined}
                  className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
                    active ? "bg-raised text-ink" : "text-ink-2 hover:bg-raised/60 hover:text-ink"
                  }`}
                >
                  {n.label}
                </Link>
              );
            })}
          </nav>
          <div className="ml-auto">
            <ModePill />
          </div>
        </div>
      </header>
      <main id="main" className="mx-auto max-w-[1400px] px-4 py-6">
        {children}
      </main>
      <footer className="mx-auto max-w-[1400px] px-4 pb-10 pt-4 text-xs text-muted">
        {RESEARCH_NOTICE} Not clinically validated. No regulatory clearance. External validation not performed.
      </footer>
    </>
  );
}
