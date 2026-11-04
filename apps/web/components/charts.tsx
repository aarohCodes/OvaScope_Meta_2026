"use client";

// Hand-written SVG charts. Palette validated on the chart surface #111a22:
// series teal #14a899 (single-series default), violet #8b6cf0 (second series).
// Every chart renders only values passed in from evaluation artifacts.
import { useId, useMemo, useState } from "react";

const SERIES = "#14a899";
const SERIES_2 = "#8b6cf0";
const GRID = "#223140";
const AXIS = "#3a4c5e";
const INK2 = "#b2c0cc";
const MUTED = "#8494a3";
const SURFACE = "#111a22";

interface LineSeries {
  id: string;
  label: string;
  points: [number, number][];
  color?: string;
}

interface LineChartProps {
  title: string;
  xLabel: string;
  yLabel: string;
  series: LineSeries[];
  reference?: { kind: "diagonal" } | { kind: "horizontal"; y: number; label: string };
  marker?: { x: number; y: number; label: string } | null;
  height?: number;
  stepped?: boolean;
}

const W = 420;
const PAD = { l: 44, r: 12, t: 10, b: 36 };

export function LineChart({ title, xLabel, yLabel, series, reference, marker, height = 300, stepped }: LineChartProps) {
  const H = height;
  const iw = W - PAD.l - PAD.r;
  const ih = H - PAD.t - PAD.b;
  const sx = (x: number) => PAD.l + x * iw;
  const sy = (y: number) => PAD.t + (1 - y) * ih;
  const [hover, setHover] = useState<{ s: number; i: number } | null>(null);
  const titleId = useId();
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  const path = (pts: [number, number][]) =>
    pts
      .map(([x, y], i) => {
        if (i === 0) return `M${sx(x)},${sy(y)}`;
        if (stepped) return `H${sx(x)}V${sy(y)}`;
        return `L${sx(x)},${sy(y)}`;
      })
      .join("");

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const r = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    const xv = (x - PAD.l) / iw;
    let best: { s: number; i: number; d: number } | null = null;
    series.forEach((s, si) =>
      s.points.forEach(([px], pi) => {
        const d = Math.abs(px - xv);
        if (!best || d < best.d) best = { s: si, i: pi, d };
      }),
    );
    setHover(best ? { s: (best as { s: number }).s, i: (best as { i: number }).i } : null);
  };

  const hp = hover ? series[hover.s].points[hover.i] : null;

  return (
    <figure className="m-0">
      <figcaption id={titleId} className="mb-1 text-sm font-medium text-ink">
        {title}
      </figcaption>
      <div className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-labelledby={titleId}>
          <rect x={0} y={0} width={W} height={H} fill={SURFACE} />
          {ticks.map((t) => (
            <g key={t}>
              <line x1={sx(0)} x2={sx(1)} y1={sy(t)} y2={sy(t)} stroke={GRID} strokeWidth={1} />
              <line x1={sx(t)} x2={sx(t)} y1={sy(0)} y2={sy(1)} stroke={GRID} strokeWidth={1} />
              <text x={PAD.l - 6} y={sy(t) + 4} fontSize={10} textAnchor="end" fill={MUTED} className="tabular">
                {t}
              </text>
              <text x={sx(t)} y={H - PAD.b + 14} fontSize={10} textAnchor="middle" fill={MUTED} className="tabular">
                {t}
              </text>
            </g>
          ))}
          <line x1={sx(0)} x2={sx(1)} y1={sy(0)} y2={sy(0)} stroke={AXIS} />
          <line x1={sx(0)} x2={sx(0)} y1={sy(0)} y2={sy(1)} stroke={AXIS} />
          {reference?.kind === "diagonal" && (
            <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke={MUTED} strokeWidth={1} strokeDasharray="4 4" />
          )}
          {reference?.kind === "horizontal" && (
            <>
              <line x1={sx(0)} x2={sx(1)} y1={sy(reference.y)} y2={sy(reference.y)} stroke={MUTED} strokeWidth={1} strokeDasharray="4 4" />
              <text x={sx(1) - 4} y={sy(reference.y) - 4} fontSize={10} textAnchor="end" fill={INK2}>
                {reference.label}
              </text>
            </>
          )}
          {series.map((s, i) => (
            <path key={s.id} d={path(s.points)} fill="none" stroke={s.color ?? (i === 0 ? SERIES : SERIES_2)} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {marker && (
            <g>
              <circle cx={sx(marker.x)} cy={sy(marker.y)} r={6} fill={SURFACE} />
              <circle cx={sx(marker.x)} cy={sy(marker.y)} r={4.5} fill={SERIES} />
              <text x={sx(marker.x) + 9} y={sy(marker.y) + 4} fontSize={10} fill={INK2}>
                {marker.label}
              </text>
            </g>
          )}
          {hp && (
            <g pointerEvents="none">
              <line x1={sx(hp[0])} x2={sx(hp[0])} y1={sy(0)} y2={sy(1)} stroke={AXIS} />
              <circle cx={sx(hp[0])} cy={sy(hp[1])} r={6} fill={SURFACE} />
              <circle cx={sx(hp[0])} cy={sy(hp[1])} r={4} fill={series[hover!.s].color ?? (hover!.s === 0 ? SERIES : SERIES_2)} />
            </g>
          )}
          <text x={PAD.l + iw / 2} y={H - 4} fontSize={11} textAnchor="middle" fill={INK2}>
            {xLabel}
          </text>
          <text x={12} y={PAD.t + ih / 2} fontSize={11} textAnchor="middle" fill={INK2} transform={`rotate(-90 12 ${PAD.t + ih / 2})`}>
            {yLabel}
          </text>
          <rect x={PAD.l} y={PAD.t} width={iw} height={ih} fill="transparent" onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
        </svg>
        {hp && (
          <div className="pointer-events-none absolute right-2 top-2 rounded-md border border-line bg-raised px-2 py-1 text-xs text-ink-2 tabular">
            {series.length > 1 && <div className="text-ink">{series[hover!.s].label}</div>}
            {xLabel}: {hp[0].toFixed(3)} · {yLabel}: {hp[1].toFixed(3)}
          </div>
        )}
      </div>
      {series.length > 1 && (
        <div className="mt-1 flex flex-wrap gap-3 text-xs text-ink-2">
          {series.map((s, i) => (
            <span key={s.id} className="inline-flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-4 rounded" style={{ background: s.color ?? (i === 0 ? SERIES : SERIES_2) }} />
              {s.label}
            </span>
          ))}
        </div>
      )}
      <DataTable
        columns={[...(series.length > 1 ? ["series"] : []), xLabel, yLabel]}
        rows={series.flatMap((s) => sample(s.points, 12).map(([x, y]) => [...(series.length > 1 ? [s.label] : []), x.toFixed(3), y.toFixed(3)]))}
      />
    </figure>
  );
}

function sample<T>(arr: T[], n: number): T[] {
  if (arr.length <= n) return arr;
  return Array.from({ length: n }, (_, i) => arr[Math.round((i * (arr.length - 1)) / (n - 1))]);
}

export function DataTable({ columns, rows }: { columns: string[]; rows: (string | number)[][] }) {
  return (
    <details className="mt-1 text-xs text-ink-2">
      <summary className="cursor-pointer text-muted hover:text-ink-2">Data table</summary>
      <table className="mt-1 w-full border-collapse tabular">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c} className="border-b border-line px-2 py-1 text-left font-medium text-muted">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {r.map((c, j) => (
                <td key={j} className="border-b border-line/60 px-2 py-0.5">
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

export interface DotRow {
  id: string;
  label: string;
  point: number | null;
  ci: [number, number] | null;
  muted?: boolean;
  note?: string;
}

/** Point estimates with 95% intervals on a shared 0..1 axis (one measure per chart). */
export function DotCIChart({ title, rows, reference, domain = [0, 1] }: { title: string; rows: DotRow[]; reference?: { x: number; label: string }; domain?: [number, number] }) {
  const rowH = 30;
  const labelW = 210;
  const H = rows.length * rowH + 34;
  const Wd = 560;
  const iw = Wd - labelW - 20;
  const sx = (x: number) => labelW + ((x - domain[0]) / (domain[1] - domain[0])) * iw;
  const [hover, setHover] = useState<string | null>(null);
  const titleId = useId();
  const ticks = useMemo(() => {
    const out: number[] = [];
    for (let i = 0; i <= 4; i++) out.push(domain[0] + ((domain[1] - domain[0]) * i) / 4);
    return out;
  }, [domain]);
  const hv = rows.find((r) => r.id === hover);
  return (
    <figure className="m-0">
      <figcaption id={titleId} className="mb-1 text-sm font-medium text-ink">
        {title}
      </figcaption>
      <div className="relative">
        <svg viewBox={`0 0 ${Wd} ${H}`} className="w-full" role="img" aria-labelledby={titleId}>
          <rect width={Wd} height={H} fill={SURFACE} />
          {ticks.map((t) => (
            <g key={t}>
              <line x1={sx(t)} x2={sx(t)} y1={6} y2={H - 24} stroke={GRID} />
              <text x={sx(t)} y={H - 10} fontSize={10} textAnchor="middle" fill={MUTED} className="tabular">
                {t.toFixed(2)}
              </text>
            </g>
          ))}
          {reference && (
            <>
              <line x1={sx(reference.x)} x2={sx(reference.x)} y1={6} y2={H - 24} stroke={MUTED} strokeDasharray="4 4" />
              <text x={sx(reference.x) + 4} y={14} fontSize={10} fill={INK2}>
                {reference.label}
              </text>
            </>
          )}
          {rows.map((r, i) => {
            const y = 20 + i * rowH;
            const color = r.muted ? MUTED : SERIES;
            return (
              <g key={r.id} onMouseEnter={() => setHover(r.id)} onMouseLeave={() => setHover(null)}>
                <rect x={0} y={y - rowH / 2} width={Wd} height={rowH} fill={hover === r.id ? "#17222c" : "transparent"} />
                <text x={labelW - 10} y={y + 4} fontSize={11} textAnchor="end" fill={r.muted ? MUTED : INK2}>
                  {r.label}
                </text>
                {r.ci && <line x1={sx(r.ci[0])} x2={sx(r.ci[1])} y1={y} y2={y} stroke={color} strokeWidth={2} strokeLinecap="round" />}
                {r.point != null && (
                  <>
                    <circle cx={sx(r.point)} cy={y} r={6.5} fill={SURFACE} />
                    <circle cx={sx(r.point)} cy={y} r={4.5} fill={color} />
                  </>
                )}
              </g>
            );
          })}
        </svg>
        {hv && (
          <div className="pointer-events-none absolute right-2 top-2 rounded-md border border-line bg-raised px-2 py-1 text-xs text-ink-2 tabular">
            <div className="text-ink">{hv.label}</div>
            {hv.point?.toFixed(3) ?? "n/a"}
            {hv.ci && ` (95% CI ${hv.ci[0].toFixed(3)}–${hv.ci[1].toFixed(3)})`}
            {hv.note && <div className="text-muted">{hv.note}</div>}
          </div>
        )}
      </div>
      <DataTable
        columns={["model", "estimate", "95% CI"]}
        rows={rows.map((r) => [r.label, r.point?.toFixed(3) ?? "n/a", r.ci ? `${r.ci[0].toFixed(3)}–${r.ci[1].toFixed(3)}` : "n/a"])}
      />
    </figure>
  );
}

/** Confusion matrix as an accessible table; colour = row-normalised share (one-hue sequential). */
export function ConfusionMatrix({ labels, matrix, display }: { labels: string[]; matrix: number[][]; display: (l: string) => string }) {
  const [hover, setHover] = useState<[number, number] | null>(null);
  return (
    <figure className="m-0">
      <figcaption className="mb-1 text-sm font-medium text-ink">Confusion matrix (rows: reference category · columns: highest-scoring category)</figcaption>
      <div className="overflow-x-auto">
        <table className="border-collapse text-xs tabular" aria-label="Confusion matrix counts">
          <thead>
            <tr>
              <th className="p-1" />
              {labels.map((l, j) => (
                <th key={l} scope="col" className={`max-w-[64px] p-1 align-bottom font-normal ${hover?.[1] === j ? "text-ink" : "text-muted"}`} title={display(l)}>
                  <span className="block truncate">{display(l).split(" (")[0]}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => {
              const tot = row.reduce((a, b) => a + b, 0) || 1;
              return (
                <tr key={labels[i]}>
                  <th scope="row" className={`whitespace-nowrap p-1 pr-2 text-left font-normal ${hover?.[0] === i ? "text-ink" : "text-muted"}`}>
                    {display(labels[i]).split(" (")[0]} <span className="text-muted">({tot})</span>
                  </th>
                  {row.map((v, j) => {
                    const share = v / tot;
                    // one-hue ramp: surface -> teal, by row share
                    const alpha = 0.08 + share * 0.92;
                    const dark = share > 0.55;
                    return (
                      <td
                        key={j}
                        onMouseEnter={() => setHover([i, j])}
                        onMouseLeave={() => setHover(null)}
                        title={`${display(labels[i])} → ${display(labels[j])}: ${v} (${(share * 100).toFixed(0)}% of row)`}
                        className={`h-9 w-14 border-2 border-surface text-center ${i === j ? "font-semibold" : ""}`}
                        style={{ background: `rgba(20,168,153,${alpha.toFixed(3)})`, color: dark ? "#06110f" : "#e8eef4", borderRadius: 4 }}
                      >
                        {v}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-1 text-xs text-muted">Cell shade encodes the share of the reference row; numbers are image counts.</p>
    </figure>
  );
}
