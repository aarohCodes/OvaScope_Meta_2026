"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  canvasTransform,
  fitView,
  imageToScreen,
  insideImage,
  onResize,
  pan,
  screenToImage,
  zoomAt,
  type Size,
  type View,
} from "@/lib/viewer-math";

export interface OverlayLayer {
  id: string;
  label: string;
  src: string;
  visible: boolean;
  opacity: number; // 0..1
}

interface Props {
  imageSrc: string;
  natural: Size;
  overlays: OverlayLayer[];
  promptBox?: [number, number, number, number] | null;
  showPromptBox?: boolean;
  polygons?: number[][][];
  draftPolygon?: number[][];
  annotating?: boolean;
  onImageClick?: (x: number, y: number) => void;
}

function useLoadedImage(src: string | null): HTMLImageElement | null {
  // State is only set from load callbacks; an image loaded for an older src is ignored.
  const [loaded, setLoaded] = useState<{ src: string; el: HTMLImageElement | null } | null>(null);
  useEffect(() => {
    if (!src) return;
    let alive = true;
    const el = new Image();
    el.decoding = "async";
    el.onload = () => alive && setLoaded({ src, el });
    el.onerror = () => alive && setLoaded({ src, el: null });
    el.src = src;
    return () => {
      alive = false;
    };
  }, [src]);
  return src && loaded?.src === src ? loaded.el : null;
}

function OverlayImage({ src, onLoad }: { src: string; onLoad: (src: string, el: HTMLImageElement | null) => void }) {
  const el = useLoadedImage(src);
  useEffect(() => onLoad(src, el), [src, el, onLoad]);
  return null;
}

export function ImageViewer({
  imageSrc,
  natural,
  overlays,
  promptBox,
  showPromptBox = true,
  polygons = [],
  draftPolygon = [],
  annotating = false,
  onImageClick,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const base = useLoadedImage(imageSrc);
  const [overlayEls, setOverlayEls] = useState<Record<string, HTMLImageElement | null>>({});
  const [container, setContainer] = useState<Size>({ w: 800, h: 560 });
  const [view, setView] = useState<View | null>(null);
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const containerRef = useRef(container);

  const registerOverlay = useCallback((src: string, el: HTMLImageElement | null) => {
    setOverlayEls((prev) => (prev[src] === el ? prev : { ...prev, [src]: el }));
  }, []);

  // Track container size; preserve the visible region on resize.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const next = { w: Math.round(entry.contentRect.width), h: Math.round(entry.contentRect.height) };
      const prev = containerRef.current;
      if (next.w === prev.w && next.h === prev.h) return;
      containerRef.current = next;
      setContainer(next);
      setView((v) => (v ? onResize(v, natural, prev, next) : v));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [natural]);

  // Fit when the image changes.
  useEffect(() => {
    setView(fitView(natural, containerRef.current));
  }, [natural.w, natural.h, imageSrc]); // eslint-disable-line react-hooks/exhaustive-deps

  // Draw.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !view) return;
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    canvas.width = Math.round(container.w * dpr);
    canvas.height = Math.round(container.h * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#06090d";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (!base) return;
    ctx.setTransform(...canvasTransform(view, dpr));
    ctx.imageSmoothingEnabled = view.scale < 2;
    ctx.drawImage(base, 0, 0, natural.w, natural.h);
    for (const o of overlays) {
      const el = overlayEls[o.src];
      if (!o.visible || !el) continue;
      ctx.globalAlpha = o.opacity;
      ctx.drawImage(el, 0, 0, natural.w, natural.h); // same transform: exact registration
    }
    ctx.globalAlpha = 1;
    const px = 1 / view.scale;
    if (promptBox && showPromptBox) {
      const [x1, y1, x2, y2] = promptBox;
      ctx.setLineDash([6 * px, 4 * px]);
      ctx.lineWidth = 1.5 * px;
      ctx.strokeStyle = "#5eead4";
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
    }
    const drawPoly = (pts: number[][], closed: boolean) => {
      if (pts.length === 0) return;
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (const p of pts.slice(1)) ctx.lineTo(p[0], p[1]);
      if (closed) ctx.closePath();
      ctx.lineWidth = 2 * px;
      ctx.strokeStyle = "#a78bfa";
      ctx.stroke();
      for (const p of pts) {
        ctx.beginPath();
        ctx.arc(p[0], p[1], 3.5 * px, 0, Math.PI * 2);
        ctx.fillStyle = "#a78bfa";
        ctx.fill();
      }
    };
    polygons.forEach((p) => drawPoly(p, true));
    drawPoly(draftPolygon, false);
  }, [view, container, base, overlays, overlayEls, natural, promptBox, showPromptBox, polygons, draftPolygon]);

  const local = (e: { clientX: number; clientY: number }): [number, number] => {
    const r = canvasRef.current!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };

  const onWheel = (e: React.WheelEvent) => {
    if (!view) return;
    const [sx, sy] = local(e);
    setView(zoomAt(view, e.deltaY < 0 ? 1.15 : 1 / 1.15, sx, sy));
  };

  // React's onWheel is passive; prevent page scroll with a native listener.
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const stop = (ev: WheelEvent) => ev.preventDefault();
    c.addEventListener("wheel", stop, { passive: false });
    return () => c.removeEventListener("wheel", stop);
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, moved: false };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!drag.current || !view) return;
    const dx = e.clientX - drag.current.x;
    const dy = e.clientY - drag.current.y;
    if (Math.abs(dx) + Math.abs(dy) > 2) drag.current.moved = true;
    if (drag.current.moved) {
      drag.current.x = e.clientX;
      drag.current.y = e.clientY;
      setView(pan(view, dx, dy));
    }
  };
  const onPointerUp = (e: React.PointerEvent) => {
    const wasDrag = drag.current?.moved;
    drag.current = null;
    if (!wasDrag && annotating && onImageClick && view) {
      const [sx, sy] = local(e);
      const [x, y] = screenToImage(view, sx, sy);
      if (insideImage(natural, x, y)) onImageClick(Math.round(x * 10) / 10, Math.round(y * 10) / 10);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (!view) return;
    const c = { x: container.w / 2, y: container.h / 2 };
    const step = 40;
    const keys: Record<string, () => View> = {
      "+": () => zoomAt(view, 1.25, c.x, c.y),
      "=": () => zoomAt(view, 1.25, c.x, c.y),
      "-": () => zoomAt(view, 0.8, c.x, c.y),
      "0": () => fitView(natural, container),
      ArrowLeft: () => pan(view, step, 0),
      ArrowRight: () => pan(view, -step, 0),
      ArrowUp: () => pan(view, 0, step),
      ArrowDown: () => pan(view, 0, -step),
    };
    const fn = keys[e.key];
    if (fn) {
      e.preventDefault();
      setView(fn());
    }
  };

  const zoomBtn = (f: number) => view && setView(zoomAt(view, f, container.w / 2, container.h / 2));
  const tl = view ? imageToScreen(view, 0, 0) : [0, 0];

  return (
    <div className="flex h-full flex-col">
      {overlays.map((o) => (
        <OverlayImage key={o.src} src={o.src} onLoad={registerOverlay} />
      ))}
      <div className="flex items-center gap-2 border-b border-line px-3 py-2 text-sm no-print">
        <button type="button" className="rounded border border-line-strong px-2 py-0.5 hover:bg-raised" onClick={() => zoomBtn(1.25)} aria-label="Zoom in">
          +
        </button>
        <button type="button" className="rounded border border-line-strong px-2 py-0.5 hover:bg-raised" onClick={() => zoomBtn(0.8)} aria-label="Zoom out">
          −
        </button>
        <button type="button" className="rounded border border-line-strong px-2 py-0.5 hover:bg-raised" onClick={() => setView(fitView(natural, container))} data-testid="viewer-reset">
          Reset view
        </button>
        <span className="tabular ml-2 text-xs text-muted" data-testid="viewer-zoom">
          {view ? `${Math.round(view.scale * 100)}%` : ""}
        </span>
        <span className="ml-auto hidden text-xs text-muted lg:inline">
          Scroll or +/− to zoom · drag or arrows to pan · 0 to reset{annotating ? " · click to add points" : ""}
        </span>
      </div>
      <div ref={wrapRef} className="relative min-h-[360px] flex-1 overflow-hidden">
        <canvas
          ref={canvasRef}
          className="absolute left-0 top-0"
          role="img"
          tabIndex={0}
          aria-label={`Uploaded ultrasound image, ${natural.w} by ${natural.h} pixels, with ${overlays.filter((o) => o.visible).length} overlay(s) visible. Use plus and minus to zoom, arrow keys to pan, 0 to reset.`}
          data-testid="viewer-canvas"
          data-view={view ? `${view.scale.toFixed(4)},${tl[0].toFixed(1)},${tl[1].toFixed(1)}` : ""}
          data-overlays={overlays.map((o) => `${o.id}:${o.visible ? 1 : 0}:${o.opacity.toFixed(2)}`).join("|")}
          style={{ width: container.w, height: container.h, cursor: annotating ? "crosshair" : "grab", touchAction: "none" }}
          onWheel={onWheel}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onKeyDown={onKeyDown}
        />
        {!base && <div className="absolute inset-0 grid place-items-center text-sm text-muted">Loading image…</div>}
      </div>
    </div>
  );
}
