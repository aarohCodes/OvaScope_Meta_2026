// Pure geometry for the canvas image viewer. Overlays are source-resolution
// images, so image and overlays share ONE transform: screen = image * scale + offset.

export interface View {
  scale: number;
  ox: number; // CSS px
  oy: number;
}

export interface Size {
  w: number;
  h: number;
}

export const MIN_SCALE = 0.05;
export const MAX_SCALE = 32;

export function fitView(image: Size, container: Size, padding = 16): View {
  const availW = Math.max(1, container.w - 2 * padding);
  const availH = Math.max(1, container.h - 2 * padding);
  const scale = clamp(Math.min(availW / image.w, availH / image.h), MIN_SCALE, MAX_SCALE);
  return {
    scale,
    ox: (container.w - image.w * scale) / 2,
    oy: (container.h - image.h * scale) / 2,
  };
}

export function imageToScreen(v: View, x: number, y: number): [number, number] {
  return [x * v.scale + v.ox, y * v.scale + v.oy];
}

export function screenToImage(v: View, sx: number, sy: number): [number, number] {
  return [(sx - v.ox) / v.scale, (sy - v.oy) / v.scale];
}

/** Zoom by `factor` keeping the image point under (sx, sy) fixed on screen. */
export function zoomAt(v: View, factor: number, sx: number, sy: number): View {
  const scale = clamp(v.scale * factor, MIN_SCALE, MAX_SCALE);
  const k = scale / v.scale;
  return { scale, ox: sx - (sx - v.ox) * k, oy: sy - (sy - v.oy) * k };
}

export function pan(v: View, dx: number, dy: number): View {
  return { ...v, ox: v.ox + dx, oy: v.oy + dy };
}

/**
 * Re-fit after the container is resized while preserving the user's relative zoom:
 * the image point at the old container centre stays at the new centre.
 */
export function onResize(v: View, image: Size, oldC: Size, newC: Size): View {
  const fitOld = fitView(image, oldC).scale;
  const fitNew = fitView(image, newC).scale;
  const scale = clamp(v.scale * (fitNew / fitOld), MIN_SCALE, MAX_SCALE);
  const [cx, cy] = screenToImage(v, oldC.w / 2, oldC.h / 2);
  return { scale, ox: newC.w / 2 - cx * scale, oy: newC.h / 2 - cy * scale };
}

/** Canvas transform for a device-pixel-ratio-scaled backing store. */
export function canvasTransform(v: View, dpr: number): [number, number, number, number, number, number] {
  return [v.scale * dpr, 0, 0, v.scale * dpr, v.ox * dpr, v.oy * dpr];
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

export function insideImage(image: Size, x: number, y: number): boolean {
  return x >= 0 && y >= 0 && x <= image.w && y <= image.h;
}
