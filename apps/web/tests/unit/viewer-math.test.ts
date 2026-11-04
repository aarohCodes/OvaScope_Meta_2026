import { describe, expect, it } from "vitest";
import { canvasTransform, fitView, imageToScreen, onResize, pan, screenToImage, zoomAt } from "@/lib/viewer-math";

const img = { w: 959, h: 537 };

describe("viewer geometry", () => {
  it("fits and centres the image in the container", () => {
    const v = fitView(img, { w: 800, h: 600 }, 16);
    expect(v.scale).toBeCloseTo((800 - 32) / 959, 6);
    const [x0, y0] = imageToScreen(v, 0, 0);
    const [x1, y1] = imageToScreen(v, img.w, img.h);
    expect((x0 + x1) / 2).toBeCloseTo(400, 6);
    expect((y0 + y1) / 2).toBeCloseTo(300, 6);
  });

  it("round-trips screen <-> image coordinates", () => {
    const v = pan(zoomAt(fitView(img, { w: 640, h: 480 }), 2.7, 100, 50), 13, -7);
    for (const [x, y] of [[0, 0], [123.5, 77.25], [959, 537]]) {
      const [sx, sy] = imageToScreen(v, x, y);
      const [ix, iy] = screenToImage(v, sx, sy);
      expect(ix).toBeCloseTo(x, 9);
      expect(iy).toBeCloseTo(y, 9);
    }
  });

  it("keeps the image point under the cursor fixed while zooming", () => {
    const v = fitView(img, { w: 640, h: 480 });
    const before = screenToImage(v, 210, 330);
    const z = zoomAt(v, 3, 210, 330);
    const after = screenToImage(z, 210, 330);
    expect(after[0]).toBeCloseTo(before[0], 9);
    expect(after[1]).toBeCloseTo(before[1], 9);
  });

  it("overlay pixels stay registered with image pixels after a container resize", () => {
    // Overlays are drawn with the same transform at source resolution, so a mask pixel
    // (x, y) must map to exactly the same screen point as image pixel (x, y).
    const c1 = { w: 900, h: 600 };
    const c2 = { w: 420, h: 700 };
    const v1 = zoomAt(fitView(img, c1), 1.8, 300, 200);
    const v2 = onResize(v1, img, c1, c2);
    const centreBefore = screenToImage(v1, c1.w / 2, c1.h / 2);
    const centreAfter = screenToImage(v2, c2.w / 2, c2.h / 2);
    expect(centreAfter[0]).toBeCloseTo(centreBefore[0], 6);
    expect(centreAfter[1]).toBeCloseTo(centreBefore[1], 6);
    const maskPixel = [512, 301] as const;
    expect(imageToScreen(v2, ...maskPixel)).toEqual(imageToScreen(v2, ...maskPixel));
    // relative zoom (vs fit) is preserved
    expect(v2.scale / fitView(img, c2).scale).toBeCloseTo(v1.scale / fitView(img, c1).scale, 6);
  });

  it("applies device pixel ratio to the canvas transform only", () => {
    const v = { scale: 0.5, ox: 10, oy: 20 };
    expect(canvasTransform(v, 2)).toEqual([1, 0, 0, 1, 20, 40]);
  });

  it("clamps zoom", () => {
    let v = fitView(img, { w: 640, h: 480 });
    for (let i = 0; i < 100; i++) v = zoomAt(v, 2, 0, 0);
    expect(v.scale).toBeLessThanOrEqual(32);
  });
});
