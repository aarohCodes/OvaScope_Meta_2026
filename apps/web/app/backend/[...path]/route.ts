// Server-side proxy: browser -> Next.js -> OvaScope API.
// Credentials are attached here from server-only env; the browser never sees them.
// Only /health and /api/v1/* are forwarded, and path segments are validated.
import type { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

const API = process.env.OVASCOPE_API_INTERNAL_URL ?? "http://127.0.0.1:8000";
const TOKEN = process.env.OVASCOPE_API_TOKEN;
const SEGMENT = /^[A-Za-z0-9._-]+$/;
const FORWARD_REQ = ["content-type", "idempotency-key", "x-request-id"];
const FORWARD_RES = ["content-type", "content-disposition", "x-request-id", "cache-control"];

async function proxy(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const ok =
    path.length > 0 &&
    path.every((s) => SEGMENT.test(s) && s !== ".." && s !== ".") &&
    (path.join("/") === "health" || (path[0] === "api" && path[1] === "v1"));
  if (!ok) {
    return Response.json({ code: "NOT_FOUND", message: "Unknown backend route." }, { status: 404 });
  }
  const url = `${API}/${path.join("/")}${req.nextUrl.search}`;
  const headers = new Headers();
  for (const h of FORWARD_REQ) {
    const v = req.headers.get(h);
    if (v) headers.set(h, v);
  }
  if (TOKEN) headers.set("authorization", `Bearer ${TOKEN}`);
  const init: RequestInit & { duplex?: "half" } = { method: req.method, headers, cache: "no-store", redirect: "manual" };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = req.body;
    init.duplex = "half";
  }
  let upstream: Response;
  try {
    upstream = await fetch(url, init);
  } catch {
    return Response.json(
      { code: "API_UNREACHABLE", message: "The OvaScope API is not reachable from the web server." },
      { status: 502 },
    );
  }
  const out = new Headers();
  for (const h of FORWARD_RES) {
    const v = upstream.headers.get(h);
    if (v) out.set(h, v);
  }
  out.set("cache-control", "no-store");
  return new Response(upstream.body, { status: upstream.status, headers: out });
}

export const GET = proxy;
export const POST = proxy;
export const DELETE = proxy;
