# Security and privacy notes

Research prototype. Not for diagnosis, screening, or treatment decisions. **No HIPAA
compliance is claimed.** Use de-identified research images only.

## Deployment posture

- Default `OVASCOPE_DEPLOYMENT=local`: no authentication; all Compose ports bind to
  `127.0.0.1`. **Do not expose this deployment publicly.**
- `OVASCOPE_DEPLOYMENT=public` refuses to start unless `OVASCOPE_API_TOKEN` (≥ 24 chars)
  is set and CORS has no wildcard. The token is attached server-side by the Next.js proxy
  (`app/backend/[...path]/route.ts`) and never reaches the browser. A public deployment
  additionally needs real user accounts, abuse limits beyond the built-in per-IP upload
  limit, TLS, and a documented deletion process. None of this has been reviewed.

## Controls implemented

| Area | Control | Where |
|---|---|---|
| Upload size | Streaming body limit before multipart parsing (413), plus post-read check | `BodySizeLimit`, `create_study` |
| File type | Magic-byte sniffing (PNG/JPEG only); declared format must match; client MIME and file name ignored | `ovascope_ml/data/imaging.py` |
| Decompression bombs | Pixel ceiling before decode; Pillow bomb warnings treated as errors | `decode_image` |
| Corrupt / multi-frame | Full decode required; multi-frame rejected | `decode_image` |
| Path traversal | Opaque random artifact keys validated by regex and resolved under the store root; IDs validated; proxy forwards only `/health` and `/api/v1/*` with safe segments | `artifact_store.py`, `main.py`, proxy |
| Storage | Non-public local store; images and overlays served only through authenticated API routes with `Cache-Control: no-store` | |
| Retention | Studies expire after `OVASCOPE_RETENTION_HOURS` (default 24); purge runs at start-up and every minute in the worker | `jobs.purge_expired` |
| Deletion | `DELETE /api/v1/studies/{id}` cancels jobs and removes rows (DB cascade) and every stored object | `delete_study` |
| CORS | Allow-list of origins; no credentials | `CORSMiddleware` |
| Headers | `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, restrictive CSP on API and web | middleware, `next.config.ts` |
| Secrets | Server-side env only; `.env` ignored by git; `.env.example` holds placeholders | |
| Logging | JSON logs with an allow-list of fields; no bodies, file names, image bytes, tokens; exception type only | `logging_setup.py`; tested in `test_no_phi_in_logs` |
| Third parties | No analytics, fonts or CDNs; CSP `default-src 'self'` | `next.config.ts` |
| External transmission | Disabled and fail-closed; no hosted provider configured; Modal adapter refuses without explicit authorisation | `capabilities`, `infra/modal` |
| Model files | Bundles are NumPy/JSON (no pickle), loaded with `allow_pickle=False`, SHA-256 verified per file | `LoadedBundle` |
| Rate / queue limits | Per-IP uploads/minute; bounded queue; one job at a time; bounded retries | settings |

## Known gaps

- File names are not stored, but **burned-in text in pixels is not detected or removed**.
  The upload page warns about this explicitly.
- No malware scanning of uploads (images are decoded by Pillow only).
- The in-memory rate limiter is per process and resets on restart.
- SQLite (dev) is unencrypted at rest; the Compose PostgreSQL volume is too.
- Dependencies pinned in mid-2025 versions; run a vulnerability scan (e.g. `pip-audit`,
  `npm audit`) before any shared deployment. ESLint 9.39 is flagged by npm as past its
  support window (dev-only tool).
