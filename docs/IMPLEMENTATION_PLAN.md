# OvaScope implementation plan

Research prototype. Not for diagnosis, screening, or treatment decisions.

This plan is written against what was actually found on 2026-11-04 (see
`docs/DECISIONS.md` for the evidence behind each choice).

## Environment constraints that shape the plan

- Laptop CPU only (Intel Core 7 150U, 10 cores, 16 GB RAM, Intel integrated GPU, no CUDA).
  All training and evaluation must be feasible on CPU, so the ML design uses
  **frozen Meta backbones + small linear heads**, not fine-tuning.
- Docker Desktop 28 is available; `make` is not installed on this Windows host,
  so a Python task runner (`tasks.py`) is the source of truth and the `Makefile`
  delegates to it.
- No Hugging Face token: gated Meta models (for example SAM 3) are not accessible.
  DINOv2 ViT-S/14 and ViT-B/14 weights are already in the local cache; SAM
  (`facebook/sam-vit-base`) and SAM 2.1 are public on the Hub.
- No Modal, Meta Llama API, or other hosted-provider credentials.

## Target system

```
apps/web                 Next.js (App Router, TS, Tailwind) research UI
services/api             FastAPI + SQLAlchemy + Alembic (SQLite dev / Postgres compose)
services/worker          DB-polling job worker (no Redis; compare-and-set claims)
ml/ovascope_ml           data manifest, audit, splits, backbones, heads, providers,
                         inference pipeline, training, evaluation
configs/                 experiment configs (YAML)
tests/                   pytest (unit + API + integration); web tests live in apps/web
infra/                   Dockerfiles, Modal adapter (disabled without credentials)
docs/                    plan, decisions, dataset readiness, cards, claims ledger, demo script
```

## Phases

1. **Inspection, decisions, dataset audit, schemas, provider interfaces.**
   Manifest schema, MMOTU adapter, validation (missing/invalid/duplicates/near-duplicates/
   mask dims/conflicting labels/missing groups), shortcut audit (burned-in markers,
   Doppler colour, image-size clusters), dataset readiness report.
2. **Vertical slice in honest NO_MODEL mode.** Upload → validation → job → result with
   `model_unavailable` and the exact missing requirement; image viewer; deletion.
3. **Baselines.** Majority class; hand-crafted intensity/texture features + logistic
   regression; ImageNet ResNet-50 (non-Meta comparator). Official MMOTU split, test set
   touched once; model selection, calibration and thresholds from grouped CV on the
   development split only.
4. **Meta components and ablations.** DINOv2 frozen features; automatic lesion
   localisation = DINOv2 patch-token localiser (trained on development masks) → box prompt
   → SAM mask refinement; ROI-assisted classifier; oracle-mask upper bound kept separate;
   ablations: DINOv2 vs ResNet-50, ROI vs whole image, SAM refinement vs localiser alone.
5. **Product surfaces.** Analysis workspace (canvas viewer, overlays, provenance),
   evaluation page from real artifacts, model/data cards, settings, report export.
6. **Hardening.** Security review, tests (pytest, vitest, Playwright), Docker Compose,
   CI, demo script from held-out cases, claims ledger.

## Primary and secondary targets (pre-specified before any test evaluation)

- Primary: `mmotu8` — the eight MMOTU dataset categories, image level.
- Secondary: `hgs_vs_other` — the MMOTU "High grade serous" category vs all other MMOTU
  categories, derived from the primary model's score for that category with a threshold
  chosen on development out-of-fold predictions. Not a general malignancy detector.
- Conditional early-stage extension: **blocked** — MMOTU has no stage or borderline labels.

## Definition of done

Engineering completion (acceptance criteria 1–12 in the brief) and scientific validation
are separate milestones. The second is not achievable with MMOTU alone: no patient
identifiers, single centre, no external validation.
