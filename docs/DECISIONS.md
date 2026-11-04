# Decision log

Newest entries at the bottom of each section. Each entry: decision, evidence, consequence.

## D-001 Repository location (2026-11-04)

- The VS Code working directory `C:\Users\aaroh` is the user's home directory and is itself a
  git checkout of an unrelated project (`aarohCodes/CodeRed`, with many pending deletions).
- **Decision:** create OvaScope as a new, independent repository at `C:\Users\aaroh\OvaScope`.
  Nothing in the home repository was modified or deleted.

## D-002 Hardware and toolchain (2026-11-04)

- CPU: Intel Core 7 150U (10 cores / 12 threads), 15.7 GB RAM, Intel integrated graphics,
  `nvidia-smi` absent, so no CUDA. Disk: ~69 GB free.
- Python 3.11.9, Node 22.14, npm 10.9, Docker 28.4 / Compose 2.39 (Docker Desktop running),
  git 2.47. `make` not installed.
- **Decision:** CPU-only ML design (frozen backbones + linear heads). `tasks.py` is the
  command interface; `Makefile` delegates to it for Linux/macOS/CI.
- **Consequence:** no fine-tuning of DINOv2 or SAM in this environment. GPU guidance in the
  README is limited to what was actually run.

## D-003 Dependency versions (2026-11-04)

- Newest releases on the registries were torch 2.15, transformers 5.22, Next 16.4,
  TypeScript 7.0, ESLint 10.
- **Decision:** pin well-established releases whose APIs could be verified exactly rather
  than the newest majors: torch 2.8.0 (CPU wheels), torchvision 0.23.0, timm 1.0.19,
  transformers 4.56.2, scikit-learn 1.7.2, FastAPI 0.116.1, SQLAlchemy 2.0.43,
  Alembic 1.16.5, Pydantic 2.11.7. Web versions are recorded in `apps/web/package.json`
  and the committed lockfile.
- **Consequence:** an upgrade pass should be done deliberately, with the test suite.

## D-004 Dataset source and governance (2026-11-04)

- No user-provided ovarian dataset exists on this machine (home tree and Downloads searched).
- MMOTU (Zhao et al., arXiv 2207.06799, Pattern Recognition): official release is a
  Google Drive folder linked from `github.com/cv516Buaa/MMOTU_DS2Net`.
  - The GitHub repository is tagged Apache-2.0. **No data licence, data-use agreement, or
    ethics statement accompanies the images.** The Apache-2.0 tag on the code repository is
    not treated as a licence for the Drive-hosted images.
  - Bulk download from Drive was rate-limited after 87 files.
  - **Transport:** the public Hugging Face mirror `norayao/MMOTU` pinned to revision
    `d29379c60ac4072cfbf9c6f960505d43f98b3f0d`; per-file SHA-256 checked against the mirror's
    LFS records; files also compared against the 87 files fetched directly from the official
    Drive folder (`ovascope-ml verify-mirror`). The mirror's `apache-2.0` tag is set by its
    uploader and is likewise not treated as authoritative.
- **Decision:** use MMOTU OTU_2d for **local, non-redistributed research development only**.
  Images, masks and derived per-image data are never committed (`.gitignore` + CI guard).
  No images are sent to any hosted provider.
- **Open item for the user:** confirm that this use is acceptable for the intended event or
  publication, or provide an authorised dataset with patient identifiers.

## D-005 What MMOTU can and cannot support (2026-11-04)

Evidence: arXiv 2207.06799v3 §III, Table I, figures; official code
`mmseg/datasets/mmotu.py`, `configs/_base_/datasets/mmotu.py`.

- 1,639 images (OTU_2d 1,469 + CEUS 170) from 294 patients; **no patient identifiers released**.
  At ~5 images per patient, image-level splits very likely place images of one patient in
  both development and test.
- Labels: 8 categories annotated by 27 Ob/Gyn experts who "refer to pathological reports";
  each image annotated by one expert and checked by another.
- Images carry burned-in device text, calipers, measurements, cursor icons and text labels
  (for example "CY", "LOV"); 216 of the 1,469 OTU_2d images are colour Doppler (CDFI).
  Image dimensions fall into device-like clusters.
- No borderline category, no stage, no other invasive histotypes, single centre.
- **Decisions:**
  - Evaluation unit is the image. All metrics are labelled *exploratory, image-level;
    patient-independent evaluation not established*. Claims of patient-level
    generalisation are blocked.
  - Near-duplicate clusters (perceptual hash) are kept together inside development CV folds
    and reported across the official split; a sensitivity analysis excludes test images that
    have a near-duplicate in development. Deduplication is not treated as patient independence.
  - The official 1000/469 split is used as development/test because it was fixed by the
    dataset authors before this project (no researcher choice of split after seeing results).
  - Early-invasive-stage extension: **blocked** (no stage labels). No stage tab is shown.
  - "External validation not performed."

## D-006 Label policy (2026-11-04)

- Index order from the official code (commented `CLASSES` block) and the paper's category
  figure agree: 0 Chocolate cyst, 1 Serous cystadenoma, 2 Teratoma, 3 Theca cell tumor,
  4 Simple cyst, 5 Normal ovary, 6 Mucinous cystadenoma, 7 High grade serous.
- **Naming discrepancy:** the code calls index 7 `HighGradeSerousCystadenoma`; the paper figure
  labels it "High grade serous". We normalise it to `high_grade_serous` and display
  "High-grade serous (MMOTU category)". We do not relabel it as "cancer" in the UI.
- "Normal ovary" is a category and its mask outlines the ovary, not a lesion; it is excluded
  from lesion-segmentation metrics and reported separately.
- The mapping is cross-checked against the colour palette of the semantic masks where present.

## D-007 Meta components (2026-11-04)

- DINOv2 ViT-S/14 and ViT-B/14 (`timm/vit_*_patch14_dinov2.lvd142m`) are in the local
  Hugging Face cache (Apache-2.0).
- SAM family: `facebook/sam-vit-base` (Apache-2.0) via `transformers.SamModel`; SAM 2.1 tiny
  evaluated for CPU cost. SAM 3 is gated and requires a token, so it is **not used**.
- Generic SAM needs prompts. **Automatic localisation** = a linear patch-token localiser on
  frozen DINOv2 features, trained on development masks, whose largest component gives a box
  prompt for SAM. The localiser alone and SAM-refined masks are both evaluated (Dice/IoU).
  If localisation fails, the result says so; a box is never relabelled as a SAM mask.
- No Meta LLM is accessible. `OptionalMultimodalProvider` is implemented as an interface and
  reported as disabled; research reports use a deterministic template.

## D-008 Infrastructure scope (2026-11-04)

- Job queue: a `jobs` table claimed with compare-and-set updates by `services/worker`
  (works on SQLite and PostgreSQL). Redis is not used: one worker, low throughput, no need.
- Storage: local filesystem artifact store with opaque keys; no paths in API responses.
- Deployment label: **local-only**. Optional bearer-token auth exists, and the API refuses to
  start in `public` mode without it.

## D-009 Dataset transport and verification (2026-11-04)

- Drive bulk download stalled (rate limit) after 87 files; the Hugging Face mirror was fetched
  sequentially with HTTP 429 back-off (`scripts/fetch_mmotu_mirror.py`).
- Result: 4,407/4,407 LFS files match the mirror's SHA-256; **87/87 official Drive files are
  byte-identical** to the mirror copies (`ovascope-ml verify-mirror`).

## D-010 Primary/secondary targets, split and selection protocol (pre-specified 2026-11-04)

- Primary target `mmotu8`, primary metric balanced accuracy; secondary `hgs_vs_other` scored by the
  primary model's score for that category (threshold: Youden J on development OOF).
- Development: official train list, 5-fold StratifiedGroupKFold grouped by perceptual-hash
  near-duplicate cluster. Test: official val list, touched once by `ovascope-ml evaluate` after
  `selection.json` was written. The training code never reads test labels; oracle experiments that
  read test reference masks write to `eval_only/` and are reported as upper bounds.
- Development images receive out-of-fold localiser outputs so ROI features used for training match
  what the automatic system produces at test time.

## D-011 Calibration wording (2026-11-04)

- Heads use class-balanced weights; temperature scaling is argmax-preserving but cannot correct
  prior shift. Status is therefore `temperature_scaled_not_prevalence_calibrated` and the UI says
  "Model scores (temperature-scaled; not calibrated probabilities)". No probability or risk wording.

## D-012 Engineering corrections found by verification (2026-11-04)

- `transformers` SAM 2 docstring example passes `reshaped_input_sizes` positionally into
  `mask_threshold`; the adapter calls `post_process_masks(masks, original_sizes)` only.
- The SAM 2.1 checkpoint declares `model_type: sam2_video`; loading it into `Sam2Model` works for
  image prompts (verified Dice 0.99 on synthetic shapes) and the warning is silenced.
- Data audit palette check was O(pixels log pixels) per image (`np.unique(axis=0)`); replaced with
  packed-integer unique (0.59 s → 0.019 s per image).
- A 56-image smoke run of the full pipeline caught a missing baseline field before the 2-hour run.
- The UI wording guard initially flagged the disclaimer "Not clinically validated"; claim phrases are
  now allowed only when negated, diagnostic phrases are banned outright.

## D-013 Tooling (2026-11-04)

- ESLint 9.39.5 is reported by npm as past its support window; kept because eslint-config-next 16.4
  targets ESLint 9. Dev-only. TypeScript 5.9.3 (not 7.0) for tooling compatibility.
- Ruff owns formatting (line length 120); E501 is not linted separately.
- At the user's request, work is committed and pushed frequently to
  github.com/aarohCodes/OvaScope_Meta_2026. Commit messages carry no AI attribution (user preference).
