# Claims ledger

Research prototype. Not for diagnosis, screening, or treatment decisions.

Only claims marked **yes** in `approved_for_pitch` may be said or shown. Numbers must be quoted
from `artifacts/evaluation/evaluation.json` (rendered on the Evaluation page and in
`docs/generated/EVALUATION_REPORT.md`), never from other papers.

## Engineering and process claims

| claim | supporting_artifact | scope | limitation | approved_for_pitch |
|---|---|---|---|---|
| Without a trained, integrity-checked bundle the app returns `model_unavailable` and no score | `tests/ml/test_engine_honesty.py::test_missing_checkpoint_gives_model_unavailable_without_scores`; E2E `NO_MODEL mode` | this codebase | — | yes |
| Inference code cannot access labels or reference masks | `test_inference_code_cannot_reach_labels_or_reference_masks`; `analyze()` signature test | static import check + signature | does not prove absence of label leakage *in the images themselves* (burned-in text) | yes |
| Every displayed metric comes from a reproducible evaluation artifact | `ovascope-ml evaluate` → `metrics.json` (SHA-256 in provenance); E2E `evaluation.spec.ts` | MMOTU OTU_2d test split | single dataset | yes |
| Test labels were not used for model selection, calibration or thresholds | `selection.json` (`test_labels_read: false`, `frozen_at`) written before evaluation; code separation | this experiment | process claim; relies on code review | yes |
| SAM masks are only shown when SAM produced them; failures are explicit | `test_segmentation_failure_never_yields_a_mask_and_fallback_is_labelled` | this codebase | — | yes |
| Uses Meta DINOv2 and SAM 2.1 (exact Hub IDs and revisions shown in the UI) | `ml/ovascope_ml/models/registry.py`; provenance panel | — | frozen, not fine-tuned; generic SAM not trained on ultrasound | yes |
| Upload validation rejects oversized, corrupt, non-PNG/JPEG and decompression-bomb inputs | `tests/ml/test_imaging_and_hashing.py`, `tests/api/test_api.py` | — | no malware scanning | yes |
| No file names, tokens or image bytes in logs | `test_no_phi_in_logs` | API logs | burned-in text in pixels is not detected | yes |
| Images never leave the machine; external transmission fails closed | capabilities endpoint; Settings page; Modal adapter gate | local deployment | — | yes |
| Dataset copy verified against the official source | `mirror_verification.json` (87/87 identical) + 4,407/4,407 LFS hashes | MMOTU OTU_2d | spot check of 87 official files, not all 4,411 | yes |
| "HIPAA compliant" | none | — | not assessed | **no** |
| "Clinically validated", "FDA approved/cleared" | none | — | not done | **no** |
| "Detects cancer", "detects stage I cancer", "early detection" | none: no stage labels; one invasive histotype | — | not evaluable with MMOTU | **no** |
| "Outperforms doctors/radiologists" | none: no reader study | — | — | **no** |
| "Works on all ultrasound devices" | none: single centre | — | — | **no** |
| "Prevents deaths", "no false negatives", "first ever" | none | — | — | **no** |
| "Patient-level" or "external" validation | none: no patient IDs; single dataset | — | "External validation not performed." | **no** |
