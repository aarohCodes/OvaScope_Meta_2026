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

## Measured results (experiment `mmotu-otu2d-v1`, MMOTU OTU_2d official test split, n = 469 images)

All numbers are image-level, single-centre, not patient-independent; 95% CIs are image-level
bootstrap and likely too narrow. Source: `artifacts/evaluation/evaluation.json` /
`docs/generated/EVALUATION_REPORT.md`.

| claim | supporting_artifact | scope | limitation | approved_for_pitch |
|---|---|---|---|---|
| The deployed model (DINOv2 ViT-S/14 whole image, chosen on development data) reached balanced accuracy 0.482 (95% CI 0.428–0.540) and macro AUROC 0.841 (0.809–0.872) across 8 MMOTU categories; chance balanced accuracy 0.125 | evaluation.json `models.dinov2_whole` | MMOTU test split, image level | inflated by near-duplicates: 0.802 on 56 test images with a development near-duplicate vs 0.453 on the other 413; not patient-independent; external validation not performed | yes, with the near-duplicate caveat stated alongside |
| DINOv2 features beat hand-crafted intensity/texture features (+0.088 balanced accuracy, CI 0.027–0.154; +0.080 macro AUROC, CI 0.044–0.114) | ablations[1] | same | same | yes |
| DINOv2 beats a supervised ImageNet ResNet-50 | ablations[0]: −0.032 balanced accuracy (CI −0.093 to 0.026); +0.008 macro AUROC (CI −0.020 to 0.037) | same | **no clear difference** | **no**: say "no clear difference from ResNet-50" |
| Adding the automatic SAM region of interest improves classification | ablations[2]: +0.041 balanced accuracy (CI −0.000 to 0.085); +0.002 macro AUROC (CI −0.012 to 0.016) | same | CI includes 0; development data selected whole-image | **no**: say "not shown to help; development selection kept whole-image" |
| SAM refinement improves the automatic mask | segmentation: SAM auto mean Dice 0.636 vs DINOv2 localiser 0.730 (difference −0.094, CI −0.116 to −0.072) | test split, against expert masks | SAM was **worse** than the localiser mask | **no** (negative result; report it) |
| Given an expert box, SAM 2.1 reaches mean Dice 0.789 (median 0.907) | segmentation `sam_oracle` | test split | oracle prompt; not automatic | yes, labelled as an upper bound |
| The automatic localiser finds a candidate region in 99.1% of test images (mean Dice 0.730) | segmentation `all_test_images` | test split | Dice for normal-ovary images measures the ovary outline | yes |
| A reference-mask ROI (oracle) would reach balanced accuracy 0.577 (0.520–0.635) | models.oracle_dinov2_roi_gt | test split | uses expert masks at test time | yes, labelled as an upper bound |
| Selective prediction: abstaining on 7% of test images (threshold frozen on development data) gives retained accuracy 0.585 vs 0.563 overall | models.dinov2_whole.abstention | test split | small gain; threshold is a heuristic | yes |
| Top-label calibration error 0.026 on the test split | models.dinov2_whole.mmotu8.calibration | test split | scores are temperature-scaled outputs of class-balanced heads, not prevalence-calibrated probabilities | yes, as "top-label ECE 0.026"; never as "probability" |
| "High grade serous" vs other: AUROC 0.815 (0.690–0.916) | models.dinov2_whole.hgs_vs_other | 15 positive test images | very wide CI; possibly few patients; one invasive histotype only; not a cancer detector | only as "AUROC for the MMOTU 'High grade serous' category, 15 positives"; **never** as "detects cancer" |
| The live app reproduces the evaluated scores exactly | `test_live_engine_reproduces_evaluated_predictions` (max difference 0.0 on checked cases) | deployed bundle | spot check of held-out images | yes |
| Colour Doppler / coloured-annotation images: balanced accuracy 0.398 (n=62) vs greyscale 0.491 (n=407) | sensitivity_analyses.colour_content | test split | subgroup, small n | yes, as a limitation |
