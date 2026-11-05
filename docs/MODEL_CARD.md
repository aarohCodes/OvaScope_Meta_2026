# Model card: OvaScope MMOTU-8 linear heads

Research prototype. Not for diagnosis, screening, or treatment decisions.

The live card (model version, checkpoint SHA-256, thresholds, evaluation reference) is rendered
by the app at `/cards` from the active bundle's `bundle.json`. This file is the static part.

## Model details

- **Representation:** Meta DINOv2 ViT-S/14 (`timm/vit_small_patch14_dinov2.lvd142m`,
  revision `4610ca14…`, Apache-2.0), frozen. Input: aspect-preserving letterbox to 448×448
  (whole image) or 224×224 (ROI crop). Feature: concat[CLS token, mean patch token] (768-d).
- **Automatic localisation:** logistic regression on DINOv2 patch tokens (32×32 grid), trained on
  development-split reference masks; threshold selected by development out-of-fold Dice; largest
  connected component → box prompt.
- **Segmentation:** Meta SAM 2.1 Hiera-Tiny (`facebook/sam2.1-hiera-tiny`, revision
  `de431c40…`, Apache-2.0), frozen, single-mask output for the box prompt. Generic SAM is not
  trained on ultrasound.
- **Classifier heads:** multinomial logistic regression (class-balanced weights; C chosen by
  development grouped-CV balanced accuracy); temperature fitted on development out-of-fold logits.
  Two deployable pathways: whole image, and whole image + automatic SAM ROI. The primary pathway is
  chosen on development data; the whole-image pathway is the labelled fallback.
- **Storage:** NumPy arrays + JSON, SHA-256 per file, `checkpoint_sha256` over the file map; no pickle.

## Intended use

Research analysis of already-acquired ovarian/adnexal ultrasound stills by ultrasound researchers or
clinicians evaluating an investigational model.

## Prohibited uses

Population screening; diagnosis or exclusion of cancer; treatment, surgical or follow-up decisions;
replacement for clinical examination, pathology or specialist assessment; any statement that the
model is clinically validated, cleared, or detects cancer earlier than clinicians.

## Training data

MMOTU OTU_2d development split (1,000 images, official `train_cls.txt`), single centre (Beijing
Shijitan Hospital). No data licence is published; see `docs/DATASET_READINESS.md`. No patient
identifiers. Labels: eight dataset categories from expert annotation referring to pathology reports.

## Target and label policy

`mmotu8`: chocolate cyst, serous cystadenoma, teratoma, theca cell tumour, simple cyst, normal ovary,
mucinous cystadenoma, high-grade serous (MMOTU category). Labels come only from the official files
via an explicit mapping. No borderline category exists; none is merged. Stage is never inferred.

## Evaluation scope

Official MMOTU test split (469 images), evaluated **once** after all selection was frozen
(`selection.json`, `test_labels_read: false`). Image level; patient-independent evaluation **not**
established; external validation **not** performed. Metrics, curves, intervals and ablations:
`docs/generated/EVALUATION_REPORT.md` and the app's Evaluation page.

## Scores, calibration, abstention

- Scores are temperature-scaled softmax outputs of class-balanced heads: **not calibrated
  probabilities**, not personal risk. Top-label ECE on test is reported.
- Abstention: highest score below a threshold set at 90% development out-of-fold coverage.
- Novelty heuristic: Mahalanobis distance in a 32-component PCA of development DINOv2 features,
  threshold at the development 99th percentile. **Not validated out-of-distribution detection.**

## Attribution

Linear decomposition of the whole-image mean-patch-token term of the highest-scoring category's logit
(relative to the mean over categories). Exact for that linear term only; patch tokens are contextual,
so it shows model sensitivity, not where disease is.

## Measured performance (held-out, image level)

Deployed pathway (DINOv2 whole image): balanced accuracy 0.482 (95% CI 0.428-0.540), macro AUROC
0.841 (0.809-0.872), top-label ECE 0.026, 93% coverage at the frozen abstention threshold. Not
clearly different from a supervised ResNet-50; the SAM ROI was not shown to help; automatic SAM masks
(Dice 0.636) were worse than the localiser's (0.730). Details: `docs/generated/EVALUATION_REPORT.md`.

## Known limitations

- Near-duplicate images span development and test (60 suspected cross-split pairs), and same-patient
  images are likely in both. Held-out metrics are probably optimistic.
- Burned-in calipers, measurement text, annotation labels (for example "CY") and device text remain,
  as does colour Doppler. These are potential shortcuts (metadata-only probe: 0.22 test balanced
  accuracy vs 0.125 chance).
- 15 held-out "High grade serous" images: any malignancy-related estimate is highly uncertain.
- Single centre; device mix unknown; generalisation to other machines or protocols is untested.
- CPU-only, frozen features; no fine-tuning was attempted.
