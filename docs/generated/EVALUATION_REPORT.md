# OvaScope evaluation report (generated)

Research prototype. Not for diagnosis, screening, or treatment decisions.

Experiment `mmotu-otu2d-v1`, generated 2026-11-05T00:01:02.881028+00:00.

## Scope

- Dataset: MMOTU OTU_2d; test split: official MMOTU val list (469 images), fixed by the dataset authors; n = 469 images.
- Evaluation unit: **image**. Patient-independent: **False**.
- Patient identifiers are not available. Evaluation is image-level and exploratory; patient-independent evaluation cannot be established and claims of patient-level generalisation are not supported. Images of one patient may appear in both development and test.
- **External validation not performed.**
- PPV/NPV and prevalence are specific to this enriched single-centre research cohort and do not transfer to screening or other populations.
- 95% intervals: image-level percentile bootstrap. Patient identifiers are not available, so correlation between images of one patient is ignored and the intervals are likely too narrow.
- Early-stage analysis: MMOTU has no stage labels; stage is never inferred.

Primary deployed pathway: `dinov2_whole`. Rule: Among deployable DINOv2 pathways (whole image; whole image + automatic SAM ROI), the one with higher development out-of-fold balanced accuracy. Chosen before test evaluation.

## Primary target `mmotu8` (8 MMOTU categories)

| Model | Role | Balanced acc. (95% CI) | Macro AUROC (95% CI) | Accuracy | Top-label ECE | Abstention coverage | Retained acc. |
|---|---|---|---|---:|---:|---:|---:|
| Majority class | baseline | 0.125 (0.125-0.125) | 0.500 (0.500-0.500) | 0.230 | 0.002 | 1.000 | 0.230 |
| Hand-crafted intensity/texture + LR | baseline | 0.394 (0.341-0.448) | 0.761 (0.726-0.795) | 0.416 | 0.070 | 0.904 | 0.432 |
| ImageNet ResNet-50 (frozen) + LR | comparator | 0.514 (0.462-0.572) | 0.833 (0.799-0.867) | 0.593 | 0.073 | 0.932 | 0.613 |
| DINOv2 ViT-S/14 whole image + LR | meta | 0.482 (0.428-0.540) | 0.841 (0.809-0.872) | 0.563 | 0.026 | 0.930 | 0.585 |
| DINOv2 whole + localiser-box ROI (no SAM) | ablation | 0.519 (0.464-0.576) | 0.841 (0.807-0.874) | 0.595 | 0.049 | 0.910 | 0.623 |
| DINOv2 whole + automatic SAM 2.1 ROI | meta | 0.523 (0.464-0.584) | 0.843 (0.812-0.874) | 0.593 | 0.060 | 0.885 | 0.648 |
| ORACLE: DINOv2 whole + reference-mask ROI | oracle | 0.577 (0.520-0.635) | 0.889 (0.860-0.916) | 0.665 | 0.053 | 0.919 | 0.691 |

The oracle row uses reference masks at test time and is an upper bound, not an automatic result.

## Secondary target `hgs_vs_other`

Score for the MMOTU 'High grade serous' category vs all other categories, threshold frozen on development data.

| Model | n pos / n | AUROC (95% CI) | AUPRC (prevalence) | Sensitivity | Specificity | PPV | NPV |
|---|---|---|---|---:|---:|---:|---:|
| Majority class | 15/469 | 0.500 (0.500-0.500) | 0.032 (0.032) | 0.000 | 1.000 | n/a | 0.968 |
| Hand-crafted intensity/texture + LR | 15/469 | 0.678 (0.515-0.819) | 0.180 (0.032) | 0.400 | 0.804 | 0.063 | 0.976 |
| ImageNet ResNet-50 (frozen) + LR | 15/469 | 0.785 (0.636-0.907) | 0.245 (0.032) | 0.800 | 0.599 | 0.062 | 0.989 |
| DINOv2 ViT-S/14 whole image + LR | 15/469 | 0.815 (0.690-0.916) | 0.242 (0.032) | 0.733 | 0.742 | 0.086 | 0.988 |
| DINOv2 whole + localiser-box ROI (no SAM) | 15/469 | 0.822 (0.682-0.935) | 0.429 (0.032) | 0.667 | 0.797 | 0.098 | 0.986 |
| DINOv2 whole + automatic SAM 2.1 ROI | 15/469 | 0.849 (0.734-0.940) | 0.396 (0.032) | 0.667 | 0.828 | 0.114 | 0.987 |
| ORACLE: DINOv2 whole + reference-mask ROI | 15/469 | 0.915 (0.838-0.973) | 0.405 (0.032) | 0.867 | 0.846 | 0.157 | 0.995 |

With only 15 positive test images, possibly from a few patients, these estimates are highly uncertain.

## Ablations: what changed when a component was removed?

| Question | Metric | A - B | 95% CI | Resamples with A better |
|---|---|---:|---|---:|
| Does the DINOv2 representation beat a supervised ImageNet ResNet-50 (whole image)? | balanced_accuracy | -0.032 | -0.093 to 0.026 | 0.14 |
| Does the DINOv2 representation beat a supervised ImageNet ResNet-50 (whole image)? | macro_auroc | 0.008 | -0.020 to 0.037 | 0.72 |
| Does the DINOv2 representation beat a supervised ImageNet ResNet-50 (whole image)? | hgs_auroc | 0.030 | -0.081 to 0.166 | 0.67 |
| Does the DINOv2 representation beat hand-crafted intensity/texture features? | balanced_accuracy | 0.088 | 0.027 to 0.154 | 1.00 |
| Does the DINOv2 representation beat hand-crafted intensity/texture features? | macro_auroc | 0.080 | 0.044 to 0.114 | 1.00 |
| Does the DINOv2 representation beat hand-crafted intensity/texture features? | hgs_auroc | 0.137 | 0.013 to 0.261 | 0.98 |
| Does adding the automatic SAM region of interest help over the whole image? | balanced_accuracy | 0.041 | -0.000 to 0.085 | 0.97 |
| Does adding the automatic SAM region of interest help over the whole image? | macro_auroc | 0.002 | -0.012 to 0.016 | 0.61 |
| Does adding the automatic SAM region of interest help over the whole image? | hgs_auroc | 0.035 | -0.023 to 0.104 | 0.86 |
| Does SAM refinement help over the DINOv2 localiser box alone (ROI pathway)? | balanced_accuracy | 0.004 | -0.037 to 0.046 | 0.60 |
| Does SAM refinement help over the DINOv2 localiser box alone (ROI pathway)? | macro_auroc | 0.002 | -0.009 to 0.013 | 0.66 |
| Does SAM refinement help over the DINOv2 localiser box alone (ROI pathway)? | hgs_auroc | 0.028 | -0.015 to 0.087 | 0.85 |
| How far is the automatic ROI from an oracle reference-mask ROI (upper bound)? | balanced_accuracy | 0.054 | 0.000 to 0.106 | 0.97 |
| How far is the automatic ROI from an oracle reference-mask ROI (upper bound)? | macro_auroc | 0.046 | 0.028 to 0.067 | 1.00 |
| How far is the automatic ROI from an oracle reference-mask ROI (upper bound)? | hgs_auroc | 0.065 | -0.036 to 0.181 | 0.90 |

## Segmentation (automatic localisation)

Reference: MMOTU binary masks (*_binary.PNG); for 'normal_ovary' images the mask outlines the ovary, not a lesion. images without an automatic candidate count as Dice 0.

| Subset | n | Candidate rate | Localiser Dice (mean / median) | SAM auto Dice (mean / median) | SAM oracle-prompt Dice (mean / median) |
|---|---:|---:|---|---|---|
| All test images | 469 | 0.991 | 0.730 / 0.824 | 0.636 / 0.775 | 0.789 / 0.907 |
| Lesion categories only | 382 | 0.992 | 0.779 / 0.856 | 0.686 / 0.831 | 0.816 / 0.913 |

SAM minus localiser mean Dice: -0.094 (95% CI -0.116 to -0.072). SAM prompted with the reference-mask box: an upper bound on prompt quality, not an automatic result.

## Sensitivity analyses (primary pathway)

- Test images whose perceptual-hash cluster also contains a development image:
  - with development near-duplicate: n=56, balanced acc. 0.802, accuracy 0.732
  - without development near-duplicate: n=413, balanced acc. 0.453, accuracy 0.540
- Images with >1% colour pixels (colour Doppler and/or coloured annotations):
  - colour present: n=62, balanced acc. 0.398, accuracy 0.484
  - greyscale: n=407, balanced acc. 0.491, accuracy 0.575

## Failure cases

Curated by fixed rule from the test set: highest-scoring wrong predictions that were NOT abstained, lowest-scoring abstained cases, and every 'High grade serous' image scored below the frozen operating threshold.

- confident_errors: mmotu2d-148 (true serous_cystadenoma, top theca_cell_tumor 0.94), mmotu2d-944 (true teratoma, top chocolate_cyst 0.92), mmotu2d-1444 (true high_grade_serous, top simple_cyst 0.91), mmotu2d-751 (true mucinous_cystadenoma, top chocolate_cyst 0.90), mmotu2d-179 (true simple_cyst, top mucinous_cystadenoma 0.90), mmotu2d-383 (true theca_cell_tumor, top mucinous_cystadenoma 0.89)
- abstained: mmotu2d-1134 (true theca_cell_tumor, top high_grade_serous 0.21), mmotu2d-203 (true normal_ovary, top simple_cyst 0.23), mmotu2d-1096 (true chocolate_cyst, top teratoma 0.23), mmotu2d-600 (true simple_cyst, top teratoma 0.24), mmotu2d-217 (true normal_ovary, top normal_ovary 0.24), mmotu2d-1054 (true theca_cell_tumor, top teratoma 0.24)
- hgs_below_threshold: mmotu2d-1444 (true high_grade_serous, top simple_cyst 0.91), mmotu2d-1461 (true high_grade_serous, top chocolate_cyst 0.53), mmotu2d-1469 (true high_grade_serous, top teratoma 0.60), mmotu2d-1460 (true high_grade_serous, top theca_cell_tumor 0.38)

## Provenance

- manifest_sha256: `3cbb934c685384859012ad3ea3e18c1c7688beb35c0473cf5a1748c55117e5da`
- split_sha256: `0f07706144e5563f2922ea60d1cae4bcb1e6d6bc34c703420a6be580c1231dae`
- config_sha256: `9d6fd90b9a03f6deb1ebcc5639f640dc9011f037d553a5f79eb9d3f965e24baf`
- selection_sha256: `38dc9b76836435ac1b060435dd49e015fd2507dc47df554e0fa940113a0ef0ac`
- test_scores_sha256: `e6b20201ca98fa195bc3c8bf322e895dc7fa298fcde449fb6182593c362dcbee`
- selection_frozen_at: `2026-11-04T23:55:47.807969+00:00`
