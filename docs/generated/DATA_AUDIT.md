# MMOTU OTU_2d data audit (generated)

Research prototype. Not for diagnosis, screening, or treatment decisions.

Generated 2026-11-04T21:38:27.670854+00:00 by `ovascope-ml data-audit`. Aggregate values only.

- Manifest `mmotu-otu2d-v1` sha256 `3cbb934c685384859012ad3ea3e18c1c7688beb35c0473cf5a1748c55117e5da`
- Split sha256 `0f07706144e5563f2922ea60d1cae4bcb1e6d6bc34c703420a6be580c1231dae`; label policy `mmotu-labels-v1`
- Images: 1469; patients: not released (Not released. The paper reports 294 patients for the full MMOTU dataset (OTU_2d + CEUS, 1,639 images); the OTU_2d patient count is not stated.)
- Split unit: **image**; patient-independent: **False**
- Patient identifiers are not available. Evaluation is image-level and exploratory; patient-independent evaluation cannot be established and claims of patient-level generalisation are not supported. Images of one patient may appear in both development and test.
- Stage labels: False; early-stage analysis: blocked: no stage labels in MMOTU

## Class distribution

| Category | Development | Test |
|---|---:|---:|
| chocolate_cyst | 226 | 110 |
| serous_cystadenoma | 153 | 66 |
| teratoma | 228 | 108 |
| theca_cell_tumor | 57 | 31 |
| simple_cyst | 47 | 19 |
| normal_ovary | 180 | 87 |
| mucinous_cystadenoma | 71 | 33 |
| high_grade_serous | 38 | 15 |

## Validation issues

| Code | Severity | Images | Detail |
|---|---|---:|---|
| suspected_near_duplicate_across_splits | warning | 113 | perceptual-hash near duplicates span development and test (possible leakage; heuristic, not proof of same patient) |
| near_duplicate_label_disagreement | warning | 16 | suspected near duplicates carry different labels (review: different lesions, annotation conflict, or hash false positive) |
| missing_grouping_identifiers | warning | 1 | no patient identifiers: patient-independent evaluation cannot be established; metrics are image-level and exploratory |

## Near duplicates (perceptual hash; heuristic)

- Thresholds: {'max_phash_hamming': 6, 'max_dhash_hamming': 8}
- Suspected pairs: 130 (cross-split: 60)
- Clusters with >1 image: 107 covering 227 images

## Shortcut audit: metadata-only probe

Features: width, height, aspect, log_area, chroma_fraction. Model: HistGradientBoostingClassifier(max_depth=3, max_iter=150).

| Target | Dev grouped-CV balanced acc. | Test balanced acc. | Chance |
|---|---:|---:|---:|
| mmotu8 | 0.184 | 0.219 | 0.125 |
| hgs_vs_other | 0.574 | 0.510 | 0.500 |

Balanced accuracy of a classifier that sees only frame size, aspect ratio and colour-pixel fraction. Values well above chance mean acquisition/annotation artefacts carry label information that image models can also exploit.

| Category | n | Colour pixels >1% | Median colour fraction | Top frame-size clusters |
|---|---:|---:|---:|---|
| chocolate_cyst | 336 | 0.11 | 0.0025 | 960x540 (100), 480x540 (75), 960x520 (29) |
| serous_cystadenoma | 219 | 0.16 | 0.0025 | 960x540 (69), 480x540 (27), 960x520 (17) |
| teratoma | 336 | 0.17 | 0.0026 | 960x540 (53), 480x540 (35), 960x520 (31) |
| theca_cell_tumor | 88 | 0.12 | 0.0025 | 960x540 (24), 480x540 (10), 960x560 (7) |
| simple_cyst | 66 | 0.21 | 0.0027 | 960x540 (17), 480x540 (13), 560x600 (6) |
| normal_ovary | 267 | 0.07 | 0.0023 | 480x540 (62), 960x540 (50), 480x520 (26) |
| mucinous_cystadenoma | 104 | 0.12 | 0.0023 | 960x540 (31), 480x540 (18), 480x520 (5) |
| high_grade_serous | 53 | 0.15 | 0.0023 | 960x520 (12), 960x540 (11), 480x540 (8) |

The dataset paper states most images carry expert-drawn symbols (lines, hands, characters, arrows); sample figures show device settings text, calipers, measurements and labels such as 'CY' and 'LOV'. These were not removed.
