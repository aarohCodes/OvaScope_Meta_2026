# Dataset readiness report

Research prototype. Not for diagnosis, screening, or treatment decisions.

Prepared 2026-11-04. Computed counts are reproduced by `python tasks.py data-audit`
(output: `docs/generated/DATA_AUDIT.md`, `artifacts/evaluation/data_audit.json`).

## 1. User-provided data

None found. The home directory and Downloads folder were searched for ovarian/adnexal
ultrasound data, MMOTU archives or related files; nothing relevant exists. **Open
question for the user:** is there an authorised dataset with patient identifiers,
histology and stage? If so, it should replace MMOTU as the primary source.

## 2. Public dataset investigated: MMOTU (OTU_2d)

| Field | Finding | Evidence |
|---|---|---|
| Name | Multi-Modality Ovarian Tumor Ultrasound (MMOTU), 2D subset OTU_2d | arXiv 2207.06799v3 |
| Official source | Google Drive folder linked from `github.com/cv516Buaa/MMOTU_DS2Net` | repository README |
| Transport used | Hugging Face mirror `norayao/MMOTU` @ `d29379c60ac4072cfbf9c6f960505d43f98b3f0d` (Drive bulk download rate-limited after 87 files) | `scripts/fetch_mmotu_mirror.py` |
| Integrity | 4,407/4,407 LFS files match the mirror's SHA-256 records; 87/87 files fetched directly from the official Drive folder are byte-identical to the mirror copy | `MIRROR_PROVENANCE.json`, `artifacts/evaluation/mirror_verification.json` |
| Version / retrieval | OTU_2d as released (1,469 images); retrieved 2026-11-04 | |
| Licence | **No data licence published.** Repository code is Apache-2.0; the mirror's `apache-2.0` tag was set by its uploader. Neither is treated as a licence for the images. | GitHub API, HF API |
| Access conditions | Public download, no registration or data-use agreement observed | |
| Permitted training use | Not explicitly stated. Used for local research development only. | |
| Permitted redistribution | Not established. Images, masks and per-image derived data are never committed or redistributed. | `.gitignore`, `tasks.py check-no-data` |
| Image count | 1,469 (development 1,000 / test 469, official split) | `*_cls.txt`, paper Table I |
| Unique patients | **Not released.** Paper: 294 patients for all 1,639 MMOTU images (OTU_2d + CEUS); OTU_2d count not stated | paper Sec. III |
| Study / lesion counts | Not released | |
| Labels | 8 categories, 0-based indices in `train_cls.txt` / `val_cls.txt` | official code + paper figure |
| Reference standard | Expert Ob/Gyn annotation "referring to pathological reports", annotated by one expert and checked by another. Per-image pathology not released | paper Sec. III-A |
| Pathology / histology fields | Category only | |
| Stage fields | **None** | |
| Borderline category | **None** | |
| Masks | Binary lesion masks (`*_binary.PNG`, values 0/1); RGB semantic masks whose colour encodes the category. For "Normal ovary" the mask outlines the ovary | files, paper |
| Exclusions / missingness | None reported by the authors; our audit found no missing files, corrupt images or mask-size mismatches | data audit |
| De-identification | Authors manually cropped images to remove private information. **Burned-in device text, calipers, measurements and annotation labels remain.** | paper Fig. 3 |
| External API transmission | **Not authorised.** All processing is local. | `external_api_transmission_authorized: false` |

## 3. Label mapping (explicit)

| Index | Official code name | Paper figure name | Normalised label |
|---:|---|---|---|
| 0 | Chocolate_Cyst | Chocolate cyst | `chocolate_cyst` |
| 1 | SerousCystadenoma | Serous cystadenoma | `serous_cystadenoma` |
| 2 | Teratoma | Teratoma | `teratoma` |
| 3 | ThecaCellTumor | Theca cell tumor | `theca_cell_tumor` |
| 4 | SimpleCyst | Simple cyst | `simple_cyst` |
| 5 | NormalOvary | Normal ovary | `normal_ovary` |
| 6 | MucinousCystadenoma | Mucinous cystadenoma | `mucinous_cystadenoma` |
| 7 | HighGradeSerousCystadenoma | High grade serous | `high_grade_serous` |

The order was confirmed three ways: the official code's commented 8-class block, the paper's
per-class count figure (counts match `*_cls.txt` exactly in both splits), and the
semantic-mask colour palette (zero disagreements across 1,469 images). The index-7 naming
discrepancy ("Cystadenoma" in code vs "High grade serous" in the paper) is recorded, not resolved
by assumption.

## 4. Audit results (from `ovascope-ml data-audit`)

- Missing files, invalid images, mask dimension mismatches, empty masks, exact duplicates,
  conflicting labels: **none**.
- Suspected near-duplicate pairs (pHash ≤ 6 and dHash ≤ 8): **130**, of which **60 span
  development and test** (113 images involved). Heuristic only.
- Near-duplicate pairs with **different labels**: 16 images (possible distinct lesions,
  annotation conflicts, or hash false positives; not adjudicated).
- Metadata-only shortcut probe (frame size, aspect ratio, colour-pixel fraction):
  balanced accuracy 0.184 (dev CV) / 0.219 (test) for 8 categories vs chance 0.125;
  0.574 / 0.510 for High-grade serous vs other. Weak but real acquisition signal for the
  8-class target; negligible on test for the binary analysis.

## 5. What MMOTU can support

| Question | Supported? |
|---|---|
| Image-level, single-centre, exploratory classification of the 8 categories | Yes, with caveats |
| Patient-level generalisation | **No**: no patient IDs; same-patient images likely span splits |
| External validation | **No**: single dataset, single centre |
| Early invasive-stage evaluation | **No**: no stage labels; stage is never inferred |
| Borderline vs invasive | **No**: no borderline category |
| Malignancy detection in general | **No**: one invasive histotype only (53 images in total, 15 in test) |
| Lesion segmentation (Dice/IoU) | Yes, image-level, against expert masks |

## 6. Blockers for the conditional research extension

A dataset with (a) pseudonymous patient IDs, (b) histopathology including borderline and
invasive histotypes, (c) FIGO stage, (d) documented consent/licence for model training, and
(e) preferably more than one centre. None was available on this machine.
