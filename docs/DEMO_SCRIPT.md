# Three-minute demonstration script

Research prototype. Not for diagnosis, screening, or treatment decisions.

**Audience:** event judges and ultrasound researchers. **Cases:** the held-out MMOTU test images
listed in `docs/generated/DEMO_CASES.md`, chosen by the fixed rules in
`scripts/select_demo_cases.py` (curated; say so on stage). Upload them from the local research copy
`data/raw/mmotu_hf/OTU_2d/images/`. Never show them outside the demo machine.

Setup before going on stage: `python tasks.py dev` with the evaluated bundle ACTIVE; open
http://localhost:3000; have the three image files in a folder; close other CPU-heavy apps (one
analysis takes about 5-15 s on this laptop CPU).

| Time | Show | Say (keep to evidence) |
|---|---|---|
| 0:00-0:20 | Overview | "OvaScope is a research prototype for researchers evaluating an investigational model on already-acquired ovarian ultrasound stills. The amber banner is permanent: not for diagnosis, screening or treatment. Readiness here is live: DINOv2 and SAM 2.1 are the Meta models, with their exact Hub IDs." |
| 0:20-0:35 | Overview data card | "The data are MMOTU: 1,469 images, 8 categories, single centre, no patient IDs. So every number you'll see is image-level, and we say patient-independent: false." |
| 0:35-1:15 | Analyze → upload **Case A** | Point at the real stage indicator (DINOv2 → localiser + SAM → classifier → attribution). "Segmentation, classifier output and attribution are separate panels. The teal mask came from SAM 2.1, prompted automatically by a box from a DINOv2 patch localiser (dashed). Toggle and fade it. The classifier shows model scores across the eight dataset categories. They're temperature-scaled, not probabilities, and the panel says so." Open the provenance panel: checkpoint SHA-256, evaluation reference. |
| 1:15-1:40 | Upload **Case B** (benign look-alike) | "This benign-category image has a relatively high score for the high-grade serous category. The highest-scoring category is still its reference category. This is why we show all eight scores instead of a yes/no." |
| 1:40-2:05 | Upload **Case C1** (abstained) or **C2** (error) | "Here the top score is below the abstention threshold frozen on development data, so the model declines to report a category. When segmentation fails, the result says which pathway produced the output. It never substitutes a box for a mask." |
| 2:05-2:40 | Evaluation page | Scope card ("External validation not performed", image-level). Balanced accuracy and macro AUROC with CIs vs baselines. Then **"What changed when this Meta component was removed?"**: read the actual ablation rows, including any null or negative result. Mention the segmentation Dice table (localiser vs SAM vs oracle prompt). |
| 2:40-3:00 | Failure cases + limits | "Near-duplicate images cross the official split, and burned-in annotations are possible shortcuts. Only 15 test images are high-grade serous. Next step: an authorised multi-centre dataset with patient IDs, histology and stage." |

Do not say: "detects cancer", "clinically validated", "outperforms doctors", "early-stage",
"works on all devices", "no false negatives", or any number not on the Evaluation page.
Check pitch wording against `docs/CLAIMS_LEDGER.md`.

Fallback if the laptop is slow: the Evaluation page and an already-analysed study (results persist
for 24 h) still show real outputs. Say that they are previously computed results, with their input
SHA-256 and model version shown.
