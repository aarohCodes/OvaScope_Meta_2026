# Demonstration cases (generated, curated by fixed rules)

Research prototype. Not for diagnosis, screening, or treatment decisions.

Experiment `mmotu-otu2d-v1`, primary pathway `dinov2_whole`, held-out test split. Selection rules are in `scripts/select_demo_cases.py`. These cases are **curated** and are not representative on their own; aggregate context: balanced accuracy 0.482 (95% CI 0.428-0.540), image level, n=469. External validation not performed.

| Case | Rule | Image (local path) | Reference category | Highest-scoring | Top score | HGS score |
|---|---|---|---|---|---:|---:|
| A | correct, no dev near-duplicate, smallest top-2 margin | `mmotu2d-877` (`OTU_2d/images/877.JPG`) | simple_cyst | simple_cyst | 0.39 | 0.01 |
| B | non-HGS, correct, highest HGS score (look-alike) | `mmotu2d-621` (`OTU_2d/images/621.JPG`) | normal_ovary | normal_ovary | 0.39 | 0.36 |
| C1 | abstained, lowest top score | `mmotu2d-1134` (`OTU_2d/images/1134.JPG`) | theca_cell_tumor | high_grade_serous | 0.21 | 0.21 |
| C2 | confident error (highest top score among wrong, not abstained) | `mmotu2d-148` (`OTU_2d/images/148.JPG`) | serous_cystadenoma | theca_cell_tumor | 0.94 | 0.02 |

Images are local research copies and are never committed or redistributed.
