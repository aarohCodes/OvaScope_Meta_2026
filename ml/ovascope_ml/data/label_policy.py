"""Explicit, versioned label policies.

Class labels come ONLY from these mappings. Nothing is inferred from file
names, folder names, image appearance or a language model.
"""

from __future__ import annotations

from dataclasses import dataclass

LABEL_POLICY_VERSION = "mmotu-labels-v1"


@dataclass(frozen=True)
class ClassDef:
    index: int
    raw_code_name: str  # as written in the official code
    raw_paper_name: str  # as written in the paper's category figure
    normalized: str
    display: str
    group: str  # descriptive grouping, NOT a diagnosis; see docs/MODEL_CARD.md


# Order verified against (a) official code mmseg/datasets/mmotu.py (commented CLASSES),
# (b) paper arXiv 2207.06799v3 category figure, and (c) per-class counts in
# OTU_2d/train_cls.txt and val_cls.txt, which match the figure exactly.
MMOTU_CLASSES: tuple[ClassDef, ...] = (
    ClassDef(0, "Chocolate_Cyst", "Chocolate cyst", "chocolate_cyst", "Chocolate cyst (endometrioma)", "benign_lesion"),
    ClassDef(1, "SerousCystadenoma", "Serous cystadenoma", "serous_cystadenoma", "Serous cystadenoma", "benign_lesion"),
    ClassDef(2, "Teratoma", "Teratoma", "teratoma", "Teratoma", "benign_lesion"),
    ClassDef(3, "ThecaCellTumor", "Theca cell tumor", "theca_cell_tumor", "Theca cell tumour", "benign_lesion"),
    ClassDef(4, "SimpleCyst", "Simple cyst", "simple_cyst", "Simple cyst", "benign_lesion"),
    ClassDef(5, "NormalOvary", "Normal ovary", "normal_ovary", "Normal ovary", "no_lesion"),
    ClassDef(6, "MucinousCystadenoma", "Mucinous cystadenoma", "mucinous_cystadenoma", "Mucinous cystadenoma", "benign_lesion"),
    # The code name says "Cystadenoma"; the paper figure says "High grade serous".
    # High-grade serous is an invasive carcinoma histotype. The discrepancy is
    # recorded in docs/DECISIONS.md (D-006); we keep the dataset's category name.
    ClassDef(7, "HighGradeSerousCystadenoma", "High grade serous", "high_grade_serous", "High-grade serous (MMOTU category)", "invasive_malignant_category"),
)

MMOTU_REFERENCE_STANDARD = (
    "Category annotation by Ob/Gyn experts at Beijing Shijitan Hospital who 'refer to "
    "pathological reports'; each image annotated by one expert and checked by another "
    "(arXiv 2207.06799v3, Sec. III-A). Per-image pathology reports are not released."
)


@dataclass(frozen=True)
class TargetDef:
    name: str
    description: str
    classes: tuple[str, ...]
    positive_class: str | None = None


TARGETS: dict[str, TargetDef] = {
    "mmotu8": TargetDef(
        name="mmotu8",
        description=(
            "The eight MMOTU OTU_2d image categories (seven lesion categories plus normal "
            "ovary), image level. Primary target."
        ),
        classes=tuple(c.normalized for c in MMOTU_CLASSES),
    ),
    "hgs_vs_other": TargetDef(
        name="hgs_vs_other",
        description=(
            "MMOTU 'High grade serous' category vs all other MMOTU categories, scored by the "
            "primary model's score for that category. Secondary analysis. Not a general "
            "malignancy detector: MMOTU contains no borderline tumours, no other invasive "
            "histotypes, no metastatic disease, and no stage information."
        ),
        classes=("other_mmotu_category", "high_grade_serous"),
        positive_class="high_grade_serous",
    ),
}


def mmotu_index_to_class(index: int) -> ClassDef:
    if not 0 <= index < len(MMOTU_CLASSES):
        raise ValueError(f"MMOTU class index {index} outside the documented range 0..7")
    cd = MMOTU_CLASSES[index]
    assert cd.index == index
    return cd


def class_mapping_payload() -> list[dict[str, object]]:
    """Serializable class mapping for API results and model cards."""
    return [
        {
            "index": c.index,
            "label": c.normalized,
            "display": c.display,
            "source_code_name": c.raw_code_name,
            "source_paper_name": c.raw_paper_name,
        }
        for c in MMOTU_CLASSES
    ]
