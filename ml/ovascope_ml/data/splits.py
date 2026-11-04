"""Split construction and leakage guards.

Rules:
  * When patient identifiers exist, every image/crop/derivative of a patient
    stays in one split, and CV folds are grouped by patient.
  * When they do not (MMOTU), CV folds are grouped by suspected near-duplicate
    cluster. That reduces, but does not remove, same-patient leakage; metrics
    are labelled image-level and exploratory.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ovascope_ml.data.manifest import ManifestRecord


class LeakageError(RuntimeError):
    pass


@dataclass(frozen=True)
class SplitStatement:
    unit: str
    patient_independent: bool
    grouping_used_for_cv: str
    statement: str


def split_statement(records: Sequence[ManifestRecord]) -> SplitStatement:
    has_patients = all(r.patient_id is not None for r in records)
    if has_patients:
        return SplitStatement(
            unit="patient",
            patient_independent=True,
            grouping_used_for_cv="patient_id",
            statement="Splits and CV folds are grouped by pseudonymous patient identifier.",
        )
    return SplitStatement(
        unit="image",
        patient_independent=False,
        grouping_used_for_cv="suspected near-duplicate cluster (perceptual hash)",
        statement=(
            "Patient identifiers are not available. Evaluation is image-level and exploratory; "
            "patient-independent evaluation cannot be established and claims of patient-level "
            "generalisation are not supported. Images of one patient may appear in both "
            "development and test."
        ),
    )


def group_keys(records: Sequence[ManifestRecord], near_dup_cluster: dict[str, int]) -> list[str]:
    """Patient ID when present, otherwise near-duplicate cluster, otherwise the image itself."""
    keys = []
    for r in records:
        if r.patient_id is not None:
            keys.append(f"patient:{r.patient_id}")
        elif r.image_id in near_dup_cluster:
            keys.append(f"ndcluster:{near_dup_cluster[r.image_id]}")
        else:
            keys.append(f"image:{r.image_id}")
    return keys


def assert_no_group_leakage(splits: Sequence[str | None], groups: Sequence[str]) -> None:
    seen: dict[str, set[str | None]] = collections.defaultdict(set)
    for s, g in zip(splits, groups, strict=True):
        seen[g].add(s)
    leaked = sorted(g for g, s in seen.items() if len(s) > 1 and g.startswith("patient:"))
    if leaked:
        raise LeakageError(f"{len(leaked)} patient groups span more than one split: {leaked[:5]}")


def grouped_cv_folds(
    labels: Sequence[int], groups: Sequence[str], n_folds: int = 5, seed: int = 20261104
) -> np.ndarray:
    """Fold index per sample; all samples of a group share a fold."""
    y = np.asarray(labels)
    g = np.asarray(groups)
    folds = np.full(len(y), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(splitter.split(np.zeros(len(y)), y, g)):
        folds[val_idx] = k
    if (folds < 0).any():
        raise RuntimeError("some samples were not assigned to a fold")
    for grp in np.unique(g):
        if len(np.unique(folds[g == grp])) != 1:
            raise LeakageError(f"group {grp} split across CV folds")
    return folds
