"""Manifest validation and leakage guards on a synthetic geometric dataset."""

import io

import numpy as np
import pytest
from PIL import Image

from conftest import geometric_png
from ovascope_ml.data.label_policy import MMOTU_CLASSES, TARGETS, mmotu_index_to_class
from ovascope_ml.data.manifest import DatasetSource, Manifest, ManifestRecord
from ovascope_ml.data.splits import LeakageError, assert_no_group_leakage, group_keys, grouped_cv_folds, split_statement
from ovascope_ml.data.validation import validate


def _src(patient_independent=False) -> DatasetSource:
    return DatasetSource(
        name="synthetic",
        official_source="tests",
        transport="tests",
        version="0",
        retrieved_at="2026-11-04",
        license="test",
        access_conditions="test",
        permitted_training_use="test",
        permitted_redistribution="no",
        de_identified="synthetic",
        external_api_transmission_authorized=False,
        split_policy="test",
        split_unit="patient" if patient_independent else "image",
        patient_independent=patient_independent,
    )


def _rec(i, label="teratoma", split="development", patient=None, mask=True) -> ManifestRecord:
    return ManifestRecord(
        image_id=f"img-{i}",
        patient_id=patient,
        image_path=f"images/{i}.png",
        image_sha256="missing",
        source_dataset="synthetic",
        modality="geometric_test_fixture",
        label_raw=label,
        label_normalized=label,
        reference_standard="synthetic",
        split=split,
        mask_path_if_available=f"masks/{i}.png" if mask else None,
        license_reference="test",
    )


def _write(root, i, png: bytes, mask_shape=None):
    (root / "images").mkdir(exist_ok=True)
    (root / "masks").mkdir(exist_ok=True)
    (root / "images" / f"{i}.png").write_bytes(png)
    w, h = Image.open(io.BytesIO(png)).size
    mh, mw = mask_shape or (h, w)
    m = np.zeros((mh, mw), np.uint8)
    m[mh // 4 : mh // 2, mw // 4 : mw // 2] = 1
    Image.fromarray(m).save(root / "masks" / f"{i}.png")


def test_validation_flags_every_issue_class(tmp_path):
    _write(tmp_path, 0, geometric_png(seed=0))
    _write(tmp_path, 1, geometric_png(seed=0))  # exact duplicate of 0, different label
    _write(tmp_path, 2, geometric_png(seed=5, shape="rect"), mask_shape=(10, 10))  # mask dims wrong
    (tmp_path / "images" / "3.png").write_bytes(b"\x89PNG\r\n\x1a\nnot really")  # invalid
    _write(tmp_path, 5, geometric_png(seed=9, shape="rect"))
    recs = [_rec(0, "teratoma"), _rec(1, "simple_cyst", split="test"), _rec(2), _rec(3), _rec(4), _rec(5, mask=False)]
    m = Manifest(manifest_version="t", label_policy_version="t", source=_src(), records=recs)
    report = validate(m, tmp_path)
    codes = report.counts()
    assert codes["missing_file"] == 1
    assert codes["invalid_image"] == 1
    assert codes["mask_dimension_mismatch"] == 1
    assert codes["mask_missing"] == 1
    assert codes["exact_duplicate_pixels"] == 2
    assert codes["conflicting_labels"] == 2
    assert codes["exact_duplicate_across_splits"] == 2
    assert "missing_grouping_identifiers" in codes
    assert report.has_errors()
    agg = report.aggregate()
    assert "image_path" not in str(agg) and str(tmp_path) not in str(agg)  # commit-safe: no paths


def test_patient_group_leakage_detected(tmp_path):
    for i in range(3):
        _write(tmp_path, i, geometric_png(seed=10 + i, shape="rect" if i else "ellipse"))
    recs = [_rec(0, patient="p1"), _rec(1, patient="p1", split="test"), _rec(2, patient="p2")]
    report = validate(
        Manifest(manifest_version="t", label_policy_version="t", source=_src(True), records=recs), tmp_path
    )
    assert report.counts().get("patient_group_leakage") == 1
    assert report.counts().get("multiple_images_per_patient") == 1
    with pytest.raises(LeakageError):
        assert_no_group_leakage([r.split for r in recs], group_keys(recs, {}))


def test_grouped_folds_keep_groups_together():
    labels = [i % 4 for i in range(80)]
    groups = [f"g{i // 3}" for i in range(80)]
    folds = grouped_cv_folds(labels, groups, n_folds=4)
    for g in set(groups):
        assert len({folds[i] for i in range(80) if groups[i] == g}) == 1


def test_split_statement_blocks_patient_claims_without_ids():
    st = split_statement([_rec(0), _rec(1)])
    assert st.unit == "image" and not st.patient_independent
    assert "cannot be established" in st.statement
    assert split_statement([_rec(0, patient="a")]).patient_independent


def test_mmotu_label_mapping_is_explicit_and_complete():
    assert [c.normalized for c in MMOTU_CLASSES] == list(TARGETS["mmotu8"].classes)
    assert mmotu_index_to_class(0).normalized == "chocolate_cyst"
    assert mmotu_index_to_class(5).normalized == "normal_ovary"
    assert mmotu_index_to_class(7).normalized == "high_grade_serous"
    assert mmotu_index_to_class(7).raw_code_name == "HighGradeSerousCystadenoma"  # discrepancy recorded, not hidden
    with pytest.raises(ValueError):
        mmotu_index_to_class(8)
    # Borderline is not silently merged anywhere.
    assert not any("borderline" in c.normalized for c in MMOTU_CLASSES)


def test_semantic_palette_cross_check():
    from ovascope_ml.data.audit import mmotu_semantic_check

    rec = ManifestRecord(
        image_id="mmotu2d-1",
        image_path="x",
        image_sha256="x",
        source_dataset="m",
        modality="u",
        label_raw="4:SimpleCyst",
        label_normalized="simple_cyst",
        reference_standard="r",
        license_reference="l",
    )
    sem = np.zeros((20, 20, 3), np.uint8)
    assert mmotu_semantic_check(rec, sem) == "semantic_mask_empty"
    sem[5:10, 5:10] = (64, 64, 0)  # palette index 4
    assert mmotu_semantic_check(rec, sem) is None
    sem[0:2, 0:2] = (64, 0, 0)  # palette index 0 -> conflict
    assert mmotu_semantic_check(rec, sem).startswith("palette_")


def test_manifest_hash_is_stable_and_content_sensitive():
    m1 = Manifest(manifest_version="t", label_policy_version="t", source=_src(), records=[_rec(0), _rec(1)])
    m2 = Manifest(manifest_version="t", label_policy_version="t", source=_src(), records=[_rec(1), _rec(0)])
    assert m1.content_hash() == m2.content_hash()
    m3 = Manifest(
        manifest_version="t", label_policy_version="t", source=_src(), records=[_rec(0), _rec(1, "simple_cyst")]
    )
    assert m1.content_hash() != m3.content_hash()
