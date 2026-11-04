import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ovascope_ml.evaluation import metrics as M
from ovascope_ml.inference.render import mask_png
from ovascope_ml.models.geometry import Letterbox, dice_iou, expand_box, mask_bbox
from ovascope_ml.models.heads import LinearHead, fit_temperature
from ovascope_ml.providers.artifact_store import InvalidArtifactKey, LocalArtifactStore


@pytest.mark.parametrize("w,h", [(959, 537), (480, 523), (302, 794), (1135, 226)])
def test_letterbox_round_trip_and_overlay_alignment(w, h):
    lb = Letterbox(w, h, 448)
    for x, y in [(0, 0), (w / 2, h / 3), (w, h)]:
        mx, my = lb.to_model(x, y)
        sx, sy = lb.to_source(mx, my)
        assert sx == pytest.approx(x, abs=1e-6) and sy == pytest.approx(y, abs=1e-6)
    m = np.zeros((h, w), bool)
    m[h // 4 : h // 2, w // 3 : 2 * w // 3] = True
    back = lb.model_map_to_source(lb.apply_mask(m).astype(np.float32)) >= 0.5
    assert back.shape == (h, w)  # overlays come back at source resolution
    assert dice_iou(back, m)[0] > 0.95
    import io

    from PIL import Image

    assert Image.open(io.BytesIO(mask_png(m))).size == (w, h)


def test_box_helpers_and_dice():
    m = np.zeros((10, 10), bool)
    m[2:5, 3:8] = True
    assert mask_bbox(m) == (3, 2, 8, 5)
    assert mask_bbox(np.zeros((4, 4), bool)) is None
    assert expand_box((3, 2, 8, 5), 0.5, 10, 10) == (0, 0, 10, 7)
    assert dice_iou(m, m) == (1.0, 1.0)
    assert dice_iou(m, ~m)[0] == 0.0


def test_linear_head_matches_sklearn_and_stores_no_pickle(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 6))
    y = (x[:, 0] + x[:, 1] > 0).astype(int) + (x[:, 2] > 1).astype(int)
    sc = StandardScaler().fit(x)
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(x), y)
    head = LinearHead.from_sklearn(sc, lr, ["a", "b", "c"])
    np.testing.assert_allclose(head.scores(x), lr.predict_proba(sc.transform(x)), atol=1e-8)
    head.temperature = 2.5
    head.save(tmp_path / "h.npz")
    loaded = LinearHead.load(tmp_path / "h.npz")
    assert loaded.temperature == 2.5 and loaded.classes == ["a", "b", "c"]
    np.testing.assert_array_equal(loaded.scores(x).argmax(1), head.scores(x, temperature_scaled=False).argmax(1))
    with np.load(tmp_path / "h.npz", allow_pickle=False) as z:  # would raise if objects were pickled
        assert set(z.files) >= {"coef", "intercept", "mean", "scale"}


def test_temperature_fit_reduces_nll_and_preserves_argmax():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 400)
    # Confident logits for a predicted class that is right only ~70% of the time -> overconfident.
    pred = np.where(rng.random(400) < 0.7, y, rng.integers(0, 3, 400))
    logits = rng.normal(size=(400, 3)) * 0.3
    logits[np.arange(400), pred] += 6.0
    t = fit_temperature(logits, y)
    assert t > 1.5
    np.testing.assert_array_equal((logits / t).argmax(1), logits.argmax(1))
    from scipy.special import log_softmax

    nll = lambda z: -log_softmax(z, 1)[np.arange(400), y].mean()  # noqa: E731
    assert nll(logits / t) < nll(logits)


def test_binary_and_multiclass_metrics_are_consistent():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    s = np.array([0.1, 0.4, 0.8, 0.7, 0.3, 0.2, 0.9, 0.6])
    b = M.binary_metrics(y, s, threshold=0.5)
    assert b["counts"] == {"tp": 3, "fp": 1, "tn": 3, "fn": 1}
    assert b["sensitivity"] == 0.75 and b["ppv"] == 0.75 and b["prevalence"] == 0.5
    scores = np.stack([1 - s, s], 1)
    mc = M.multiclass_metrics(y, scores, ["neg", "pos"])
    assert mc["confusion_matrix"]["rows_true_cols_pred"] == [[3, 1], [1, 3]]
    assert 0 <= mc["calibration"]["top_label_ece_10bin"] <= 1


def test_bootstrap_and_paired_difference():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    s = np.stack([1 - (y * 0.6 + rng.random(200) * 0.4), y * 0.6 + rng.random(200) * 0.4], 1)
    ci = M.bootstrap_ci(M.balanced_accuracy, y, s, n=200, seed=0)
    assert ci["ci95"][0] <= ci["point"] <= ci["ci95"][1]
    d = M.paired_bootstrap_diff(M.balanced_accuracy, y, s, s, n=100, seed=0)
    assert d["difference"] == 0 and d["ci95"] == [0.0, 0.0]


def test_abstention_metrics_report_coverage():
    y = np.array([0, 1, 0, 1])
    s = np.array([[0.9, 0.1], [0.45, 0.55], [0.6, 0.4], [0.2, 0.8]])
    a = M.abstention_metrics(y, s, threshold=0.7)
    assert a["coverage"] == 0.5 and a["n_abstained"] == 2 and a["accuracy_retained"] == 1.0


def test_artifact_store_rejects_traversal_and_deletes_namespace(tmp_path):
    st = LocalArtifactStore(tmp_path / "store")
    k = st.put("abc123", b"data", "image/png")
    assert st.get(k) == b"data"
    for bad in ("../etc/passwd", "abc123/../../x", "abc123/" + "0" * 31 + "g", "a/b/c", "..\\x"):
        with pytest.raises(InvalidArtifactKey):
            st.get(bad)
    with pytest.raises(InvalidArtifactKey):
        st.put("../evil", b"x", "image/png")
    assert st.delete_namespace("abc123") == 1
    assert not st.exists(k)
