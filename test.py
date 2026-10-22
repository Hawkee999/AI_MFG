"""
Smoke and data-leakage tests for the MILLING model.

Run from the project folder (the folder containing model.py):
    python test.py

It does NOT retrain the production models. Train first with:
    python model.py --operation milling

It checks:
1. model.py imports.
2. The saved milling bundle, its held-out evaluation and its metric consistency.
3. The held-out test rows are disjoint from the training rows and match the saved split.
4. SMOTE is applied only inside fit() (train data), never at prediction time.
5. Replay sampling returns only held-out rows.
6. Single-row and batch prediction return failure + failure-mode fields.
7. The failure-mode decision logic.
8. (Optional, if streamlit is installed) the replay, manual-prediction and evaluation views run end to end.
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import model
    import dataSampling
except Exception:
    print("[FAIL] Could not import model.py / dataSampling.py")
    traceback.print_exc()
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
MODEL_DIR = BASE_DIR / "models"
SAMPLE_ROW = {
    "Type": "M", "Air temperature [K]": 298.1, "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 120,
}
METRIC_KEYS = {
    "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision",
    "confusion_matrix", "per_class", "balanced_accuracy",
}

_cache = {}


def bundle():
    if "bundle" not in _cache:
        _cache["bundle"] = model.load_models(MODEL_DIR)["milling"]
    return _cache["bundle"]


def holdout():
    path = MODEL_DIR / "milling" / model.HOLDOUT_FILENAME
    assert path.is_file(), f"{path} missing; run: python model.py --operation milling"
    return pd.read_csv(path)


def check(name, fn):
    print(f"\n--- {name} ---")
    try:
        result = fn()
        print("[PASS]", result if result is not None else "OK")
        return True
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def test_saved_bundle():
    b = bundle()
    assert b["operation"] == "milling"
    assert "Machine failure" in b["models"], "Machine failure model missing"
    assert b["split"]["models_trained_on"].startswith("training split only")
    return {"targets": list(b["models"]), "split": b["split"]}


def test_metrics_complete_and_consistent():
    b = bundle()
    for target, m in b["metrics"].items():
        assert m["resampling"]["method"] in {"SMOTENC", "class_weight"}
        assert len(m["model_comparison"]) >= 3, f"{target}: expected several models"
        for name, e in m["model_comparison"].items():
            missing = METRIC_KEYS - set(e)
            assert not missing, f"{target}/{name} missing {missing}"
            cm = e["confusion_matrix"]
            total = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]
            assert total == m["test_rows"], f"{target}/{name}: confusion matrix does not cover the test rows"
            assert abs(e["accuracy"] - (cm["tn"] + cm["tp"]) / total) < 1e-9
            assert set(e["per_class"]) == {"0", "1"}
        if m["resampling"]["method"] == "SMOTENC":
            after = m["resampling"]["train_class_counts_after"]
            before = m["resampling"]["train_class_counts_before"]
            assert after["1"] > before["1"] and after["0"] == before["0"], "SMOTE should only add minority rows"
            # test class balance must be the natural one (not resampled)
            assert sum(m["test_class_counts"].values()) == m["test_rows"]
    return f"{len(b['metrics'])} targets, all metrics present and self-consistent"


def test_split_is_disjoint_and_matches_saved_holdout():
    df, X, Y, targets = model.prepare_milling_data(DATA_DIR)
    train_pos, test_pos = model.make_milling_split(df, X, targets)
    assert not set(train_pos) & set(test_pos), "train and test overlap"
    rows = df.groupby("__source_file__").cumcount()
    expected = set(zip(df["__source_file__"].iloc[test_pos], rows.iloc[test_pos]))
    saved = holdout()
    got = set(zip(saved["__source_file__"], saved["__source_row__"]))
    assert expected == got, "saved holdout file does not match the split model.py recomputes (retrain?)"
    train_keys = set(zip(df["__source_file__"].iloc[train_pos], rows.iloc[train_pos]))
    assert not (train_keys & got), "a held-out row is also a training row"
    assert len(saved) == bundle()["split"]["test_rows"]
    return f"{len(train_pos)} train / {len(test_pos)} test, disjoint, matches saved holdout"


def test_smote_only_during_fit():
    df, X, Y, targets = model.prepare_milling_data(DATA_DIR)
    train_pos, test_pos = model.make_milling_split(df, X, targets)
    y = Y["Machine failure"].astype(int)
    X_train, y_train = X.iloc[train_pos], y.iloc[train_pos]
    X_test = X.iloc[test_pos]
    estimator = model.RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=-1)
    pipe = model.build_milling_pipeline(estimator, X_train, True, 0.5, 5)
    pipe.fit(X_train, y_train)
    _, y_res = pipe[:-1].fit_resample(X_train, y_train)
    assert len(y_res) > len(y_train), "SMOTE did not add training rows"
    assert pipe.predict_proba(X_test).shape[0] == len(X_test), "prediction must not resample rows"
    return f"train {len(y_train)} -> {len(y_res)} rows after SMOTE; {len(X_test)} test rows untouched"


def test_replay_samples_only_holdout():
    saved = holdout()
    keys = set(zip(saved["__source_file__"], saved["__source_row__"]))
    sample = dataSampling.sample_milling(n=300, data_dir=DATA_DIR, model_dir=MODEL_DIR, random_state=1)
    got = set(zip(sample["__source_file__"], sample["__source_row__"]))
    assert got <= keys, "replay returned a row that is not in the held-out test set"
    assert {"Machine failure", "TWF"} <= set(sample.columns), "true labels should be included for actual-vs-predicted"
    return f"{len(got)} distinct sampled rows, all held-out"


def test_single_and_batch_prediction():
    allowed_modes = set(model.MILLING_MODE_TARGETS) | {"No failure"}
    single = model.predict_milling(SAMPLE_ROW, models=model.load_models(MODEL_DIR))
    first = single["predictions"][0]
    assert single["count"] == 1 and "Machine failure" in first["failures"]
    assert first["predicted_failure_mode"] in allowed_modes
    assert isinstance(first["predicted_failure"], bool)
    assert 0.0 <= first["failures"]["Machine failure"]["probability"] <= 1.0
    batch = model.predict_milling(holdout().head(3)[model.MILLING_FEATURES], models=model.load_models(MODEL_DIR))
    assert batch["count"] == 3 and len(batch["predictions"]) == 3
    return {"mode": first["predicted_failure_mode"], "failure": first["predicted_failure"]}


def test_failure_mode_logic():
    def f(**scores):
        return {k: {"probability": v} for k, v in scores.items()}

    none = model.derive_failure_mode(f(**{"Machine failure": 0.1, "TWF": 0.1, "HDF": 0.2}), 0.5)
    assert not none["predicted_failure"] and none["predicted_failure_mode"] == "No failure"
    hdf = model.derive_failure_mode(f(**{"Machine failure": 0.9, "TWF": 0.1, "HDF": 0.8, "PWF": 0.6}), 0.5)
    assert hdf["predicted_failure"] and hdf["predicted_failure_mode"] == "HDF" and hdf["flagged_modes"] == ["HDF", "PWF"]
    weak = model.derive_failure_mode(f(**{"Machine failure": 0.9, "TWF": 0.1, "HDF": 0.4}), 0.5)
    assert weak["predicted_failure_mode"] == "HDF" and weak["failure_mode_confident"] is False
    return "OK"


def test_streamlit_views():
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:
        return "skipped (streamlit not installed)"
    at = AppTest.from_file(str(BASE_DIR / "frontend.py"), default_timeout=90)
    at.session_state["auto_refresh"] = False  # otherwise the replay view sleeps and reruns forever

    at.run()  # default view: replay of a held-out row
    assert not at.exception, at.exception
    assert any("held-out test row" in m.value for m in at.success), "replay did not report a held-out test row"
    assert any("Actual (from the dataset)" in m.value for m in at.info), "replay should show the actual labels"

    at.session_state["milling_view"] = "Manual prediction"
    at.run()
    assert not at.exception, at.exception
    at.button[0].click().run()  # submit the form with its default (median) values
    assert not at.exception, at.exception
    verdicts = " ".join(m.value for m in list(at.error) + list(at.success))
    assert "failure" in verdicts.lower(), "manual prediction produced no failure verdict"

    at.session_state["milling_view"] = "Model evaluation"
    at.run()
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 2, "evaluation tables missing"
    return f"replay, manual prediction and evaluation views all ran; manual verdict: {verdicts[:70]}"


def main():
    print("=" * 65)
    print("MILLING MODEL SMOKE + LEAKAGE TESTS")
    print("=" * 65, flush=True)
    if not (MODEL_DIR / "milling" / "model.joblib").exists():
        print("[FAIL] No saved milling model. Train first: python model.py --operation milling")
        sys.exit(1)

    tests = [
        ("Saved milling bundle", test_saved_bundle),
        ("Evaluation metrics complete and consistent", test_metrics_complete_and_consistent),
        ("Train/test split disjoint and matches holdout file", test_split_is_disjoint_and_matches_saved_holdout),
        ("SMOTE applied to training data only", test_smote_only_during_fit),
        ("Replay samples only held-out rows", test_replay_samples_only_holdout),
        ("Single-row and batch prediction", test_single_and_batch_prediction),
        ("Failure-mode decision logic", test_failure_mode_logic),
        ("Streamlit replay / manual / evaluation views", test_streamlit_views),
    ]
    results = [(name, check(name, fn)) for name, fn in tests]

    print("\n" + "=" * 65)
    print("TEST SUMMARY")
    print("=" * 65)
    for name, passed in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    passed_count = sum(passed for _, passed in results)
    print(f"\nResult: {passed_count}/{len(results)} tests passed")
    if passed_count != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
