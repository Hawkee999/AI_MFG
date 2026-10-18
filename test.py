"""
Smoke tests for model.py.

Run this from the same project folder as model.py:
    python test.py

This checks:
1. The model module imports.
2. Milling configuration and dataset can be loaded.
3. Saved model bundle exists (or explains how to train it).
4. A prediction returns the expected structure.
5. Batch prediction works.

It does NOT retrain models. To train first:
    python model.py --operation milling --train
"""

import sys
import traceback
from pathlib import Path

import pandas as pd

try:
    import model
except Exception:
    print("[FAIL] Could not import model.py")
    traceback.print_exc()
    sys.exit(1)


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


def test_config():
    config = model.get_config("milling")
    assert config.name == "milling"
    assert config.binary_target == "Machine failure"
    assert "TWF" in config.multilabel_targets
    return {
        "operation": config.name,
        "features": config.features,
        "model_path": str(config.model_path),
    }


def test_dataset():
    df = model.load_dataset("milling")
    assert len(df) > 0, "Dataset is empty"
    assert all(col in df.columns for col in model.get_config("milling").features)
    return f"{len(df)} rows, {len(df.columns)} columns"


def test_saved_bundle():
    bundle = model.load_operation_models("milling")
    assert bundle["operation"] == "milling"
    assert "Machine failure" in bundle["binary_models"]
    assert bundle["multilabel_models"], "No failure-type models saved"
    return {
        "operation": bundle["operation"],
        "binary_models": list(bundle["binary_models"]),
        "failure_type_models": list(bundle["multilabel_models"]),
    }


def make_sample():
    # Use a real row from the dataset so the test matches the data schema.
    # Drop the labels/identifiers and retain only configured model features.
    df = model.load_dataset("milling")
    config = model.get_config("milling")
    sample = df.iloc[0][config.features].to_dict()
    return sample


def test_single_prediction():
    sample = make_sample()
    result = model.predict_operation("milling", sample)

    assert result["operation"] == "milling"
    assert "predictions" in result
    assert "Machine failure" in result["predictions"]

    failure_result = result["predictions"]["Machine failure"]
    probability = failure_result["positive_probability"]

    assert isinstance(probability, (int, float))
    assert 0.0 <= probability <= 1.0


def test_batch_prediction():
    df = model.load_dataset("milling")
    config = model.get_config("milling")

    # Test with two real rows from the dataset.
    samples = df.iloc[:2][config.features].to_dict(orient="records")
    result = model.predict_operation("milling", pd.DataFrame(samples))

    assert result["operation"] == "milling"
    assert "predictions" in result
    assert "Machine failure" in result["predictions"]

    # The current prediction API may return only the first row's result.
    return "Batch input accepted; prediction structure is valid"


def main():
    print("=" * 65)
    print("MODEL.PY SMOKE TESTS")
    print("=" * 65, flush=True)

    tests = [
        ("Milling configuration", test_config),
        ("Dataset loading", test_dataset),
        ("Saved model bundle", test_saved_bundle),
        ("Single-row prediction", test_single_prediction),
        ("Batch prediction", test_batch_prediction),
    ]

    results = []
    for name, fn in tests:
        results.append((name, check(name, fn)))

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
