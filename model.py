from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Optional high-performance libraries. The script still runs if either is absent.
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = CatBoostRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None

MILLING_FAILURE_TARGETS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
MILLING_FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",]
DRILLING_AXES = ["X", "Y", "Z"]

LOG = logging.getLogger("warehouse_model")

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def normalize_name(value: Any) -> str:
    """Normalize headers to make matching tolerant of case and whitespace."""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()

def findData(folder: Path) -> list[Path]:
    if not folder.exists():
        return []

    supported_extensions = {".csv", ".txt", ".xlsx", ".xls"}
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in supported_extensions
        and not p.name.startswith("~$")
    )

def read_table(path: Path) -> pd.DataFrame:

    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, header=4)
        frame.columns = [str(c).strip() for c in frame.columns]

        axis_cols = [
            c for c in frame.columns
            if str(c).strip().upper() in {"X", "Y", "Z"}
        ]
        if axis_cols:
            frame[axis_cols] = frame[axis_cols].apply(
                pd.to_numeric, errors="coerce"
            )
            frame = frame.dropna(subset=axis_cols, how="all")

        return frame.reset_index(drop=True)

    errors = []
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        for sep in (",", ";", "\t", None):
            try:
                frame = pd.read_csv(
                    path,
                    encoding=encoding,
                    sep=sep,
                    engine="python",
                )
                if frame.shape[1] > 1 and len(frame) > 0:
                    frame.columns = [str(c).strip() for c in frame.columns]
                    return frame
            except Exception as exc:
                errors.append(str(exc))

    raise ValueError(
        f"Could not read {path}. "
        f"Last error: {errors[-1] if errors else 'unknown'}"
    )

def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {normalize_name(c): c for c in df.columns}
    for candidate in candidates:
        if normalize_name(candidate) in lookup:
            return lookup[normalize_name(candidate)]
    return None

def numeric_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    result = []
    for col in df.columns:
        if col in exclude:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= max(3, int(0.5 * len(df))):
            result.append(col)
    return result

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

    num_cols = [
        c for c in X.columns
        if pd.api.types.is_numeric_dtype(X[c])
        and not pd.api.types.is_bool_dtype(X[c])
    ]
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols)
        )
    if cat_cols:
        transformers.append(
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns found.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def save_bundle(model_dir: Path, operation: str, bundle: dict) -> Path:
    out_dir = model_dir / operation
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.joblib"
    joblib.dump(bundle, out_path)
    LOG.info("Saved %s model bundle: %s", operation, out_path)
    return out_path


def combine_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            df = read_table(path)
            df["__source_file__"] = path.name
            frames.append(df)
            LOG.info("Loaded %s: %s rows × %s columns", path, len(df), len(df.columns))
        except Exception as exc:
            LOG.warning("Skipping %s: %s", path, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)

def train_milling(data_dir: Path, model_dir: Path) -> dict:
    paths = findData(data_dir / "milling")
    if not paths:
        LOG.warning("Milling: no CSV found in %s; skipping.", data_dir / "milling")
        return {"status": "skipped", "reason": "No milling CSV found"}

    df = combine_csvs(paths)
    if df.empty:
        return {"status": "skipped", "reason": "Milling files could not be read"}

    # Find columns case/whitespace-insensitively, then rename to expected names.
    rename_map = {}
    for expected in MILLING_FEATURES + MILLING_FAILURE_TARGETS:
        actual = find_column(df, [expected])
        if actual is not None:
            rename_map[actual] = expected
    df = df.rename(columns=rename_map)

    missing_features = [c for c in MILLING_FEATURES if c not in df.columns]
    if missing_features:
        LOG.warning("Milling: missing required features %s; skipping.", missing_features)
        LOG.warning("Milling columns found: %s", list(df.columns))
        return {"status": "skipped", "reason": f"Missing features: {missing_features}"}

    # Avoid identifiers, source filenames and all output labels as input features.
    X = df[MILLING_FEATURES].copy()
    X["Type"] = X["Type"].astype("string").fillna("Unknown").astype(str)
    for col in MILLING_FEATURES:
        if col != "Type":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    trained = {}
    metrics = {}
    for target in MILLING_FAILURE_TARGETS:
        if target not in df.columns:
            LOG.warning("Milling target '%s' missing; it will not be trained.", target)
            continue

        y = pd.to_numeric(df[target], errors="coerce")
        valid = y.isin([0, 1]) & X.notna().any(axis=1)
        X_t = X.loc[valid].copy()
        y_t = y.loc[valid].astype(int)
        counts = y_t.value_counts().to_dict()

        # Do not create fake probabilities from a single-class dataset.
        if len(counts) < 2:
            LOG.warning(
                "Milling target '%s' skipped: needs both 0 and 1 labels; observed %s.",
                target, counts
            )
            continue
        if min(counts.values()) < 3 or len(y_t) < 20:
            LOG.warning(
                "Milling target '%s' skipped: only %s rows / class counts %s. "
                "Use the full labeled dataset for a more reliable classifier.",
                target, len(y_t), counts
            )
            continue

        # Holdout split only when both classes can be represented in the test set.
        test_count = max(2, int(round(len(y_t) * 0.2)))
        stratify = y_t if min(counts.values()) >= 2 and test_count >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_t, y_t, test_size=0.2, random_state=42, stratify=stratify
        )
        # Compare several model families on the same holdout. Average precision
        # is the selection metric because positive failure labels are very rare.
        candidates = {
            "RandomForest": RandomForestClassifier(
                n_estimators=400, class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1,
            ),
            "ExtraTrees": __import__("sklearn.ensemble", fromlist=["ExtraTreesClassifier"]).ExtraTreesClassifier(
                n_estimators=400, class_weight="balanced", min_samples_leaf=2,
                random_state=42, n_jobs=-1,
            ),
        }
        if CatBoostClassifier is not None:
            candidates["CatBoost"] = CatBoostClassifier(
                iterations=500, depth=6, learning_rate=0.05,
                loss_function="Logloss", auto_class_weights="Balanced",
                verbose=False, random_seed=42, allow_writing_files=False,
            )
        if XGBClassifier is not None:
            positives = max(1, int((y_train == 1).sum()))
            negatives = max(1, int((y_train == 0).sum()))
            candidates["XGBoost"] = XGBClassifier(
                n_estimators=450, max_depth=4, learning_rate=0.04,
                subsample=0.9, colsample_bytree=0.9,
                scale_pos_weight=negatives / positives,
                eval_metric="logloss", random_state=42, n_jobs=-1,
            )

        comparison = {}
        fitted = {}
        for name, estimator in candidates.items():
            candidate = Pipeline([
                ("preprocess", make_preprocessor(X_train)),
                ("classifier", estimator),
            ])
            try:
                candidate.fit(X_train, y_train)
                probabilities = candidate.predict_proba(X_test)[:, 1]
                predictions = (probabilities >= 0.5).astype(int)
                ap = float(average_precision_score(y_test, probabilities))
                comparison[name] = {
                    "average_precision": ap,
                    "roc_auc": float(roc_auc_score(y_test, probabilities)),
                    "f1_at_0_5": float(f1_score(y_test, predictions, zero_division=0)),
                    "precision_at_0_5": float(precision_score(y_test, predictions, zero_division=0)),
                    "recall_at_0_5": float(recall_score(y_test, predictions, zero_division=0)),
                    "accuracy_at_0_5": float(accuracy_score(y_test, predictions)),
                }
                fitted[name] = candidate
                LOG.info("Milling %s / %s comparison: %s", target, name, comparison[name])
            except Exception as exc:
                LOG.warning("Milling candidate %s failed for %s: %s", name, target, exc)

        if not fitted:
            LOG.warning("All milling candidate models failed for target %s", target)
            continue
        best_name = max(comparison, key=lambda name: comparison[name]["average_precision"])
        # Refit the selected model on every labelled row after holdout evaluation,
        # so the saved production model does not discard the holdout rows.
        pipeline = fitted[best_name]
        pipeline.fit(X_t, y_t)
        selected_metrics = comparison[best_name]
        metric = {
            "rows": int(len(y_t)),
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "selected_model": best_name,
            "selection_metric": "average_precision (PR-AUC approximation)",
            "best_model_metrics": selected_metrics,
            "model_comparison": comparison,
            "warning": "Holdout metrics are used for model selection; use a separate untouched test set for final unbiased evaluation.",
        }
        trained[target] = pipeline
        metrics[target] = metric
        LOG.info("Milling target %s metrics: %s", target, metric)

    bundle = {
        "operation": "milling",
        "model_type": "one binary classifier per available failure label",
        "feature_columns": MILLING_FEATURES,
        "target_columns": list(trained.keys()),
        "models": trained,
        "metrics": metrics,
        "note": "Probability is model-estimated, not a guarantee of failure.",
    }
    if not trained:
        LOG.warning("Milling: no target had enough positive and negative labels to train.")
        return {
            "status": "skipped",
            "reason": "No eligible failure target. Need full data with both 0 and 1 for at least one label.",
            "rows": len(df),
        }
    save_bundle(model_dir, "milling", bundle)
    return {"status": "trained", "rows": len(df), "targets": list(trained), "metrics": metrics}


# ----------------------------------------------------------------------
# TURNING: regress flank wear Vb
# ----------------------------------------------------------------------
def train_turning(data_dir: Path, model_dir: Path) -> dict:
    paths = findData(data_dir / "turning")
    if not paths:
        LOG.warning("Turning: no CSV found in %s; skipping.", data_dir / "turning")
        return {"status": "skipped", "reason": "No turning CSV found"}

    df = combine_csvs(paths)
    if df.empty:
        return {"status": "skipped", "reason": "Turning files could not be read"}

    vb_col = find_column(df, ["Vb", "VB", "flank wear", "flank_wear", "wear"])
    if vb_col is None:
        LOG.warning("Turning: no Vb/flank-wear target column found; columns=%s", list(df.columns))
        return {"status": "skipped", "reason": "No Vb/flank-wear target column found"}

    y = pd.to_numeric(df[vb_col], errors="coerce")
    # Exclude target, tool ID, UDI/product IDs, and file metadata to reduce leakage.
    excluded = {vb_col, "__source_file__"}
    for col in df.columns:
        norm = normalize_name(col)
        if norm in {"tool", "tool id", "tool_id", "product id", "product_id", "udi", "run", "sample no", "sample no."}:
            excluded.add(col)

    feature_frame = df.drop(columns=[c for c in excluded if c in df.columns], errors="ignore")
    features = numeric_columns(feature_frame)
    if len(features) < 2:
        return {"status": "skipped", "reason": f"Not enough numeric sensor features; found {features}"}

    X = feature_frame[features].apply(pd.to_numeric, errors="coerce")
    valid = y.notna() & X.notna().any(axis=1)
    X, y = X.loc[valid], y.loc[valid]
    if len(X) < 30 or y.nunique() < 3:
        LOG.warning(
            "Turning: only %s usable rows / %s unique Vb values. Need more labeled rows to train a useful regressor.",
            len(X), y.nunique()
        )
        return {"status": "skipped", "reason": "Too few labeled Vb rows (need at least 30 and 3 distinct values)", "rows": len(X)}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Compare multiple tree/boosting regressors on the same holdout.
    candidates = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=42
        ),
    }
    if CatBoostRegressor is not None:
        candidates["CatBoost"] = CatBoostRegressor(
            iterations=700, depth=7, learning_rate=0.04,
            loss_function="RMSE", verbose=False, random_seed=42,
            allow_writing_files=False,
        )
    if XGBRegressor is not None:
        candidates["XGBoost"] = XGBRegressor(
            n_estimators=600, max_depth=5, learning_rate=0.04,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=2.0, objective="reg:squarederror",
            random_state=42, n_jobs=-1,
        )

    comparison = {}
    fitted = {}
    for name, estimator in candidates.items():
        candidate = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", estimator),
        ])
        try:
            candidate.fit(X_train, y_train)
            pred_candidate = candidate.predict(X_test)
            comparison[name] = {
                "mae": float(mean_absolute_error(y_test, pred_candidate)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred_candidate))),
                "r2": float(r2_score(y_test, pred_candidate)) if len(y_test) >= 2 else None,
            }
            fitted[name] = candidate
            LOG.info("Turning %s comparison: %s", name, comparison[name])
        except Exception as exc:
            LOG.warning("Turning candidate %s failed: %s", name, exc)

    if not fitted:
        return {"status": "skipped", "reason": "All turning candidate models failed"}
    best_name = min(comparison, key=lambda name: comparison[name]["rmse"])
    model = fitted[best_name]
    pred = model.predict(X_test)
    metrics = {
        "rows": int(len(X)),
        "features": int(len(features)),
        "selected_model": best_name,
        "selection_metric": "RMSE (lower is better)",
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)) if len(y_test) >= 2 else None,
        "model_comparison": comparison,
        "warning": "Random row split can overstate performance when nearby samples/runs are correlated; validate by tool/run before trusting this score.",
    }
    # Refit the chosen pipeline on all labelled observations for production inference.
    model.fit(X, y)
    LOG.info("Turning Vb selected model metrics: %s", metrics)
    bundle = {
        "operation": "turning",
        "model_type": "flank wear Vb regression",
        "feature_columns": features,
        "target_column": "Vb",
        "model": model,
        "metrics": metrics,
        "note": "Vb is predicted flank wear. Tool-condition classes need documented wear thresholds.",
    }
    save_bundle(model_dir, "turning", bundle)
    return {"status": "trained", "metrics": metrics, "features": features}


# ----------------------------------------------------------------------
# DRILLING: raw X/Y/Z signal -> window features -> anomaly detector
# ----------------------------------------------------------------------
def find_axis_columns(df: pd.DataFrame) -> dict[str, str] | None:
    found = {}
    for axis in DRILLING_AXES:
        candidates = [axis, axis.lower(), f"{axis}_acceleration", f"acceleration_{axis}",
                      f"{axis} axis", f"{axis}_axis", f"vibration_{axis}"]
        col = find_column(df, candidates)
        if col is None:
            return None
        found[axis] = col
    return found


def window_features(df: pd.DataFrame, axis_cols: dict[str, str], window_size: int) -> pd.DataFrame:
    """Summarize consecutive raw samples into interpretable windows."""
    rows = []
    for start in range(0, len(df) - window_size + 1, window_size):
        part = df.iloc[start:start + window_size]
        row = {"window_start_sample": start, "window_end_sample": start + window_size - 1}
        for axis, col in axis_cols.items():
            values = pd.to_numeric(part[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            row[f"{axis}_mean"] = float(np.mean(values))
            row[f"{axis}_std"] = float(np.std(values))
            row[f"{axis}_rms"] = float(np.sqrt(np.mean(values ** 2)))
            row[f"{axis}_max_abs"] = float(np.max(np.abs(values)))
            row[f"{axis}_peak_to_peak"] = float(np.ptp(values))
            row[f"{axis}_skew"] = float(pd.Series(values).skew()) if len(values) > 2 else 0.0
            row[f"{axis}_kurtosis"] = float(pd.Series(values).kurt()) if len(values) > 3 else 0.0
        if any(k.endswith("_rms") for k in row):
            rows.append(row)
    return pd.DataFrame(rows)


def train_drilling(data_dir: Path, model_dir: Path, window_size: int = 1000) -> dict:
    paths = findData(data_dir / "drilling")
    if not paths:
        LOG.warning("Drilling: no CSV found in %s; skipping.", data_dir / "drilling")
        return {"status": "skipped", "reason": "No drilling CSV found"}

    raw_frames = []
    for path in paths:
        try:
            df = read_table(path)
            axes = find_axis_columns(df)
            if axes is None:
                LOG.warning("Drilling: %s has no recognizable X/Y/Z columns; skipping.", path)
                LOG.warning("Columns found: %s", list(df.columns))
                continue
            df["__source_file__"] = path.name
            raw_frames.append((path, df, axes))
        except Exception as exc:
            LOG.warning("Skipping drilling file %s: %s", path, exc)

    if not raw_frames:
        return {"status": "skipped", "reason": "No drilling files with X/Y/Z columns found"}

    feature_frames = []
    for path, df, axes in raw_frames:
        feats = window_features(df, axes, window_size)
        if len(feats):
            feats["source_file"] = path.name
            feature_frames.append(feats)

    if not feature_frames:
        return {
            "status": "skipped",
            "reason": f"No complete windows. Need at least {window_size} raw samples per file.",
        }

    all_features = pd.concat(feature_frames, ignore_index=True)
    feature_cols = [
        c for c in all_features.columns
        if c not in {"window_start_sample", "window_end_sample", "source_file"}
    ]
    X = all_features[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=0, how="all")
    if len(X) < 5:
        return {
            "status": "skipped",
            "reason": f"Only {len(X)} usable windows. Collect more samples; aim for at least 20-50 windows across runs.",
            "windows": len(X),
        }

    # Anomaly detector, not a supervised wear predictor. If data has mixed wear states,
    # normal-only baseline data should be selected for fitting for best interpretation.
    detector = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("detector", IsolationForest(
            n_estimators=300, contamination="auto", random_state=42
        )),
    ])
    detector.fit(X)
    # sklearn decision_function: larger means more normal. Negate to make larger = more anomalous.
    anomaly_scores = -detector.decision_function(X)
    predictions = detector.predict(X)  # -1 anomaly, +1 inlier
    metrics = {
        "windows": int(len(X)),
        "flagged_anomalies_in_training_data": int(np.sum(predictions == -1)),
        "mean_anomaly_score": float(np.mean(anomaly_scores)),
    }
    bundle = {
        "operation": "drilling",
        "model_type": "unsupervised vibration anomaly detector",
        "feature_columns": feature_cols,
        "axis_columns": DRILLING_AXES,
        "window_size_samples": int(window_size),
        "sampling_rate_hz": 1000,
        "model": detector,
        "metrics": metrics,
        "note": "Anomaly score is not a tool-wear percentage. Supervised wear prediction requires labeled wear measurements.",
    }
    save_bundle(model_dir, "drilling", bundle)
    LOG.info("Drilling anomaly model summary: %s", metrics)
    return {"status": "trained", "metrics": metrics, "feature_columns": feature_cols}



# ----------------------------------------------------------------------
# INFERENCE API: import these functions directly from frontend.py
# ----------------------------------------------------------------------
def load_models(model_dir: str | Path = "models", strict: bool = False) -> dict[str, dict]:
    """Load trained bundles once at frontend startup. Missing models are skipped unless strict=True."""
    root = Path(model_dir)
    bundles: dict[str, dict] = {}
    for operation in ("milling", "turning", "drilling"):
        path = root / operation / "model.joblib"
        if not path.exists():
            if strict:
                raise FileNotFoundError(f"Trained {operation} model not found at {path}. Run model.py first.")
            continue
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or bundle.get("operation") != operation:
            raise ValueError(f"Invalid model bundle for {operation}: {path}")
        bundles[operation] = bundle
    if strict and len(bundles) != 3:
        missing = sorted(set(("milling", "turning", "drilling")) - set(bundles))
        raise FileNotFoundError(f"Missing trained model bundles: {missing}. Run model.py first.")
    return bundles


def get_model_info(
    models: dict[str, dict] | None = None,
    model_dir: str | Path = "models",
) -> dict:
    """Return model metadata and required input columns for frontend form generation."""
    bundles = models if models is not None else load_models(model_dir)
    info = {}
    for operation, bundle in bundles.items():
        if operation == "milling":
            info[operation] = {
                "feature_columns": list(bundle.get("feature_columns", [])),
                "targets": {
                    target: {
                        "selected_model": bundle.get("metrics", {}).get(target, {}).get("selected_model", "unknown"),
                        "class_counts": bundle.get("metrics", {}).get(target, {}).get("class_counts", {}),
                    }
                    for target in bundle.get("target_columns", [])
                },
            }
        elif operation == "turning":
            info[operation] = {
                "feature_columns": list(bundle.get("feature_columns", [])),
                "target": bundle.get("target_column", "Vb"),
                "selected_model": bundle.get("metrics", {}).get("selected_model", "unknown"),
            }
        elif operation == "drilling":
            info[operation] = {
                "required_axis_columns": ["X", "Y", "Z"],
                "window_size_samples": int(bundle.get("window_size_samples", 1000)),
                "sampling_rate_hz": int(bundle.get("sampling_rate_hz", 1000)),
                "model_type": bundle.get("model_type", "unsupervised vibration anomaly detector"),
            }
    return info


def _as_frame(data: dict | list[dict] | pd.DataFrame, expected: list[str], operation: str) -> pd.DataFrame:
    """Convert frontend input to a frame and align it to the saved training schema."""
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, dict):
        frame = pd.DataFrame([data])
    elif isinstance(data, list) and all(isinstance(row, dict) for row in data):
        frame = pd.DataFrame(data)
    else:
        raise TypeError(f"{operation} input must be a dict, list of dicts, or pandas DataFrame")

    # Match incoming headers case-insensitively while preserving saved feature names.
    lookup = {normalize_name(col): col for col in frame.columns}
    rename = {}
    for col in expected:
        actual = lookup.get(normalize_name(col))
        if actual is not None and actual != col:
            rename[actual] = col
    frame = frame.rename(columns=rename)
    missing = [col for col in expected if col not in frame.columns]
    for col in missing:
        frame[col] = np.nan
    frame = frame[expected].copy()
    return frame


def predict_milling(
    data: dict | list[dict] | pd.DataFrame,
    models: dict[str, dict] | None = None,
    model_dir: str | Path = "models",
    threshold: float = 0.5,
) -> dict:
    """Predict milling failure probabilities. Input keys are MILLING_FEATURES.

    Example: predict_milling({"Type":"M", "Air temperature [K]":298.1,
      "Process temperature [K]":308.6, "Rotational speed [rpm]":1500,
      "Torque [Nm]":40, "Tool wear [min]":120})
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    bundles = models if models is not None else load_models(model_dir)
    if "milling" not in bundles:
        raise FileNotFoundError("Milling model bundle is missing. Train models first with python model.py")
    bundle = bundles["milling"]
    frame = _as_frame(data, list(bundle["feature_columns"]), "milling")
    if "Type" in frame.columns:
        frame["Type"] = frame["Type"].astype("string").fillna("Unknown").astype(str)
    for col in frame.columns:
        if col != "Type":
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    results = []
    for row_idx in range(len(frame)):
        failures = {}
        for target, estimator in bundle["models"].items():
            probability = float(estimator.predict_proba(frame.iloc[[row_idx]])[0, 1])
            failures[target] = {
                "probability": probability,
                "risk_percent": round(probability * 100.0, 2),
                "predicted_failure": bool(probability >= threshold),
                "threshold": float(threshold),
                "selected_model": bundle.get("metrics", {}).get(target, {}).get("selected_model", "unknown"),
            }
        results.append({"failures": failures, "any_failure_predicted": any(v["predicted_failure"] for v in failures.values())})
    return {"operation": "milling", "count": len(results), "predictions": results}


def predict_turning(
    data: dict | list[dict] | pd.DataFrame,
    models: dict[str, dict] | None = None,
    model_dir: str | Path = "models",
) -> dict:
    """Predict flank wear Vb from the feature columns used during training."""
    bundles = models if models is not None else load_models(model_dir)
    if "turning" not in bundles:
        raise FileNotFoundError("Turning model bundle is missing. Train models first with python model.py")
    bundle = bundles["turning"]
    frame = _as_frame(data, list(bundle["feature_columns"]), "turning")
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    values = bundle["model"].predict(frame)
    return {
        "operation": "turning",
        "count": int(len(values)),
        "target": "Vb",
        "unit": "same unit as the Vb training labels",
        "selected_model": bundle.get("metrics", {}).get("selected_model", "unknown"),
        "predictions": [{"predicted_vb": float(v)} for v in values],
        "warning": "Predictions are only valid for feature inputs and operating conditions represented by the training data.",
    }


def predict_drilling(
    raw_data: pd.DataFrame | list[dict] | dict,
    models: dict[str, dict] | None = None,
    model_dir: str | Path = "models",
) -> dict:
    """Score raw drilling vibration samples (X/Y/Z) in saved-size windows.

    Requires at least one complete window, normally 1000 rows = 1 second at 1 kHz.
    Returns anomaly scores/flags, not a wear percentage.
    """
    bundles = models if models is not None else load_models(model_dir)
    if "drilling" not in bundles:
        raise FileNotFoundError("Drilling model bundle is missing. Train models first with python model.py")
    bundle = bundles["drilling"]
    if isinstance(raw_data, pd.DataFrame):
        frame = raw_data.copy()
    elif isinstance(raw_data, dict):
        frame = pd.DataFrame([raw_data])
    elif isinstance(raw_data, list) and all(isinstance(row, dict) for row in raw_data):
        frame = pd.DataFrame(raw_data)
    else:
        raise TypeError("drilling input must be a DataFrame, dict, or list of sample dictionaries")
    frame.columns = [str(c).strip() for c in frame.columns]
    axes = find_axis_columns(frame)
    if axes is None:
        raise ValueError("Drilling input must contain X, Y, and Z columns (or recognized axis aliases).")
    for axis, col in axes.items():
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=list(axes.values()), how="all").reset_index(drop=True)
    window_size = int(bundle.get("window_size_samples", 1000))
    feats = window_features(frame, axes, window_size)
    if feats.empty:
        raise ValueError(f"Need at least {window_size} usable raw samples for one complete drilling window; received {len(frame)}.")
    feature_cols = list(bundle["feature_columns"])
    X = feats.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    detector = bundle["model"]
    scores = -detector.decision_function(X)  # larger = more anomalous
    flags = detector.predict(X) == -1
    predictions = []
    for i, (score, flagged) in enumerate(zip(scores, flags)):
        predictions.append({
            "window_index": int(i),
            "start_sample": int(i * window_size),
            "end_sample": int((i + 1) * window_size - 1),
            "anomaly_score": float(score),
            "is_anomaly": bool(flagged),
        })
    return {
        "operation": "drilling",
        "count": len(predictions),
        "window_size_samples": window_size,
        "sampling_rate_hz": int(bundle.get("sampling_rate_hz", 1000)),
        "predictions": predictions,
        "warning": "Anomaly detection identifies unusual vibration patterns; it does not estimate tool wear unless labelled wear data is used to train a supervised model.",
    }


def predict_tool_condition(
    operation: str,
    data: Any,
    models: dict[str, dict] | None = None,
    model_dir: str | Path = "models",
    threshold: float = 0.5,
) -> dict:
    """Single dispatch function suitable for a frontend or API endpoint."""
    op = normalize_name(operation)
    bundles = models if models is not None else load_models(model_dir)
    if op == "milling":
        return predict_milling(data, models=bundles, threshold=threshold)
    if op == "turning":
        return predict_turning(data, models=bundles)
    if op == "drilling":
        return predict_drilling(data, models=bundles)
    raise ValueError("operation must be one of: milling, turning, drilling")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train milling, turning and drilling tool models.")
    parser.add_argument("--data-dir", default="data", help="Folder containing milling/turning/drilling subfolders")
    parser.add_argument("--model-dir", default="models", help="Output folder for trained model bundles")
    parser.add_argument("--drilling-window", type=int, default=1000,
                        help="Raw drilling samples per feature window (default 1000 at 1000 Hz = 1 second)")
    args = parser.parse_args()

    configure_logging()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("=" * 70)
    LOG.info("Starting combined tool-wear model training")
    LOG.info("Data directory: %s", data_dir.resolve())
    LOG.info("Model directory: %s", model_dir.resolve())
    LOG.info("=" * 70)

    results = {
        "optional_dependencies": {
            "catboost_available": CatBoostClassifier is not None,
            "xgboost_available": XGBClassifier is not None,
            "install_hint": "Install optional candidates with: pip install catboost xgboost",
        },
        "milling": train_milling(data_dir, model_dir),
        "turning": train_turning(data_dir, model_dir),
        "drilling": train_drilling(data_dir, model_dir, window_size=args.drilling_window),
    }

    report_path = model_dir / "training_report.json"
    report_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    LOG.info("=" * 70)
    LOG.info("TRAINING SUMMARY")
    for operation, result in results.items():
        LOG.info("%s: %s", operation.upper(), result.get("status", "unknown"))
        if result.get("reason"):
            LOG.info("  Reason: %s", result["reason"])
        if result.get("metrics"):
            LOG.info("  Metrics: %s", result["metrics"])
    LOG.info("Training report saved to %s", report_path.resolve())
    LOG.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
