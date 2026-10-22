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
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# SMOTE lives in imbalanced-learn. Its Pipeline applies resampling only during fit(),
# so validation/test rows and prediction-time rows are never resampled.
try:
    from imblearn.over_sampling import SMOTE, SMOTENC
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    SMOTE = SMOTENC = None
    from sklearn.pipeline import Pipeline as ImbPipeline  # only used when resampling is disabled

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
MILLING_MODE_TARGETS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
MILLING_FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",]
DRILLING_AXES = ["X", "Y", "Z"]

TEST_SIZE = 0.2             # held-out fraction (never resampled, never trained on)
RANDOM_STATE = 42
CV_FOLDS = 5                # inner CV on the training split, used only for model selection
DEFAULT_THRESHOLD = 0.5
DEFAULT_SMOTE_RATIO = 0.5   # minority:majority ratio after SMOTE (1.0 = fully balanced)
HOLDOUT_FILENAME = "holdout_test.csv"

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
    if not folder.exists() and folder.parent.is_dir():
        # Tolerate folder-name capitalization (Data/Milling vs data/milling) on case-sensitive systems.
        for child in folder.parent.iterdir():
            if child.is_dir() and child.name.casefold() == folder.name.casefold():
                folder = child
                break
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

def split_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols = [
        c for c in X.columns
        if pd.api.types.is_numeric_dtype(X[c])
        and not pd.api.types.is_bool_dtype(X[c])
    ]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def make_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool = False,
    ordinal_categoricals: bool = False,
) -> ColumnTransformer:
    """Numeric columns first, then categorical columns (SMOTENC relies on this order).

    ordinal_categoricals=True encodes each categorical column as ONE integer column so SMOTENC can
    synthesise valid categories (one-hot columns could be recombined into an invalid all-zero code).
    The categories are taken from the training frame passed in here.
    """
    num_cols, cat_cols = split_feature_columns(X)
    transformers = []
    if num_cols:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:  # keeps SMOTE nearest-neighbour distances from being dominated by rpm
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), num_cols))
    if cat_cols:
        if ordinal_categoricals:
            encoder = OrdinalEncoder(
                categories=[sorted(X[c].dropna().astype(str).unique()) for c in cat_cols],
                handle_unknown="use_encoded_value", unknown_value=-1,
            )
        else:
            encoder = OneHotEncoder(handle_unknown="ignore")
        transformers.append(
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", encoder),
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

# ----------------------------------------------------------------------
# MILLING: failure / failure-mode classifiers
#
# Leakage-safe design:
#   * ONE stratified train/test split is shared by every target, so a row that is
#     "test" for one failure label can never be "train" for another.
#   * SMOTE(NC) lives inside the pipeline: it runs only when .fit() is called on
#     training data (including the inner CV folds) and never on validation/test rows.
#   * Model selection uses cross-validation on the TRAINING split only. The test
#     split is used once, for reporting.
#   * The saved model is the one that was evaluated (trained on the training split
#     only). It is NOT refit on the test rows, so the held-out rows stay unseen and
#     can be safely replayed in the Streamlit simulation.
# ----------------------------------------------------------------------
def make_milling_split(df: pd.DataFrame, X: pd.DataFrame, target_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Shared stratified split. Strata = the exact combination of failure labels."""
    usable = np.flatnonzero(X.notna().any(axis=1).to_numpy())
    labels = (
        df[target_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).iloc[usable]
    )
    codes = pd.Series(labels.to_numpy() @ (2 ** np.arange(len(target_cols))), index=labels.index)
    # Label combinations seen fewer than 10 times are pooled so stratification is possible.
    strata = codes.where(codes.map(codes.value_counts()) >= 10, -1)
    try:
        train_pos, test_pos = train_test_split(
            usable, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strata.to_numpy()
        )
    except ValueError as exc:
        LOG.warning("Milling: stratified split failed (%s); using an unstratified split.", exc)
        train_pos, test_pos = train_test_split(usable, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return np.sort(train_pos), np.sort(test_pos)


def build_milling_pipeline(estimator, X_train: pd.DataFrame, use_smote: bool,
                           smote_ratio: float, k_neighbors: int) -> ImbPipeline:
    """preprocess -> (SMOTENC) -> classifier. Resampling is skipped at predict time."""
    steps: list[tuple[str, Any]] = [
        ("preprocess", make_preprocessor(X_train, scale_numeric=True, ordinal_categoricals=True))
    ]
    if use_smote:
        num_cols, cat_cols = split_feature_columns(X_train)
        if cat_cols:
            sampler = SMOTENC(
                categorical_features=list(range(len(num_cols), len(num_cols) + len(cat_cols))),
                sampling_strategy=smote_ratio, k_neighbors=k_neighbors, random_state=RANDOM_STATE,
            )
        else:
            sampler = SMOTE(sampling_strategy=smote_ratio, k_neighbors=k_neighbors, random_state=RANDOM_STATE)
        steps.append(("resample", sampler))
    steps.append(("classifier", estimator))
    return ImbPipeline(steps)


def milling_candidates(y_train: pd.Series, use_smote: bool) -> dict[str, Any]:
    """Five model families. Class weights are used only when SMOTE is off, so the
    imbalance is never corrected twice."""
    cw = None if use_smote else "balanced"
    candidates: dict[str, Any] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, class_weight=None if use_smote else "balanced_subsample",
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=400, class_weight=cw, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, class_weight=cw, random_state=RANDOM_STATE,
        ),
    }
    if CatBoostClassifier is not None:
        candidates["CatBoost"] = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05, loss_function="Logloss",
            auto_class_weights=None if use_smote else "Balanced",
            verbose=False, random_seed=RANDOM_STATE, allow_writing_files=False,
        )
    if XGBClassifier is not None:
        positives = max(1, int((y_train == 1).sum()))
        negatives = max(1, int((y_train == 0).sum()))
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=450, max_depth=4, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=1.0 if use_smote else negatives / positives,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        )
    return candidates


def evaluate_binary(y_true, probabilities, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Full binary evaluation on held-out data: overall, per-class and confusion matrix."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    pred = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    report = classification_report(
        y_true, pred, labels=[0, 1], target_names=["0", "1"], output_dict=True, zero_division=0
    )
    both_classes = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "test_rows": int(len(y_true)),
        "test_positives": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if both_classes else None,
        "average_precision": float(average_precision_score(y_true, probabilities)) if both_classes else None,
        "confusion_matrix": {
            "labels": [0, 1],
            "layout": "[[TN, FP], [FN, TP]] (rows = actual, columns = predicted)",
            "matrix": [[tn, fp], [fn, tp]],
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        },
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in ("0", "1")
        },
        "macro_avg": {k: float(report["macro avg"][k]) for k in ("precision", "recall", "f1-score")},
        "weighted_avg": {k: float(report["weighted avg"][k]) for k in ("precision", "recall", "f1-score")},
    }


def evaluate_mode_attribution(models: dict, X_test: pd.DataFrame, Y_test: pd.DataFrame) -> dict | None:
    """Given a real single-mode failure, does the highest-scoring mode classifier name the right mode?

    Only test rows with Machine failure = 1 and exactly one failure-mode flag are scored; rows with
    several simultaneous modes (or none recorded) are excluded and counted.
    """
    modes = [t for t in MILLING_MODE_TARGETS if t in models and t in Y_test.columns]
    if len(modes) < 2:
        return None
    flags = Y_test[modes].apply(pd.to_numeric, errors="coerce")
    is_failure = (
        pd.to_numeric(Y_test["Machine failure"], errors="coerce") == 1
        if "Machine failure" in Y_test.columns else flags.sum(axis=1) >= 1
    )
    single = is_failure & (flags.sum(axis=1) == 1)
    if int(single.sum()) == 0:
        return None
    scores = np.column_stack([models[m].predict_proba(X_test.loc[single])[:, 1] for m in modes])
    y_true = flags.loc[single].idxmax(axis=1).to_numpy()
    y_pred = np.array(modes)[scores.argmax(axis=1)]
    report = classification_report(y_true, y_pred, labels=modes, output_dict=True, zero_division=0)
    return {
        "labels": modes,
        "rows_evaluated": int(single.sum()),
        "rows_excluded": int(is_failure.sum() - single.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=modes, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=modes).tolist(),
        "confusion_layout": "rows = actual mode, columns = predicted mode (order = labels)",
        "per_class": {
            m: {
                "precision": float(report[m]["precision"]),
                "recall": float(report[m]["recall"]),
                "f1": float(report[m]["f1-score"]),
                "support": int(report[m]["support"]),
            }
            for m in modes
        },
        "note": "Scored only on held-out failures with exactly one recorded mode; excludes multi-mode failures.",
    }


def log_milling_evaluation(metrics: dict, mode_attribution: dict | None) -> None:
    """Readable evaluation tables for the console (everything here is computed on the held-out test split)."""
    LOG.info("=" * 70)
    LOG.info("MILLING EVALUATION (held-out test split, threshold %.2f)", DEFAULT_THRESHOLD)
    for target, m in metrics.items():
        LOG.info("-" * 70)
        LOG.info("[%s] test rows=%s positives=%s | resampling=%s | selected by CV: %s",
                 target, m["test_rows"], m["test_class_counts"].get("1", 0),
                 m["resampling"]["method"], m["selected_model"])
        LOG.info("  %-21s %6s %6s %6s %6s %7s %7s | %5s %4s %4s %4s",
                 "model", "acc", "prec", "rec", "f1", "roc_auc", "pr_auc", "TN", "FP", "FN", "TP")
        for name, e in m["model_comparison"].items():
            cm = e["confusion_matrix"]
            auc = "    n/a" if e["roc_auc"] is None else f"{e['roc_auc']:7.3f}"
            pr = "    n/a" if e["average_precision"] is None else f"{e['average_precision']:7.3f}"
            LOG.info("%s %-21s %6.3f %6.3f %6.3f %6.3f %s %s | %5d %4d %4d %4d",
                     "*" if name == m["selected_model"] else " ", name,
                     e["accuracy"], e["precision"], e["recall"], e["f1"], auc, pr,
                     cm["tn"], cm["fp"], cm["fn"], cm["tp"])
        for label, p in m["best_model_metrics"]["per_class"].items():
            LOG.info("  selected, class %s: precision=%.3f recall=%.3f f1=%.3f support=%d",
                     label, p["precision"], p["recall"], p["f1"], p["support"])
    if mode_attribution:
        LOG.info("-" * 70)
        LOG.info("FAILURE-MODE ATTRIBUTION (given a real single-mode failure): rows=%s accuracy=%.3f macro-F1=%.3f",
                 mode_attribution["rows_evaluated"], mode_attribution["accuracy"], mode_attribution["macro_f1"])
        LOG.info("  labels (rows=actual, cols=predicted): %s", mode_attribution["labels"])
        for label, row in zip(mode_attribution["labels"], mode_attribution["confusion_matrix"]):
            LOG.info("  %-4s %s", label, row)
        for label, p in mode_attribution["per_class"].items():
            LOG.info("  %-4s precision=%.3f recall=%.3f f1=%.3f support=%d",
                     label, p["precision"], p["recall"], p["f1"], p["support"])
    LOG.info("=" * 70)


class MillingDataError(Exception):
    """Milling data could not be prepared; the message is reported as the skip reason."""


def prepare_milling_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the milling CSV(s). Returns (df, X features, Y labels, available label names)."""
    paths = findData(data_dir / "milling")
    if not paths:
        LOG.warning("Milling: no CSV found in %s; skipping.", data_dir / "milling")
        raise MillingDataError("No milling CSV found")

    df = combine_csvs(paths)
    if df.empty:
        raise MillingDataError("Milling files could not be read")

    # Find columns case/whitespace-insensitively, then rename to expected names.
    rename_map = {}
    for expected in MILLING_FEATURES + MILLING_FAILURE_TARGETS + ["UDI", "Product ID"]:
        actual = find_column(df, [expected])
        if actual is not None:
            rename_map[actual] = expected
    df = df.rename(columns=rename_map)

    missing_features = [c for c in MILLING_FEATURES if c not in df.columns]
    if missing_features:
        LOG.warning("Milling: missing required features %s; skipping.", missing_features)
        LOG.warning("Milling columns found: %s", list(df.columns))
        raise MillingDataError(f"Missing features: {missing_features}")

    # Avoid identifiers, source filenames and all output labels as input features.
    X = df[MILLING_FEATURES].copy()
    X["Type"] = X["Type"].astype("string").fillna("Unknown").astype(str)
    for col in MILLING_FEATURES:
        if col != "Type":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    available_targets = [t for t in MILLING_FAILURE_TARGETS if t in df.columns]
    for t in MILLING_FAILURE_TARGETS:
        if t not in df.columns:
            LOG.warning("Milling target '%s' missing; it will not be trained.", t)
    if not available_targets:
        raise MillingDataError("No failure label columns found")
    Y = df[available_targets].apply(pd.to_numeric, errors="coerce")
    return df, X, Y, available_targets


def train_milling(data_dir: Path, model_dir: Path, resampling: str = "smote",
                  smote_ratio: float = DEFAULT_SMOTE_RATIO) -> dict:
    use_smote = resampling == "smote"
    if use_smote and SMOTENC is None:
        raise ImportError("SMOTE resampling needs imbalanced-learn: pip install imbalanced-learn "
                          "(or run with --resampling none to use class weights instead).")
    try:
        df, X, Y, available_targets = prepare_milling_data(data_dir)
    except MillingDataError as exc:
        return {"status": "skipped", "reason": str(exc)}

    # One split for all targets, made BEFORE any resampling, scaling or model fitting.
    train_pos, test_pos = make_milling_split(df, X, available_targets)
    LOG.info("Milling split: %s train rows / %s held-out test rows (stratified on label combination, seed %s)",
             len(train_pos), len(test_pos), RANDOM_STATE)

    trained: dict[str, Any] = {}
    metrics: dict[str, dict] = {}
    for target in available_targets:
        y_all = Y[target]
        tr = train_pos[y_all.iloc[train_pos].isin([0, 1]).to_numpy()]
        te = test_pos[y_all.iloc[test_pos].isin([0, 1]).to_numpy()]
        X_train, y_train = X.iloc[tr], y_all.iloc[tr].astype(int)
        X_test, y_test = X.iloc[te], y_all.iloc[te].astype(int)
        counts = y_all[y_all.isin([0, 1])].astype(int).value_counts().to_dict()
        train_counts = y_train.value_counts().to_dict()

        # Do not create fake probabilities from a single-class dataset.
        if len(counts) < 2:
            LOG.warning("Milling target '%s' skipped: needs both 0 and 1 labels; observed %s.", target, counts)
            continue
        if min(counts.values()) < 3 or sum(counts.values()) < 20 or len(train_counts) < 2:
            LOG.warning("Milling target '%s' skipped: class counts %s are too small.", target, counts)
            continue

        n_minority = min(train_counts.values())
        folds = min(CV_FOLDS, n_minority)
        if folds < 2:
            LOG.warning("Milling target '%s' skipped: only %s minority training rows.", target, n_minority)
            continue
        # SMOTE needs k_neighbors < minority rows in every (smaller) CV training fold.
        minority_in_fold = n_minority - int(np.ceil(n_minority / folds))
        k_neighbors = max(1, min(5, minority_in_fold - 1))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

        comparison, cv_scores, fitted = {}, {}, {}
        for name, estimator in milling_candidates(y_train, use_smote).items():
            candidate = build_milling_pipeline(estimator, X_train, use_smote, smote_ratio, k_neighbors)
            try:
                # Selection metric: average precision on validation folds of the TRAINING split.
                # SMOTE is applied only to each fold's training part, never to its validation part.
                scores = cross_val_score(candidate, X_train, y_train, cv=cv, scoring="average_precision")
                cv_scores[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "folds": int(folds)}
                candidate.fit(X_train, y_train)  # resamples X_train/y_train only
                comparison[name] = evaluate_binary(y_test, candidate.predict_proba(X_test)[:, 1])
                fitted[name] = candidate
                LOG.info("Milling %s / %s: CV AP %.3f | test F1 %.3f",
                         target, name, cv_scores[name]["mean"], comparison[name]["f1"])
            except Exception as exc:
                LOG.warning("Milling candidate %s failed for %s: %s", name, target, exc)

        if not fitted:
            LOG.warning("All milling candidate models failed for target %s", target)
            continue
        # Chosen on training-split CV only; test metrics did not influence the choice.
        best_name = max(cv_scores, key=lambda n: cv_scores[n]["mean"])
        pipeline = fitted[best_name]  # already fit on the training split only; NOT refit on test rows

        resampling_info: dict[str, Any] = {
            "method": "SMOTENC" if use_smote else "class_weight",
            "applied_to": "training split only (inside each CV training fold and the final fit)",
        }
        if use_smote:
            _, y_resampled = pipeline[:-1].fit_resample(X_train, y_train)
            resampling_info.update({
                "sampling_strategy": float(smote_ratio),
                "k_neighbors": int(k_neighbors),
                "train_class_counts_before": {str(k): int(v) for k, v in train_counts.items()},
                "train_class_counts_after": {str(k): int(v) for k, v in pd.Series(y_resampled).value_counts().items()},
            })

        metric = {
            "rows": int(sum(counts.values())),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "train_class_counts": {str(k): int(v) for k, v in train_counts.items()},
            "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().items()},
            "resampling": resampling_info,
            "selected_model": best_name,
            "selection_metric": f"{folds}-fold CV average precision on the training split only",
            "cv_scores": cv_scores,
            "best_model_metrics": comparison[best_name],
            "model_comparison": comparison,
            "note": "Metrics are on the held-out test split at threshold "
                    f"{DEFAULT_THRESHOLD}. The model was not refit on these rows.",
        }
        trained[target] = pipeline
        metrics[target] = metric

    if not trained:
        LOG.warning("Milling: no target had enough positive and negative labels to train.")
        return {
            "status": "skipped",
            "reason": "No eligible failure target. Need full data with both 0 and 1 for at least one label.",
            "rows": len(df),
        }

    X_eval, Y_eval = X.iloc[test_pos], Y.iloc[test_pos]
    mode_attribution = evaluate_mode_attribution(trained, X_eval, Y_eval)
    log_milling_evaluation(metrics, mode_attribution)

    # Held-out rows are saved so the Streamlit simulation can replay ONLY unseen data.
    holdout_cols = [c for c in ["UDI", "Product ID"] if c in df.columns]
    holdout = pd.concat([df.iloc[test_pos][holdout_cols], X.iloc[test_pos], df.iloc[test_pos][available_targets]], axis=1)
    holdout["__source_file__"] = df["__source_file__"].iloc[test_pos].to_numpy()
    holdout["__source_row__"] = df.groupby("__source_file__").cumcount().iloc[test_pos].to_numpy()
    out_dir = model_dir / "milling"
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(out_dir / HOLDOUT_FILENAME, index=False)
    LOG.info("Saved %s held-out test rows: %s", len(holdout), out_dir / HOLDOUT_FILENAME)

    numeric_cols = [c for c in MILLING_FEATURES if c != "Type"]
    bundle = {
        "operation": "milling",
        "model_type": "one binary classifier per available failure label"
                      + (" (SMOTENC on the training split only)" if use_smote else " (class weights)"),
        "feature_columns": MILLING_FEATURES,
        "target_columns": list(trained.keys()),
        "models": trained,
        "metrics": metrics,
        "mode_attribution": mode_attribution,
        "feature_ranges": {
            c: {"min": float(X.iloc[train_pos][c].min()), "max": float(X.iloc[train_pos][c].max()),
                "median": float(X.iloc[train_pos][c].median())}
            for c in numeric_cols
        },
        "category_values": {"Type": sorted(X.iloc[train_pos]["Type"].unique().tolist())},
        "split": {
            "test_size": TEST_SIZE, "random_state": RANDOM_STATE,
            "train_rows": int(len(train_pos)), "test_rows": int(len(test_pos)),
            "holdout_file": HOLDOUT_FILENAME,
            "stratified_on": "combination of all failure labels (rare combinations pooled)",
            "models_trained_on": "training split only (not refit on test rows)",
        },
        "note": "Scores are model-estimated risk, not guarantees. With SMOTE the scores are not calibrated "
                "probabilities, so treat the threshold as an operating point, not a literal percentage.",
    }
    save_bundle(model_dir, "milling", bundle)
    (out_dir / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "mode_attribution": mode_attribution, "split": bundle["split"]},
                   indent=2, default=str),
        encoding="utf-8",
    )
    return {
        "status": "trained", "rows": len(df), "targets": list(trained),
        "split": bundle["split"], "metrics": metrics, "mode_attribution": mode_attribution,
    }


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
                "feature_ranges": bundle.get("feature_ranges", {}),
                "category_values": bundle.get("category_values", {}),
                "split": bundle.get("split"),
                "has_evaluation": any("test_rows" in m for m in bundle.get("metrics", {}).values()),
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


def derive_failure_mode(failures: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Turn per-target scores into one failure decision and one most-likely failure mode.

    A failure is predicted when the 'Machine failure' score or any mode score reaches the threshold.
    If so, the primary mode is the mode classifier with the highest score (even if it is below the
    threshold, in which case the mode is reported as low-confidence). Otherwise: 'No failure'.
    """
    mode_scores = {t: float(v["probability"]) for t, v in failures.items() if t in MILLING_MODE_TARGETS}
    machine = failures.get("Machine failure")
    failure_score = float(machine["probability"]) if machine else (max(mode_scores.values()) if mode_scores else 0.0)
    flagged = sorted((t for t, p in mode_scores.items() if p >= threshold), key=lambda t: -mode_scores[t])
    predicted = failure_score >= threshold or bool(flagged)
    primary, confident = "No failure", True
    if predicted:
        if flagged:
            primary = flagged[0]
        elif mode_scores:
            primary, confident = max(mode_scores, key=mode_scores.get), False
        else:
            primary, confident = "Unknown (no mode models trained)", False
    return {
        "predicted_failure": bool(predicted),
        "failure_score": failure_score,
        "predicted_failure_mode": primary,
        "failure_mode_confident": bool(confident),
        "flagged_modes": flagged,
        "mode_scores": mode_scores,
    }


def predict_milling(
    data: dict | list[dict] | pd.DataFrame,
    models: dict[str, dict] | None = None,
    model_dir: str | Path = "models",
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Predict milling failure risk and failure mode. Input keys are MILLING_FEATURES.

    Each prediction has per-target scores under "failures", plus "predicted_failure",
    "predicted_failure_mode", "flagged_modes" and "mode_scores".

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
        results.append({
            "failures": failures,
            "any_failure_predicted": any(v["predicted_failure"] for v in failures.values()),
            **derive_failure_mode(failures, threshold),
        })
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
    parser.add_argument("--data-dir", default="Data", help="Folder containing milling/turning/drilling subfolders")
    parser.add_argument("--model-dir", default="models", help="Output folder for trained model bundles")
    parser.add_argument("--operation", choices=["all", "milling", "turning", "drilling"], default="all",
                        help="Which model(s) to train. Other operations' saved models/reports are left untouched.")
    parser.add_argument("--resampling", choices=["smote", "none"], default="smote",
                        help="Milling class-imbalance handling: SMOTE on the training split only (default) "
                             "or none (class weights, no resampling).")
    parser.add_argument("--smote-ratio", type=float, default=DEFAULT_SMOTE_RATIO,
                        help="Milling SMOTE target minority:majority ratio on training data (default 0.5).")
    parser.add_argument("--drilling-window", type=int, default=1000,
                        help="Raw drilling samples per feature window (default 1000 at 1000 Hz = 1 second)")
    args = parser.parse_args()
    if not 0.0 < args.smote_ratio <= 1.0:
        parser.error("--smote-ratio must be in (0, 1]")

    configure_logging()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    wanted = {"milling", "turning", "drilling"} if args.operation == "all" else {args.operation}

    LOG.info("=" * 70)
    LOG.info("Starting combined tool-wear model training (%s)", args.operation)
    LOG.info("Data directory: %s", data_dir.resolve())
    LOG.info("Model directory: %s", model_dir.resolve())
    LOG.info("=" * 70)

    report_path = model_dir / "training_report.json"
    results: dict[str, Any] = {}
    if args.operation != "all" and report_path.exists():
        try:  # keep the previous results of operations that are not retrained
            results = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            results = {}
    results["optional_dependencies"] = {
        "catboost_available": CatBoostClassifier is not None,
        "xgboost_available": XGBClassifier is not None,
        "imbalanced_learn_available": SMOTENC is not None,
        "install_hint": "Install optional candidates with: pip install catboost xgboost imbalanced-learn",
    }
    if "milling" in wanted:
        results["milling"] = train_milling(data_dir, model_dir, resampling=args.resampling, smote_ratio=args.smote_ratio)
    if "turning" in wanted:
        results["turning"] = train_turning(data_dir, model_dir)
    if "drilling" in wanted:
        results["drilling"] = train_drilling(data_dir, model_dir, window_size=args.drilling_window)

    report_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    LOG.info("=" * 70)
    LOG.info("TRAINING SUMMARY")
    for operation in ("milling", "turning", "drilling"):
        result = results.get(operation)
        if not result or operation not in wanted:
            continue
        LOG.info("%s: %s", operation.upper(), result.get("status", "unknown"))
        if result.get("reason"):
            LOG.info("  Reason: %s", result["reason"])
        if operation != "milling" and result.get("metrics"):  # milling tables were already logged in full
            LOG.info("  Metrics: %s", result["metrics"])
    LOG.info("Training report saved to %s", report_path.resolve())
    LOG.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
