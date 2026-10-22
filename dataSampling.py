from __future__ import annotations

"""Randomly sample real rows/windows from the local machining datasets.

Expected folders (case-insensitive on Windows):
    Data/milling/   CSV/TXT/XLSX/XLS
    Data/turning/   CSV/TXT/XLSX/XLS
    Data/drilling/  CSV/TXT/XLSX/XLS

Functions return pandas DataFrames ready to pass to the prediction functions in model.py.
Drilling sampling returns a contiguous raw X/Y/Z window, not summary features.
"""

from pathlib import Path
from typing import Iterable
import random
import re

import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".xlsx", ".xls"}
MILLING_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COLUMNS = {"machine failure", "twf", "hdf", "pwf", "osf", "rnf", "vb", "wear"}
AXES = ("X", "Y", "Z")


def _norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _find_folder(root: Path, name: str) -> Path:
    """Find a named data subfolder without relying on capitalization."""
    direct = root / name
    if direct.is_dir():
        return direct
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and child.name.casefold() == name.casefold():
                return child
    return direct


def _find_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    files = sorted(
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        and not path.name.startswith("~$")
    )
    if not files:
        raise FileNotFoundError(f"No CSV/TXT/Excel dataset files found in: {folder}")
    return files


def _read_table(path: Path, *, drilling: bool = False) -> pd.DataFrame:
    if path.suffix.casefold() in {".xlsx", ".xls"}:
        # The drilling workbooks have metadata in rows 1-4 and headers on row 5.
        frame = pd.read_excel(path, header=4 if drilling else 0)
    else:
        errors: list[str] = []
        frame = None
        for encoding in ("utf-8-sig", "utf-8", "latin1"):
            for sep in (",", ";", "\t", None):
                try:
                    candidate = pd.read_csv(path, encoding=encoding, sep=sep, engine="python")
                    if candidate.shape[1] > 1 and len(candidate) > 0:
                        frame = candidate
                        break
                except Exception as exc:  # try other common encodings/separators
                    errors.append(str(exc))
            if frame is not None:
                break
        if frame is None:
            detail = errors[-1] if errors else "could not infer delimiter"
            raise ValueError(f"Could not read {path}: {detail}")

    frame.columns = [str(col).strip() for col in frame.columns]
    # Some datasets include unnamed index columns; preserve useful sensor columns only later.
    return frame.reset_index(drop=True)


def _load_frames(operation: str, data_dir: str | Path = "Data") -> list[tuple[Path, pd.DataFrame]]:
    root = Path(data_dir).expanduser().resolve()
    folder = _find_folder(root, operation)
    files = _find_files(folder)
    frames: list[tuple[Path, pd.DataFrame]] = []
    failures: list[str] = []
    for path in files:
        try:
            frames.append((path, _read_table(path, drilling=operation == "drilling")))
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")
    if not frames:
        extra = " | ".join(failures[:3])
        raise ValueError(f"No readable {operation} datasets in {folder}. {extra}")
    return frames


def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {_norm(col): col for col in frame.columns}


def _rename_known_columns(frame: pd.DataFrame, expected: Iterable[str]) -> pd.DataFrame:
    lookup = _column_lookup(frame)
    rename = {lookup[_norm(col)]: col for col in expected if _norm(col) in lookup}
    return frame.rename(columns=rename)


MILLING_LABELS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
MILLING_HOLDOUT_FILENAME = "holdout_test.csv"  # written by model.py next to the milling model


def sample_milling(
    n: int = 1,
    data_dir: str | Path = "Data",
    random_state: int | None = None,
    split: str = "test",
    model_dir: str | Path = "models",
) -> pd.DataFrame:
    """Return n random milling rows with the six model input columns.

    split="test" (default): sample ONLY from the held-out test rows saved by model.py
        (models/milling/holdout_test.csv). The milling model never trained on these rows, so
        predictions on them are an honest replay. The true failure labels are included so the
        UI can show actual vs predicted.
    split="full": sample from every row of Data/milling. Those rows include the ones the model was
        trained on, so do not use this to judge accuracy.

    If n exceeds the number of rows, sampling is performed with replacement. Use the returned
    DataFrame directly with model.predict_milling().
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if split not in {"test", "full"}:
        raise ValueError('split must be "test" or "full"')
    if split == "test":
        holdout_path = Path(model_dir).expanduser().resolve() / "milling" / MILLING_HOLDOUT_FILENAME
        if not holdout_path.is_file():
            raise FileNotFoundError(
                f"Held-out milling test rows not found at {holdout_path}. "
                "Retrain with `python model.py --operation milling` to create them."
            )
        pool = pd.read_csv(holdout_path)
        missing = [col for col in MILLING_FEATURES if col not in pool.columns]
        if missing or pool.empty:
            raise ValueError(f"{holdout_path} is empty or missing columns {missing}. Retrain the milling model.")
        keep = MILLING_FEATURES + [c for c in MILLING_LABELS if c in pool.columns] + ["__source_file__", "__source_row__"]
        pool = pool[[c for c in keep if c in pool.columns]]
        return pool.sample(n=n, replace=n > len(pool), random_state=random_state).reset_index(drop=True)
    frames = []
    for path, frame in _load_frames("milling", data_dir):
        frame = _rename_known_columns(frame, MILLING_FEATURES)
        missing = [col for col in MILLING_FEATURES if col not in frame.columns]
        if missing:
            continue
        selected = frame[MILLING_FEATURES].copy()
        selected["__source_file__"] = path.name
        selected["__source_row__"] = np.arange(len(selected))
        frames.append(selected)
    if not frames:
        raise ValueError(f"No milling files contain all required columns: {MILLING_FEATURES}")
    pool = pd.concat(frames, ignore_index=True)
    return pool.sample(n=n, replace=n > len(pool), random_state=random_state).reset_index(drop=True)


def sample_turning(
    n: int = 1,
    feature_columns: list[str] | None = None,
    data_dir: str | Path = "Data",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Return n random turning rows, aligned to model feature columns if provided.

    If feature_columns is None, the function infers numeric columns while excluding
    target/identifier fields. For best compatibility pass the saved model bundle's
    feature_columns from get_model_info(models)["turning"].
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    frames = []
    for path, frame in _load_frames("turning", data_dir):
        frame = frame.copy()
        frame["__source_file__"] = path.name
        frames.append(frame)
    pool = pd.concat(frames, ignore_index=True, sort=False)

    if feature_columns is not None:
        lookup = _column_lookup(pool)
        rename = {lookup[_norm(col)]: col for col in feature_columns if _norm(col) in lookup}
        pool = pool.rename(columns=rename)
        missing = [col for col in feature_columns if col not in pool.columns]
        if missing:
            raise ValueError(
                f"Turning dataset is missing {len(missing)} model feature columns. "
                f"Examples: {missing[:10]}"
            )
        selected = pool[feature_columns].copy()
    else:
        excluded_norms = TARGET_COLUMNS | {
            "tool", "tool id", "tool_id", "product id", "product_id", "udi",
            "run", "sample no", "sample no.", "__source_file__",
        }
        columns = []
        for col in pool.columns:
            if _norm(col) in excluded_norms:
                continue
            numeric = pd.to_numeric(pool[col], errors="coerce")
            if numeric.notna().sum() >= max(3, int(0.5 * len(pool))):
                columns.append(col)
        if not columns:
            raise ValueError("No numeric turning sensor features could be inferred.")
        selected = pool[columns].copy()

    selected["__source_file__"] = pool.loc[selected.index, "__source_file__"].to_numpy()
    selected["__source_row__"] = selected.index.to_numpy()
    return selected.sample(n=n, replace=n > len(selected), random_state=random_state).reset_index(drop=True)


def _axis_columns(frame: pd.DataFrame) -> dict[str, str] | None:
    lookup = _column_lookup(frame)
    aliases = {
        "X": ("x", "x_acceleration", "acceleration_x", "x axis", "x_axis", "vibration_x"),
        "Y": ("y", "y_acceleration", "acceleration_y", "y axis", "y_axis", "vibration_y"),
        "Z": ("z", "z_acceleration", "acceleration_z", "z axis", "z_axis", "vibration_z"),
    }
    result = {}
    for axis, candidates in aliases.items():
        found = next((lookup[name] for name in candidates if name in lookup), None)
        if found is None:
            return None
        result[axis] = found
    return result


def sample_drilling_window(
    window_size: int = 1000,
    data_dir: str | Path = "Data",
    random_state: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return a random contiguous X/Y/Z window from a drilling file.

    Returns (raw_window, metadata). The window is suitable for model.predict_drilling()
    when window_size matches the trained detector's window size (normally 1000 rows).
    Only complete windows are sampled; no artificial samples are generated.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    rng = random.Random(random_state)
    candidates: list[tuple[Path, pd.DataFrame, dict[str, str], int]] = []
    for path, frame in _load_frames("drilling", data_dir):
        axes = _axis_columns(frame)
        if axes is None:
            continue
        raw = frame[[axes["X"], axes["Y"], axes["Z"]]].copy()
        raw.columns = list(AXES)
        for axis in AXES:
            raw[axis] = pd.to_numeric(raw[axis], errors="coerce")
        raw = raw.dropna(subset=list(AXES), how="all").reset_index(drop=True)
        if len(raw) >= window_size:
            candidates.append((path, raw, axes, len(raw) - window_size + 1))
    if not candidates:
        raise ValueError(
            f"No drilling file contains at least {window_size} usable X/Y/Z samples. "
            "Check the Data/drilling files and the model's window size."
        )
    # Weight each possible start equally across all files, rather than choosing a file uniformly.
    total_starts = sum(item[3] for item in candidates)
    pick = rng.randrange(total_starts)
    chosen = candidates[0]
    start = 0
    for item in candidates:
        if pick < item[3]:
            chosen = item
            start = pick
            break
        pick -= item[3]
    path, raw, _axes, _count = chosen
    result = raw.iloc[start:start + window_size].reset_index(drop=True)
    metadata = {
        "operation": "drilling",
        "source_file": path.name,
        "start_sample": int(start),
        "end_sample": int(start + window_size - 1),
        "window_size_samples": int(window_size),
    }
    return result, metadata


def sample_random_data(
    operation: str,
    n: int = 1,
    data_dir: str | Path = "Data",
    feature_columns: list[str] | None = None,
    window_size: int = 1000,
    random_state: int | None = None,
):
    """Convenience dispatcher: milling/turning -> DataFrame; drilling -> (DataFrame, metadata)."""
    op = str(operation).strip().casefold()
    if op == "milling":
        return sample_milling(n=n, data_dir=data_dir, random_state=random_state)
    if op == "turning":
        return sample_turning(n=n, feature_columns=feature_columns, data_dir=data_dir, random_state=random_state)
    if op == "drilling":
        if n != 1:
            raise ValueError("Drilling sampling returns one contiguous window per call; use sample_drilling_window directly.")
        return sample_drilling_window(window_size=window_size, data_dir=data_dir, random_state=random_state)
    raise ValueError("operation must be one of: milling, turning, drilling")
