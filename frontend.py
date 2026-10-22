from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from model import (
    get_model_info,
    load_models,
    predict_drilling,
    predict_milling,
    predict_turning,
)
from dataSampling import sample_milling, sample_turning, sample_drilling_window

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Warehouse Tool Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Explicitly set metric text colors: Streamlit themes can otherwise render white
# text on the light metric cards.
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      [data-testid="stSidebar"] { background: #111827; }
      [data-testid="stSidebar"] * { color: #f9fafb; }
      .dashboard-subtitle { color: #64748b; margin-top: -8px; }
      div[data-testid="stMetric"] {
        background: #f8fafc !important;
        border: 1px solid #dbe3ee !important;
        padding: 14px 16px !important;
        border-radius: 12px !important;
      }
      div[data-testid="stMetric"] label,
      div[data-testid="stMetric"] label p,
      div[data-testid="stMetricLabel"],
      div[data-testid="stMetricLabel"] p {
        color: #334155 !important;
      }
      div[data-testid="stMetricValue"],
      div[data-testid="stMetricValue"] div,
      div[data-testid="stMetricValue"] p {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        font-weight: 700 !important;
      }
      div[data-testid="stMetricDelta"] { color: #475569 !important; }
      .stCaption, [data-testid="stCaptionContainer"] { color: #64748b; }
    </style>
    """,
    unsafe_allow_html=True,
)

VIEW_REPLAY = "Live replay (held-out test rows)"
VIEW_MANUAL = "Manual prediction"
VIEW_EVAL = "Model evaluation"

FAILURE_DESCRIPTIONS = {
    "Machine failure": "Any recorded machine failure",
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

TOOLS = {
    "Milling Machine 01": {
        "id": "MILL-001", "operation": "milling", "location": "Zone A · Bay 01"
    },
    "Lathe Machine 02": {
        "id": "LATHE-002", "operation": "turning", "location": "Zone A · Bay 02"
    },
    "Drilling Machine 03": {
        "id": "DRILL-003", "operation": "drilling", "location": "Zone B · Bay 01"
    },
}


@st.cache_resource
def get_loaded_models():
    return load_models(MODEL_DIR, strict=False)


def display_name(name: str) -> str:
    return str(name).replace("_", " ").strip()


def show_milling_result(result: dict, actual: dict | None = None) -> None:
    """Show failure / failure-mode predictions. `actual` (held-out replay only) adds true labels."""
    predictions = result.get("predictions", [])
    if not predictions:
        st.warning("The milling model returned no predictions.")
        return
    first = predictions[0]
    failures = first.get("failures", {})
    if not failures:
        st.warning("No trained milling failure targets were found in the saved model.")
        return

    st.markdown("#### Predicted failure and failure mode")
    mode = first.get("predicted_failure_mode", "No failure")
    if first.get("predicted_failure"):
        text = f"**Failure predicted** · most likely mode: **{mode}** ({FAILURE_DESCRIPTIONS.get(mode, '')})"
        if not first.get("failure_mode_confident", True):
            text += " · low confidence: no individual mode reached the alert threshold"
        st.error(text)
        if len(first.get("flagged_modes", [])) > 1:
            st.caption("Several modes are above the threshold: " + ", ".join(first["flagged_modes"]))
    else:
        st.success("**No failure predicted** at the current alert threshold.")

    if actual:
        actual_modes = [t for t in ("TWF", "HDF", "PWF", "OSF", "RNF") if actual.get(t) == 1]
        st.info(
            "**Actual (from the dataset):** "
            + ("failure" if actual.get("Machine failure") == 1 else "no failure")
            + (f" · mode(s): {', '.join(actual_modes)}" if actual_modes else "")
        )

    st.markdown("#### Per-target scores")
    rows = []
    for label, detail in failures.items():
        row = {
            "Failure condition": label,
            "Description": FAILURE_DESCRIPTIONS.get(label, ""),
            "Selected model": detail.get("selected_model", "unknown"),
            "Risk score (%)": float(detail.get("risk_percent", 0.0)),
            "Predicted failure": "Yes" if detail.get("predicted_failure") else "No",
        }
        if actual and label in actual:
            row["Actual"] = "Yes" if actual[label] == 1 else "No"
            row["Match"] = "✓" if bool(actual[label]) == bool(detail.get("predicted_failure")) else "✗"
        rows.append(row)
    frame = pd.DataFrame(rows)
    st.dataframe(frame, hide_index=True, use_container_width=True)
    cols = st.columns(min(3, len(rows)))
    for index, row in enumerate(rows):
        with cols[index % len(cols)]:
            st.metric(row["Failure condition"], f'{row["Risk score (%)"]:.2f}%')
    st.caption(
        "Scores are model estimates, not guarantees. The models were trained with SMOTE, so scores are not "
        "calibrated probabilities. The alert threshold is configurable in the sidebar."
    )


def render_manual_milling(models: dict, model_info: dict, threshold: float) -> None:
    """Form: enter a new milling datapoint and predict failure + failure mode."""
    st.subheader("Manual prediction")
    if "milling" not in models:
        st.warning("Milling model is unavailable. Run `python model.py --operation milling` to train it.")
        return
    info = model_info.get("milling", {})
    ranges = info.get("feature_ranges", {})
    types = info.get("category_values", {}).get("Type") or ["L", "M", "H"]

    def default(column: str, fallback: float) -> float:
        return float(ranges[column]["median"]) if column in ranges else fallback

    st.caption("Enter sensor values for a new datapoint. Defaults are the training-data medians.")
    with st.form("manual_milling_form"):
        left, right = st.columns(2)
        product_type = left.selectbox("Product quality type", types, help="L = low, M = medium, H = high quality variant")
        air = left.number_input("Air temperature [K]", min_value=0.0, value=round(default("Air temperature [K]", 300.0), 1), step=0.1, format="%.1f")
        process = left.number_input("Process temperature [K]", min_value=0.0, value=round(default("Process temperature [K]", 310.0), 1), step=0.1, format="%.1f")
        rpm = right.number_input("Rotational speed [rpm]", min_value=0.0, value=float(round(default("Rotational speed [rpm]", 1500.0))), step=1.0, format="%.0f")
        torque = right.number_input("Torque [Nm]", min_value=0.0, value=round(default("Torque [Nm]", 40.0), 1), step=0.1, format="%.1f")
        wear = right.number_input("Tool wear [min]", min_value=0.0, value=float(round(default("Tool wear [min]", 100.0))), step=1.0, format="%.0f")
        submitted = st.form_submit_button("Predict failure and failure mode", type="primary")

    if not submitted:
        return
    row = {
        "Type": product_type,
        "Air temperature [K]": air,
        "Process temperature [K]": process,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
    }
    outside = [
        f"{column} = {row[column]:g} (training range {r['min']:g} to {r['max']:g})"
        for column, r in ranges.items()
        if column in row and not (r["min"] <= row[column] <= r["max"])
    ]
    if outside:
        st.warning("Outside the range the model was trained on, so the prediction is an extrapolation: " + "; ".join(outside))
    try:
        result = predict_milling(row, models=models, threshold=threshold)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return
    show_milling_result(result)
    st.markdown("#### Sensor checks")
    warnings = rule_based_warnings(row)
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("No configured rule-based sensor warnings for these values.")
    with st.expander("Submitted values"):
        st.dataframe(pd.DataFrame([row]), hide_index=True, use_container_width=True)


def _metric_row(name: str, e: dict) -> dict:
    cm = e["confusion_matrix"]
    return {
        "Model": name, "Accuracy": e["accuracy"], "Precision": e["precision"], "Recall": e["recall"],
        "F1": e["f1"], "ROC-AUC": e["roc_auc"], "PR-AUC": e["average_precision"],
        "TP": cm["tp"], "FP": cm["fp"], "FN": cm["fn"], "TN": cm["tn"],
    }


def _confusion_figure(matrix, labels_x, labels_y, title: str):
    fig = px.imshow(
        matrix, x=labels_x, y=labels_y, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual"), template="plotly_white", aspect="auto",
    )
    fig.update_layout(title=title, height=330, margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False)
    return fig


def render_milling_evaluation(bundle: dict | None) -> None:
    """Held-out test-set metrics saved by model.py. Nothing here is computed in the UI."""
    st.subheader("Model evaluation (held-out test set)")
    metrics = (bundle or {}).get("metrics", {})
    if not metrics or not any("test_rows" in m for m in metrics.values()):
        st.warning("The saved milling model has no held-out evaluation. Retrain with `python model.py --operation milling`.")
        return
    split = bundle.get("split", {})
    first = next(iter(metrics.values()))
    st.caption(
        f'Trained on {split.get("train_rows", "?"):,} rows, evaluated on {split.get("test_rows", "?"):,} rows the models never saw. '
        f'Imbalance handling: {first["resampling"]["method"]} on the training split only. '
        f'Metrics at threshold {first["best_model_metrics"]["threshold"]:.2f}; the model was selected by '
        f'{first["selection_metric"]}.'
    )

    st.markdown("#### Selected model per target")
    summary = [{"Target": t, **_metric_row(m["selected_model"], m["best_model_metrics"])} for t, m in metrics.items()]
    st.dataframe(pd.DataFrame(summary).round(3), hide_index=True, use_container_width=True)
    st.caption("Rare labels (especially RNF, a random failure with only a handful of test cases) give very noisy metrics.")

    target = st.selectbox("Inspect a target", list(metrics), key="eval_target")
    m = metrics[target]
    st.markdown(f"##### {target} · {FAILURE_DESCRIPTIONS.get(target, '')}")
    counts = {
        "Full dataset": m["class_counts"], "Training split": m["train_class_counts"],
        "Test split": m["test_class_counts"],
    }
    after = m["resampling"].get("train_class_counts_after")
    if after:
        counts["Training split after SMOTE"] = after
    st.dataframe(
        pd.DataFrame(counts).T.rename(columns={"0": "No failure (0)", "1": "Failure (1)"}).fillna(0).astype(int),
        use_container_width=True,
    )

    comparison = pd.DataFrame([_metric_row(n, e) for n, e in m["model_comparison"].items()])
    comparison["CV average precision"] = [m["cv_scores"][n]["mean"] for n in comparison["Model"]]
    comparison["Selected"] = np.where(comparison["Model"] == m["selected_model"], "✓", "")
    st.markdown("All five models on the test set")
    st.dataframe(comparison.round(3), hide_index=True, use_container_width=True)

    chosen = st.selectbox(
        "Model for confusion matrix and per-class metrics", list(m["model_comparison"]),
        index=list(m["model_comparison"]).index(m["selected_model"]), key="eval_model",
    )
    e = m["model_comparison"][chosen]
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            _confusion_figure(e["confusion_matrix"]["matrix"], ["No failure (0)", "Failure (1)"],
                              ["No failure (0)", "Failure (1)"], f"{chosen} · confusion matrix"),
            use_container_width=True,
        )
    with right:
        per_class = pd.DataFrame(e["per_class"]).T.rename(index={"0": "No failure (0)", "1": "Failure (1)"})
        per_class.columns = ["Precision", "Recall", "F1", "Support"]
        st.markdown("Per-class performance")
        st.dataframe(per_class.round(3), use_container_width=True)
        st.caption(f'Balanced accuracy {e["balanced_accuracy"]:.3f} · specificity '
                   f'{e["specificity"] if e["specificity"] is None else round(e["specificity"], 3)}')

    mode = bundle.get("mode_attribution")
    if mode:
        st.markdown("#### Failure-mode attribution")
        st.caption(
            f'Given a real single-mode failure ({mode["rows_evaluated"]} test rows; {mode["rows_excluded"]} multi-mode or '
            f'unlabelled failures excluded), the mode with the highest score is the predicted mode. '
            f'Accuracy {mode["accuracy"]:.3f} · macro-F1 {mode["macro_f1"]:.3f}.'
        )
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                _confusion_figure(mode["confusion_matrix"], mode["labels"], mode["labels"], "Failure mode · confusion matrix"),
                use_container_width=True,
            )
        with right:
            per_mode = pd.DataFrame(mode["per_class"]).T
            per_mode.columns = ["Precision", "Recall", "F1", "Support"]
            st.dataframe(per_mode.round(3), use_container_width=True)


def show_turning_result(result: dict) -> None:
    predictions = result.get("predictions", [])
    if not predictions:
        st.warning("The turning model returned no predictions.")
        return
    first = predictions[0]
    st.metric("Predicted flank wear (Vb)", f'{first["predicted_vb"]:.4f}')
    st.caption(
        f'Selected model: {result.get("selected_model", "unknown")} · '
        f'Unit: {result.get("unit", "same unit as training labels")}'
    )
    if result.get("warning"):
        st.info(result["warning"])


def show_drilling_result(result: dict) -> None:
    rows = result.get("predictions", [])
    st.markdown("#### Vibration anomaly detection")
    st.caption(
        f'{result.get("count", len(rows))} complete windows · '
        f'{result.get("window_size_samples", "N/A")} samples/window · '
        f'{result.get("sampling_rate_hz", "N/A")} Hz'
    )
    if not rows:
        st.warning("No complete drilling windows were scored.")
        return
    frame = pd.DataFrame(rows)
    frame["Status"] = np.where(frame["is_anomaly"], "Anomaly", "Normal-like")
    st.dataframe(frame, hide_index=True, use_container_width=True)
    a, b = st.columns(2)
    a.metric("Windows flagged", int(frame["is_anomaly"].sum()))
    b.metric("Windows analyzed", len(frame))
    if frame["is_anomaly"].any():
        st.warning("One or more vibration windows differ from the learned baseline.")
    else:
        st.success("No windows were flagged by the anomaly detector.")
    st.info("Anomaly scores are not direct measurements or percentages of tool wear.")


def rule_based_warnings(reading: dict) -> list[str]:
    warnings = []
    try:
        air = float(reading["Air temperature [K]"])
        process = float(reading["Process temperature [K]"])
        rpm = float(reading["Rotational speed [rpm]"])
        torque = float(reading["Torque [Nm]"])
        wear = float(reading["Tool wear [min]"])
    except (KeyError, TypeError, ValueError):
        return warnings
    if process - air < 5:
        warnings.append("Small process-to-air temperature difference; verify temperature sensors.")
    if process >= 313:
        warnings.append("Process temperature is high (≥ 313 K); inspect cooling and operating load.")
    if torque >= 55:
        warnings.append("Torque is high (≥ 55 Nm); check cutting load and tool condition.")
    if rpm <= 1000:
        warnings.append("Rotational speed is low (≤ 1000 rpm); check operating settings and drive.")
    if wear >= 200:
        warnings.append("Tool wear is high (≥ 200 min); inspect and consider tool replacement.")
    return warnings


def add_history(operation: str, values: dict) -> pd.DataFrame:
    """Keep a small history of actual sampled rows across Streamlit reruns."""
    now = datetime.now()
    row = {"Timestamp": now, **values}
    key = f"history_{operation}"
    history = st.session_state.get(key, pd.DataFrame())
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    history = history.tail(60)
    st.session_state[key] = history
    return history


def sensor_metrics(reading: dict) -> None:
    wanted = [
        ("Air temperature", "Air temperature [K]", "K", 1),
        ("Process temperature", "Process temperature [K]", "K", 1),
        ("Rotational speed", "Rotational speed [rpm]", "rpm", 0),
        ("Torque", "Torque [Nm]", "Nm", 1),
        ("Tool wear", "Tool wear [min]", "min", 0),
    ]
    available = [(label, key, unit, digits) for label, key, unit, digits in wanted if key in reading and pd.notna(reading[key])]
    if not available:
        st.info("This randomly selected sample has no standard milling-style sensor fields. Showing its feature values below.")
        return
    columns = st.columns(len(available))
    for column, (label, key, unit, digits) in zip(columns, available):
        value = float(reading[key])
        formatted = f"{value:,.{digits}f} {unit}"
        column.metric(label, formatted)


# Load saved models once. Sampling below always comes from the local datasets.
try:
    models = get_loaded_models()
    model_info = get_model_info(models)
    model_load_error = None
except Exception as exc:
    models, model_info = {}, {}
    model_load_error = str(exc)

with st.sidebar:
    st.markdown("## 🏭 Warehouse Monitor")
    st.caption("RANDOM DATASET SAMPLING")
    selected_tool = st.selectbox("Select a machine", list(TOOLS))
    config = TOOLS[selected_tool]
    st.divider()
    st.markdown("### Machine information")
    st.write(f'**Machine ID:** {config["id"]}')
    st.write(f'**Operation:** {config["operation"].title()}')
    st.write(f'**Location:** {config["location"]}')
    st.divider()
    view = VIEW_REPLAY
    if config["operation"] == "milling":
        view = st.radio("Milling view", [VIEW_REPLAY, VIEW_MANUAL, VIEW_EVAL], key="milling_view")
        st.divider()
    auto_refresh = st.toggle(
        "Automatically sample new data", value=True, disabled=view != VIEW_REPLAY, key="auto_refresh",
    ) and view == VIEW_REPLAY
    refresh_seconds = st.select_slider(
        "Sampling refresh interval",
        options=[5, 10, 15, 30, 60],
        value=5,
        format_func=lambda value: f"{value} seconds",
        disabled=not auto_refresh,
    )
    threshold = st.slider("Milling failure alert threshold", 0.05, 0.95, 0.50, 0.05)
    if config["operation"] == "milling":
        st.caption(
            "Replay samples a random row from the held-out milling test split, which the model never "
            "trained on. It is dataset replay, not a physical live sensor connection."
        )
    else:
        st.caption("Every refresh randomly selects a new row/window from Data/. This is dataset replay, not a physical live sensor connection.")

if model_load_error:
    st.error(f"Could not load saved model bundles: {model_load_error}")
if not models:
    st.warning(f"No trained model bundles found in `{MODEL_DIR}`. Run `python model.py` first.")

operation = config["operation"]
now = datetime.now()
st.title(f'🔧 {selected_tool}')
st.markdown(
    f'<p class="dashboard-subtitle">{config["id"]} · {config["location"]} · '
    f'{operation.title()} · Sample time {now:%d %b %Y, %H:%M:%S}</p>',
    unsafe_allow_html=True,
)

# Manual prediction and evaluation are static views: no sampling and no auto-refresh.
if operation == "milling" and view == VIEW_MANUAL:
    render_manual_milling(models, model_info, threshold)
    st.stop()
if operation == "milling" and view == VIEW_EVAL:
    render_milling_evaluation(models.get("milling"))
    st.stop()

sample_metadata = {}
sampling_error = None
sample_frame = None
prediction = None
actual_labels = None

# Replay data comes from dataSampling.py. Milling rows come ONLY from the held-out test split.
try:
    if operation == "milling":
        sample_frame = sample_milling(n=1, data_dir=DATA_DIR, split="test", model_dir=MODEL_DIR)
        sample_metadata = {
            "source_file": str(sample_frame.iloc[0].get("__source_file__", "unknown")),
            "source_row": int(sample_frame.iloc[0].get("__source_row__", -1)),
        }
        reading = sample_frame.iloc[0].to_dict()
        actual_labels = {
            label: int(reading[label])
            for label in ("Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF")
            if label in reading and pd.notna(reading[label])
        }
        if "milling" in models:
            prediction = predict_milling(sample_frame, models=models, threshold=threshold)
    elif operation == "turning":
        required = model_info.get("turning", {}).get("feature_columns", [])
        if not required:
            raise ValueError("The turning model has no saved feature_columns metadata. Retrain model.py first.")
        sample_frame = sample_turning(n=1, feature_columns=required, data_dir=DATA_DIR)
        sample_metadata = {
            "source_file": str(sample_frame.iloc[0].get("__source_file__", "unknown")),
            "source_row": int(sample_frame.iloc[0].get("__source_row__", -1)),
        }
        prediction_input = sample_frame[required].copy()
        reading = sample_frame.iloc[0].to_dict()
        if "turning" in models:
            prediction = predict_turning(prediction_input, models=models)
    else:
        info = model_info.get("drilling", {})
        window_size = int(info.get("window_size_samples", 1000))
        sample_frame, sample_metadata = sample_drilling_window(window_size=window_size, data_dir=DATA_DIR)
        reading = {}
        if "drilling" in models:
            prediction = predict_drilling(sample_frame, models=models)
except Exception as exc:
    reading = {}
    sampling_error = str(exc)

if sampling_error:
    st.error(f"Could not sample or predict {operation} data: {sampling_error}")
    st.info(f"Check that the dataset exists under `{DATA_DIR / operation}` and has the expected columns.")
else:
    st.success(
        f'Random {"held-out test row (never used in training)" if operation == "milling" else "sample"} selected · '
        f'Source: {sample_metadata.get("source_file", "dataset window")} · '
        f'Row/start: {sample_metadata.get("source_row", sample_metadata.get("start_sample", "N/A"))}'
    )

    if operation == "milling":
        sensor_metrics(reading)
        warnings = rule_based_warnings(reading)
        st.subheader("Milling model output")
        if prediction is not None:
            show_milling_result(prediction, actual=actual_labels)
            st.caption(
                "One replayed row says little about accuracy. For overall performance see the "
                "'Model evaluation' view, computed once on the full held-out test set."
            )
        else:
            st.warning("Milling model is unavailable. Run `python model.py --operation milling` to train it.")
        st.subheader("Sensor checks")
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("No configured rule-based sensor warnings for this sample.")
        st.caption("Rule-based checks are separate from the ML predictions.")

        # Only chart fields that exist in the sampled dataset.
        chart_fields = [
            "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        ]
        available_fields = [field for field in chart_fields if field in sample_frame.columns]
        if available_fields:
            history = add_history("milling", {field: reading[field] for field in available_fields})
            st.subheader("Recently sampled sensor values")
            chosen = st.multiselect(
                "Signals to plot", available_fields,
                default=available_fields[:3], key="milling_chart_fields",
            )
            if chosen and len(history) > 1:
                melted = history[["Timestamp"] + chosen].melt(
                    id_vars="Timestamp", var_name="Sensor", value_name="Reading"
                )
                fig = px.line(melted, x="Timestamp", y="Reading", color="Sensor", markers=True, template="plotly_white")
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
        with st.expander("View the randomly selected held-out test row (features and actual labels)"):
            st.dataframe(sample_frame, hide_index=True, use_container_width=True)

    elif operation == "turning":
        st.subheader("Turning flank-wear prediction")
        if prediction is not None:
            show_turning_result(prediction)
        else:
            st.warning("Turning model is unavailable. Run `python model.py` to train it.")
        with st.expander("View randomly selected turning features"):
            st.dataframe(sample_frame, hide_index=True, use_container_width=True)

    else:
        st.subheader("Drilling vibration sample")
        st.write(f'{len(sample_frame):,} raw samples selected from one contiguous window.')
        if prediction is not None:
            show_drilling_result(prediction)
        else:
            st.warning("Drilling model is unavailable. Run `python model.py` to train it.")
        with st.expander("Preview sampled X/Y/Z data"):
            st.dataframe(sample_frame.head(25), hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "Milling rows are replayed from the held-out test split; turning and drilling rows are sampled from the full local "
    "datasets and refreshed on the selected interval. None of it is a direct physical sensor feed. "
    "Predictions and anomaly flags should be validated before operational use."
)

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
