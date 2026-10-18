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


def show_milling_result(result: dict) -> None:
    predictions = result.get("predictions", [])
    if not predictions:
        st.warning("The milling model returned no predictions.")
        return
    failures = predictions[0].get("failures", {})
    if not failures:
        st.warning("No trained milling failure targets were found in the saved model.")
        return

    st.markdown("#### Failure predictions")
    rows = []
    for label, detail in failures.items():
        rows.append({
            "Failure condition": label,
            "Description": FAILURE_DESCRIPTIONS.get(label, ""),
            "Selected model": detail.get("selected_model", "unknown"),
            "Risk (%)": float(detail.get("risk_percent", 0.0)),
            "Predicted failure": "Yes" if detail.get("predicted_failure") else "No",
        })
    frame = pd.DataFrame(rows)
    st.dataframe(frame, hide_index=True, use_container_width=True)
    cols = st.columns(min(3, len(rows)))
    for index, row in enumerate(rows):
        with cols[index % len(cols)]:
            st.metric(row["Failure condition"], f'{row["Risk (%)"]:.2f}%')
    st.caption("Probabilities are model estimates, not guarantees. The alert threshold is configurable in the sidebar.")


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
    auto_refresh = st.toggle("Automatically sample new data", value=True)
    refresh_seconds = st.select_slider(
        "Sampling refresh interval",
        options=[5, 10, 15, 30, 60],
        value=5,
        format_func=lambda value: f"{value} seconds",
        disabled=not auto_refresh,
    )
    threshold = st.slider("Milling failure alert threshold", 0.05, 0.95, 0.50, 0.05)
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

sample_metadata = {}
sampling_error = None
sample_frame = None
prediction = None

# Always select data from dataSampling.py; no upload or manual data entry is shown.
try:
    if operation == "milling":
        sample_frame = sample_milling(n=1, data_dir=DATA_DIR)
        sample_metadata = {
            "source_file": str(sample_frame.iloc[0].get("__source_file__", "unknown")),
            "source_row": int(sample_frame.iloc[0].get("__source_row__", -1)),
        }
        reading = sample_frame.iloc[0].to_dict()
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
        f'Random sample selected · Source: {sample_metadata.get("source_file", "dataset window")} · '
        f'Row/start: {sample_metadata.get("source_row", sample_metadata.get("start_sample", "N/A"))}'
    )

    if operation == "milling":
        sensor_metrics(reading)
        warnings = rule_based_warnings(reading)
        st.subheader("Milling model output")
        if prediction is not None:
            show_milling_result(prediction)
        else:
            st.warning("Milling model is unavailable. Run `python model.py` to train it.")
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
        with st.expander("View the randomly selected source row"):
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
    "Data is randomly sampled from local datasets and refreshed on the selected interval. "
    "It is not a direct physical sensor feed. Predictions and anomaly flags should be validated before operational use."
)

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
