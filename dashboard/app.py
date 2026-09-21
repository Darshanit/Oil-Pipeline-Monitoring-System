"""
Streamlit Control Room Dashboard for Oil Pipeline Pressure Monitoring & Leak Detection System.

Run with:
  streamlit run dashboard/app.py
"""

import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
import streamlit as st

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generation import STATIONS_KM
from src.detection import fuse_evidence_signals, get_recommended_action
from src.localization import estimate_leak_location

# Page Configuration
st.set_page_config(
    page_title="Oil Pipeline Monitoring System",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Control Room Aesthetics
st.markdown("""
<style>
    /* Dark glassmorphism container styling */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
    }
    .status-normal {
        background-color: rgba(40, 167, 69, 0.2);
        border: 2px solid #28a745;
        color: #28a745;
        padding: 12px;
        border-radius: 8px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
    }
    .status-suspect {
        background-color: rgba(255, 193, 7, 0.2);
        border: 2px solid #ffc107;
        color: #ffc107;
        padding: 12px;
        border-radius: 8px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
    }
    .status-leak {
        background-color: rgba(220, 53, 69, 0.2);
        border: 2px solid #dc3545;
        color: #dc3545;
        padding: 12px;
        border-radius: 8px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 0 15px rgba(220, 53, 69, 0.5);
    }
    .action-box {
        background: rgba(23, 162, 184, 0.15);
        border-left: 4px solid #17a2b8;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 15px;
        margin-top: 8px;
        margin-bottom: 14px;
    }
    .suppression-banner {
        background: rgba(49, 130, 206, 0.2);
        border-left: 5px solid #3182ce;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 14px;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .localization-card {
        background: rgba(229, 62, 62, 0.12);
        border: 1px solid rgba(229, 62, 62, 0.35);
        border-radius: 8px;
        padding: 12px 18px;
        margin-top: 8px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_scenario_detection_results(scenario_name: str, seed: int = 42) -> pd.DataFrame:
    """
    Loads or executes detection results for the specified demonstration scenario.
    Uses Streamlit caching so that scrub/playback do not trigger unnecessary recomputation.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if "FULL" in scenario_name:
        csv_path = os.path.join(base_dir, "data/processed/detection_results.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        from src.pipeline.run_detection import run_full_pipeline
        return run_full_pipeline(base_dir)

    # Deterministic scenario generation & real pipeline execution
    from src.simulation import generate_scenario_data
    from src.pipeline.run_detection import execute_detection_pipeline

    if "NORMAL" in scenario_name:
        sc_type = "NORMAL"
    elif "SMALL" in scenario_name:
        sc_type = "SMALL CHRONIC LEAK"
    elif "LARGE" in scenario_name:
        sc_type = "LARGE LEAK"
    elif "PUMP" in scenario_name:
        sc_type = "PUMP TRANSIENT"
    else:
        sc_type = "VALVE EVENT"

    df_raw = generate_scenario_data(sc_type, duration_sec=120.0, random_seed=seed)
    return execute_detection_pipeline(df_raw, base_dir=base_dir)


def main():
    st.title("🛢️ Oil Pipeline Pressure Monitoring & Leak Detection System")
    st.markdown("*Real-Time Multi-Sensor Feature Fusion & Acoustic/Hydraulic Localization Prototype*")

    # Sidebar Controls
    st.sidebar.header("⚙️ Simulation Controls")

    # Scenario Selector
    scenario_options = [
        "FULL 600s TIMELINE (Original)",
        "1. NORMAL (Steady Nominal Baseline)",
        "2. SMALL CHRONIC LEAK (Pinhole at 57 km)",
        "3. LARGE LEAK (NPW Rupture at 78 km)",
        "4. PUMP TRANSIENT (Surge with Mode Gating)",
        "5. VALVE EVENT (Throttling Transient)"
    ]
    selected_scenario = st.sidebar.selectbox("Active Demonstration Scenario:", scenario_options, index=3)

    # Load cached scenario results
    df = load_scenario_detection_results(selected_scenario)

    # Interactive Time Slider
    max_t = float(df["timestamp"].max())
    default_val = 540.0 if "FULL" in selected_scenario else min(50.0, max_t)
    selected_t = st.sidebar.slider(
        "Playback Timeline (seconds):",
        min_value=0.0,
        max_value=max_t,
        value=float(default_val),
        step=0.5
    )

    # Configurable Evidence Weights
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚖️ Evidence Fusion Weights")
    w_ml = st.sidebar.slider("ML Anomaly Weight", 0.0, 1.0, 0.30, 0.05)
    w_flow = st.sidebar.slider("Flow Imbalance Weight", 0.0, 1.0, 0.35, 0.05)
    w_npw = st.sidebar.slider("NPW Wave Front Weight", 0.0, 1.0, 0.15, 0.05)
    w_press = st.sidebar.slider("Pressure Evidence Weight", 0.0, 1.0, 0.20, 0.05)

    # Dynamic 4-channel fusion recalculation if weights adjusted
    custom_weights = {"ml": w_ml, "flow": w_flow, "npw": w_npw, "pressure": w_press}
    press_scores = df["pressure_evidence_score"].values if "pressure_evidence_score" in df.columns else None
    df_fusion, _ = fuse_evidence_signals(
        df["ml_evidence_score"].values,
        df["flow_evidence_score"].values,
        df["npw_evidence_score"].values,
        pressure_scores=press_scores,
        weights=custom_weights,
        operating_modes=df.get("operating_mode"),
        event_types=df.get("event_ground_truth"),
        timestamps=df["timestamp"].values
    )

    # Update dynamic columns
    df["confidence_pct"] = df_fusion["confidence_pct"]
    df["status"] = df_fusion["status"]
    if "suppression_reason" in df_fusion.columns:
        df["suppression_reason"] = df_fusion["suppression_reason"]

    # Filter data up to selected timestamp
    df_current = df[df["timestamp"] <= selected_t]
    current_row = df_current.iloc[-1] if len(df_current) > 0 else df.iloc[0]

    status = current_row["status"]
    conf = current_row["confidence_pct"]
    mode = current_row["operating_mode"]
    gt_event = current_row["event_ground_truth"]
    supp_reason = current_row.get("suppression_reason")

    # Top KPI Metrics Header
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**System Status**")
        if status == "LEAK DETECTED":
            st.markdown('<div class="status-leak">🚨 LEAK DETECTED</div>', unsafe_allow_html=True)
        elif status == "SUSPECTED ANOMALY":
            st.markdown('<div class="status-suspect">⚠️ SUSPECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-normal">✅ NORMAL</div>', unsafe_allow_html=True)

    with col2:
        st.metric("Leak Confidence", f"{conf:.1f}%", delta=f"{conf - 10.0:.1f}%" if conf > 10 else None)
        st.progress(min(1.0, max(0.0, conf / 100.0)))

    with col3:
        st.metric("Operating Mode", mode)
        st.caption(f"Ground Truth Event: **{gt_event}**")

    with col4:
        est_km = current_row.get("estimated_leak_km", np.nan)
        if pd.notna(est_km):
            st.metric("Estimated Location", f"{est_km:.1f} km", delta=f"{current_row['nearest_station']}")
        else:
            st.metric("Estimated Location", "None", delta="Pipeline Nominal")

    # Scenario-Specific Notifications & Honest Proof Display
    if supp_reason and str(supp_reason) not in ["nan", "None", ""]:
        st.markdown(
            f'<div class="suppression-banner">🛡️ <strong>Mode Gating Active:</strong> {supp_reason}. '
            f'Confidence is capped to prevent false CRITICAL alarms.</div>',
            unsafe_allow_html=True
        )

    # Large Leak Localization Verification Display
    gt_km = current_row.get("ground_truth_km")
    err_km = current_row.get("localization_error_km")
    if pd.notna(gt_km):
        loc_c1, loc_c2, loc_c3, loc_c4 = st.columns(4)
        with loc_c1:
            st.metric("🎯 Ground Truth", f"{gt_km:.1f} km")
        with loc_c2:
            st.metric("📍 Estimated Location", f"{est_km:.1f} km" if pd.notna(est_km) else "Awaiting Wave...")
        with loc_c3:
            st.metric("📏 Localization Error", f"{err_km:.1f} km" if pd.notna(err_km) else "—")
        with loc_c4:
            st.metric("⚡ Wave Speed", "1.0 km/s (1000 m/s)")

    # Small Chronic Leak Honest Diagnostic Notice
    if "SMALL" in selected_scenario and "LEAK" in gt_event:
        st.caption("ℹ️ **Honest Diagnostic Note**: Pinhole chronic leak (~16 m³/h). Detectable via sustained flow mass imbalance and subtle gradient shift; gradual onset correctly does not generate an acoustic shockwave.")

    # Recommended Action Box
    action_text = current_row.get("recommended_action", "NORMAL OPERATION: All parameters within standard operational baseline.")
    st.markdown(f'<div class="action-box"><strong>Recommended Operator Action:</strong> {action_text}</div>', unsafe_allow_html=True)

    # Row 1: Pipeline Visualization Schematic
    st.subheader("📍 Pipeline Schematic & Spatial Sensor Layout")

    fig_pipe = go.Figure()

    # Draw pipeline main line
    fig_pipe.add_trace(go.Scatter(
        x=[0, 100], y=[0, 0],
        mode="lines",
        line=dict(color="#4a5568", width=10),
        name="Pipeline Body",
        hoverinfo="skip"
    ))

    # Add 6 monitoring stations
    st_names = [f"Station {i+1}<br>{int(x)} km" for i, x in enumerate(STATIONS_KM)]
    fig_pipe.add_trace(go.Scatter(
        x=STATIONS_KM, y=[0]*6,
        mode="markers+text",
        marker=dict(symbol="square", size=24, color="#3182ce", line=dict(color="#ffffff", width=2)),
        text=st_names,
        textposition="top center",
        name="Monitoring Stations"
    ))

    # Add ground truth leak marker if present
    if pd.notna(gt_km):
        fig_pipe.add_trace(go.Scatter(
            x=[gt_km], y=[0],
            mode="markers+text",
            marker=dict(symbol="circle-open", size=36, color="#48bb78", line=dict(color="#48bb78", width=3)),
            text=[f"GROUND TRUTH<br>{gt_km:.1f} km"],
            textposition="top center",
            name="Ground Truth"
        ))

    # Add estimated leak beacon if active
    if pd.notna(est_km):
        fig_pipe.add_trace(go.Scatter(
            x=[est_km], y=[0],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=32, color="#e53e3e", line=dict(color="#ffffff", width=3)),
            text=[f"ESTIMATED<br>{est_km:.1f} km"],
            textposition="bottom center",
            name="Estimated Leak"
        ))

    fig_pipe.update_layout(
        xaxis=dict(range=[-5, 105], title="Pipeline Distance (km)", showgrid=True, zeroline=False),
        yaxis=dict(range=[-1.5, 1.5], showticklabels=False, showgrid=False, zeroline=False),
        height=220,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )
    st.plotly_chart(fig_pipe, width="stretch")

    # Row 2: Two-column layout for telemetry graphs & evidence scores
    chart_col, evidence_col = st.columns([2, 1])

    with chart_col:
        st.subheader("📈 Multi-Station Sensor Telemetry (10 Hz)")

        # Pressure graph
        fig_p = go.Figure()
        for i in range(1, 7):
            fig_p.add_trace(go.Scatter(
                x=df_current["timestamp"],
                y=df_current[f"P_st{i}"],
                mode="lines",
                name=f"Station {i} ({int(STATIONS_KM[i-1])} km)"
            ))

        fig_p.add_vline(x=selected_t, line_dash="dash", line_color="white", annotation_text="Current Time")
        fig_p.update_layout(
            title="Station Pressures (bar)",
            xaxis_title="Time (seconds)",
            yaxis_title="Pressure (bar)",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.15, x=0)
        )
        st.plotly_chart(fig_p, width="stretch")

        # Flow graph
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=df_current["timestamp"], y=df_current["flow_in"], mode="lines", name="Inlet Flow (m³/h)", line=dict(color="#3182ce")))
        fig_q.add_trace(go.Scatter(x=df_current["timestamp"], y=df_current["flow_out"], mode="lines", name="Outlet Flow (m³/h)", line=dict(color="#805ad5")))
        if "corrected_flow_imbalance" in df_current.columns:
            fig_q.add_trace(go.Scatter(x=df_current["timestamp"], y=df_current["corrected_flow_imbalance"], mode="lines", name="Corrected Imbalance (m³/h)", line=dict(color="#319795")))
        fig_q.add_vline(x=selected_t, line_dash="dash", line_color="white")
        fig_q.update_layout(
            title="Flow Dynamics & Line-Pack Corrected Imbalance (m³/h)",
            xaxis_title="Time (seconds)",
            yaxis_title="Flow Rate (m³/h)",
            height=260,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.15, x=0)
        )
        st.plotly_chart(fig_q, width="stretch")

    with evidence_col:
        st.subheader("🔍 Evidence Stream Decomposition")

        # Current row evidence breakdown
        ml_ev = current_row["ml_evidence_score"] * 100.0
        flow_ev = current_row["flow_evidence_score"] * 100.0
        npw_ev = current_row["npw_evidence_score"] * 100.0
        press_ev = current_row.get("pressure_evidence_score", 0.0) * 100.0

        st.markdown(f"**ML Anomaly Score ({w_ml*100:.0f}% Weight)**")
        st.progress(min(1.0, max(0.0, ml_ev / 100.0)))
        st.caption(f"Score: **{ml_ev:.1f}%** (Mode: '{mode}')")

        st.markdown(f"**Flow Imbalance Score ({w_flow*100:.0f}% Weight)**")
        st.progress(min(1.0, max(0.0, flow_ev / 100.0)))
        st.caption(f"Score: **{flow_ev:.1f}%** (Line-pack corrected mass balance)")

        st.markdown(f"**Pressure Evidence Score ({w_press*100:.0f}% Weight)**")
        st.progress(min(1.0, max(0.0, press_ev / 100.0)))
        st.caption(f"Score: **{press_ev:.1f}%** (Hydraulic gradient kink & slopes)")

        st.markdown(f"**NPW Wave Front Score ({w_npw*100:.0f}% Weight)**")
        st.progress(min(1.0, max(0.0, npw_ev / 100.0)))
        st.caption(f"Score: **{npw_ev:.1f}%** (10 Hz acoustic drop wave front)")

        st.markdown("---")
        st.subheader("📊 Fused Leak Confidence Timeline")
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(x=df["timestamp"], y=df["confidence_pct"], mode="lines", name="Confidence %", line=dict(color="#e53e3e", width=2)))
        fig_conf.add_hline(y=65.0, line_dash="dash", line_color="red", annotation_text="Leak Threshold")
        fig_conf.add_hline(y=35.0, line_dash="dot", line_color="orange", annotation_text="Suspect Threshold")
        fig_conf.add_vline(x=selected_t, line_dash="dash", line_color="white")
        fig_conf.update_layout(
            title="Fused Confidence (%) over Time",
            xaxis_title="Time (s)",
            yaxis_title="Confidence (%)",
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        st.plotly_chart(fig_conf, width="stretch")

    # Footer
    st.markdown("---")
    st.caption("Oil Pipeline Pressure Monitoring & Leak Detection System • Built with Python, Scikit-Learn, SciPy & Streamlit")


if __name__ == "__main__":
    main()
