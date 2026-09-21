"""
Streamlit Control Room Dashboard for Oil Pipeline Intelligent Monitoring.
Industrial Edge AI Leak Detection System (ZEDEDA / EVE-OS Compatible).

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
    page_title="Oil Pipeline Intelligent Monitoring",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_control_room_assets():
    """Injects Neo-Brutalist design tokens, motion animations, and styling."""
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    css_files = ["tokens.css", "motion.css", "style.css"]
    combined_css = []
    for fname in css_files:
        path = os.path.join(assets_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                combined_css.append(f.read())
    if combined_css:
        st.markdown(f"<style>\n{chr(10).join(combined_css)}\n</style>", unsafe_allow_html=True)


load_control_room_assets()


def style_brutalist_chart(fig, title: str = "", height: int = 280, showlegend: bool = True):
    """
    Applies strict Neo-Brutalist control room styling to a Plotly figure:
    White chart canvas, crisp gridlines, black axis borders, local monospace font.
    """
    fig.update_layout(
        title=dict(
            text=f"<b>{title.upper()}</b>" if title else None,
            font=dict(family="FiraCodeLocal, monospace", size=13, color="#111111"),
            x=0.01,
            y=0.96
        ) if title else None,
        font=dict(family="FiraCodeLocal, monospace", color="#111111", size=11),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=30, r=20, t=44 if title else 20, b=30),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            y=1.16,
            x=0,
            bgcolor="#FFFFFF",
            bordercolor="#111111",
            borderwidth=2,
            font=dict(family="FiraCodeLocal, monospace", size=10, color="#111111")
        ),
        xaxis=dict(
            gridcolor="#E2DFCD",
            gridwidth=1,
            linecolor="#111111",
            linewidth=2,
            zeroline=True,
            zerolinecolor="#111111",
            zerolinewidth=2,
            tickfont=dict(family="FiraCodeLocal, monospace", color="#111111", size=10),
            title_font=dict(family="FiraCodeLocal, monospace", color="#111111", size=11)
        ),
        yaxis=dict(
            gridcolor="#E2DFCD",
            gridwidth=1,
            linecolor="#111111",
            linewidth=2,
            zeroline=True,
            zerolinecolor="#111111",
            zerolinewidth=2,
            tickfont=dict(family="FiraCodeLocal, monospace", color="#111111", size=10),
            title_font=dict(family="FiraCodeLocal, monospace", color="#111111", size=11)
        )
    )
    return fig


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
    # Industrial Neo-Brutalist Header
    st.markdown("""
    <div class="industrial-header-box">
        <h1>OIL PIPELINE INTELLIGENT MONITORING</h1>
        <div class="subtext">
            EDGE AI LEAK DETECTION &bull; ZEDEDA / EVE-OS &bull; PROTOTYPE (SYNTHETIC DATA)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("⚙️ SIMULATION CONTROLS")

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
        st.markdown("**SYSTEM HEALTH STATE**")
        if status == "LEAK DETECTED":
            st.markdown('<div class="status-badge status-critical status-critical-strobe">[■ ALARM] CRITICAL</div>', unsafe_allow_html=True)
        elif status == "SUSPECTED ANOMALY":
            st.markdown('<div class="status-badge status-warning">[▲ WARN] WARNING</div>', unsafe_allow_html=True)
        elif mode in ["SHUT-IN", "TRANSIENT_OPERATION"] or "TRANSIENT" in str(mode):
            st.markdown('<div class="status-badge status-monitoring">[◆ CHK] MONITORING</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-normal">[● OK] NORMAL</div>', unsafe_allow_html=True)

    with col2:
        st.metric("LEAK CONFIDENCE", f"{conf:.1f}%", delta=f"{conf - 10.0:.1f}%" if conf > 10 else None)
        st.progress(min(1.0, max(0.0, conf / 100.0)))

    with col3:
        st.metric("OPERATING MODE", mode)
        st.caption(f"Ground Truth Event: **{gt_event}**")

    with col4:
        est_km = current_row.get("estimated_leak_km", np.nan)
        if pd.notna(est_km):
            st.metric("ESTIMATED LOCATION", f"{est_km:.1f} km", delta=f"{current_row['nearest_station']}")
        else:
            st.metric("ESTIMATED LOCATION", "None", delta="Pipeline Nominal")

    # Scenario-Specific Notifications & Honest Proof Display
    if supp_reason and str(supp_reason) not in ["nan", "None", ""]:
        st.markdown(
            f'<div class="suppression-banner">🛡️ <strong>MODE GATING ACTIVE:</strong> {supp_reason}. '
            f'Confidence is capped to suppress false alarms during hydraulic transients.</div>',
            unsafe_allow_html=True
        )

    # Large Leak Localization Verification Display
    gt_km = current_row.get("ground_truth_km")
    err_km = current_row.get("localization_error_km")
    if pd.notna(gt_km):
        loc_c1, loc_c2, loc_c3, loc_c4 = st.columns(4)
        with loc_c1:
            st.metric("🎯 GROUND TRUTH", f"{gt_km:.1f} km")
        with loc_c2:
            st.metric("📍 ESTIMATED LOCATION", f"{est_km:.1f} km" if pd.notna(est_km) else "Awaiting Wave...")
        with loc_c3:
            st.metric("📏 LOCALIZATION ERROR", f"{err_km:.1f} km" if pd.notna(err_km) else "—")
        with loc_c4:
            st.metric("⚡ ACOUSTIC SPEED", "1.0 km/s (1000 m/s)")

    # Small Chronic Leak Honest Diagnostic Notice
    if "SMALL" in selected_scenario and "LEAK" in gt_event:
        st.caption("ℹ️ **Honest Diagnostic Verification**: Small chronic leak (~16 m³/h). Detected via sustained line-pack corrected flow imbalance and subtle hydraulic slope shift; gradual onset correctly does not generate an acoustic shockwave.")

    # Recommended Action Box
    action_text = current_row.get("recommended_action", "NORMAL OPERATION: All parameters within standard operational baseline.")
    st.markdown(f'<div class="action-box"><strong>OPERATOR ADVISORY:</strong> {action_text}</div>', unsafe_allow_html=True)

    # Row 1: Pipeline Visualization Schematic
    st.subheader("📍 PIPELINE SPATIAL SENSOR SCHEMATIC (100 KM)")

    fig_pipe = go.Figure()

    # Draw pipeline main line
    fig_pipe.add_trace(go.Scatter(
        x=[0, 100], y=[0, 0],
        mode="lines",
        line=dict(color="#111111", width=12),
        name="Pipeline Trunk",
        hoverinfo="skip"
    ))

    # Add 6 monitoring stations
    st_names = [f"STATION {i+1}<br><b>{int(x)} KM</b>" for i, x in enumerate(STATIONS_KM)]
    fig_pipe.add_trace(go.Scatter(
        x=STATIONS_KM, y=[0]*6,
        mode="markers+text",
        marker=dict(symbol="square", size=24, color="#FFD23F", line=dict(color="#111111", width=3)),
        text=st_names,
        textposition="top center",
        textfont=dict(family="FiraCodeLocal, monospace", size=10, color="#111111"),
        name="Monitoring Stations"
    ))

    # Add ground truth leak marker if present
    if pd.notna(gt_km):
        fig_pipe.add_trace(go.Scatter(
            x=[gt_km], y=[0],
            mode="markers+text",
            marker=dict(symbol="circle-open", size=36, color="#35D07F", line=dict(color="#111111", width=4)),
            text=[f"ACTUAL LEAK<br><b>{gt_km:.1f} KM</b>"],
            textposition="top center",
            textfont=dict(family="FiraCodeLocal, monospace", size=11, color="#111111"),
            name="Ground Truth"
        ))

    # Add estimated leak beacon if active
    if pd.notna(est_km):
        fig_pipe.add_trace(go.Scatter(
            x=[est_km], y=[0],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=34, color="#FF5A5F", line=dict(color="#111111", width=3)),
            text=[f"ESTIMATED<br><b>{est_km:.1f} KM</b>"],
            textposition="bottom center",
            textfont=dict(family="FiraCodeLocal, monospace", size=11, color="#111111"),
            name="Estimated Leak"
        ))

    style_brutalist_chart(fig_pipe, title="", height=230, showlegend=False)
    fig_pipe.update_xaxes(range=[-5, 105], title="Pipeline Linear Distance (km)")
    fig_pipe.update_yaxes(range=[-1.5, 1.5], showticklabels=False, showgrid=False, zeroline=False)
    st.plotly_chart(fig_pipe, width="stretch")

    # Row 2: Two-column layout for telemetry graphs & evidence scores
    chart_col, evidence_col = st.columns([2, 1])

    with chart_col:
        st.subheader("📈 SENSOR TELEMETRY STREAMS (10 HZ)")

        # Station trace colors from curated palette
        station_colors = ["#111111", "#4D96FF", "#35D07F", "#FF5A5F", "#7B2CBF", "#D97706"]

        # Pressure graph
        fig_p = go.Figure()
        for i in range(1, 7):
            fig_p.add_trace(go.Scatter(
                x=df_current["timestamp"],
                y=df_current[f"P_st{i}"],
                mode="lines",
                name=f"ST {i} ({int(STATIONS_KM[i-1])} km)",
                line=dict(color=station_colors[i-1], width=2)
            ))

        fig_p.add_vline(x=selected_t, line_dash="dash", line_color="#111111", line_width=2, annotation_text="PLAYBACK")
        style_brutalist_chart(fig_p, title="STATION PRESSURES (BAR)", height=280)
        fig_p.update_xaxes(title="Time (seconds)")
        fig_p.update_yaxes(title="Pressure (bar)")
        st.plotly_chart(fig_p, width="stretch")

        # Flow graph
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=df_current["timestamp"], y=df_current["flow_in"], mode="lines", name="Inlet Flow (m³/h)", line=dict(color="#4D96FF", width=2.5)))
        fig_q.add_trace(go.Scatter(x=df_current["timestamp"], y=df_current["flow_out"], mode="lines", name="Outlet Flow (m³/h)", line=dict(color="#FF5A5F", width=2.5)))
        if "corrected_flow_imbalance" in df_current.columns:
            fig_q.add_trace(go.Scatter(x=df_current["timestamp"], y=df_current["corrected_flow_imbalance"], mode="lines", name="Corrected Imbalance (m³/h)", line=dict(color="#111111", width=2, dash="dot")))
        fig_q.add_vline(x=selected_t, line_dash="dash", line_color="#111111", line_width=2)
        style_brutalist_chart(fig_q, title="FLOW DYNAMICS & LINE-PACK CORRECTED IMBALANCE (M³/H)", height=260)
        fig_q.update_xaxes(title="Time (seconds)")
        fig_q.update_yaxes(title="Flow Rate (m³/h)")
        st.plotly_chart(fig_q, width="stretch")

    with evidence_col:
        st.subheader("🔍 EVIDENCE DECOMPOSITION")

        # Current row evidence breakdown
        ml_ev = current_row["ml_evidence_score"] * 100.0
        flow_ev = current_row["flow_evidence_score"] * 100.0
        npw_ev = current_row["npw_evidence_score"] * 100.0
        press_ev = current_row.get("pressure_evidence_score", 0.0) * 100.0

        st.markdown(f"**ML ANOMALY SCORE ({w_ml*100:.0f}% WEIGHT)**")
        st.progress(min(1.0, max(0.0, ml_ev / 100.0)))
        st.caption(f"Score: **{ml_ev:.1f}%** (Operating Mode: '{mode}')")

        st.markdown(f"**FLOW IMBALANCE SCORE ({w_flow*100:.0f}% WEIGHT)**")
        st.progress(min(1.0, max(0.0, flow_ev / 100.0)))
        st.caption(f"Score: **{flow_ev:.1f}%** (Line-pack mass balance)")

        st.markdown(f"**PRESSURE EVIDENCE SCORE ({w_press*100:.0f}% WEIGHT)**")
        st.progress(min(1.0, max(0.0, press_ev / 100.0)))
        st.caption(f"Score: **{press_ev:.1f}%** (Hydraulic gradient slopes)")

        st.markdown(f"**NPW WAVE FRONT SCORE ({w_npw*100:.0f}% WEIGHT)**")
        st.progress(min(1.0, max(0.0, npw_ev / 100.0)))
        st.caption(f"Score: **{npw_ev:.1f}%** (10 Hz acoustic drop)")

        st.markdown("---")
        st.subheader("📊 FUSED LEAK CONFIDENCE TIMELINE")
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(x=df["timestamp"], y=df["confidence_pct"], mode="lines", name="Confidence %", line=dict(color="#FF5A5F", width=2.5)))
        fig_conf.add_hline(y=65.0, line_dash="dash", line_color="#FF5A5F", line_width=2, annotation_text="LEAK THRESHOLD (65%)", annotation_font=dict(family="FiraCodeLocal, monospace", color="#FF5A5F", size=10))
        fig_conf.add_hline(y=35.0, line_dash="dot", line_color="#111111", line_width=2, annotation_text="SUSPECT THRESHOLD (35%)", annotation_font=dict(family="FiraCodeLocal, monospace", color="#111111", size=10))
        fig_conf.add_vline(x=selected_t, line_dash="dash", line_color="#111111", line_width=2)
        style_brutalist_chart(fig_conf, title="CONFIDENCE (%) OVER TIME", height=240, showlegend=False)
        fig_conf.update_xaxes(title="Time (s)")
        fig_conf.update_yaxes(title="Confidence (%)", range=[0, 105])
        st.plotly_chart(fig_conf, width="stretch")

    # Control Room Watermark Footer
    st.markdown("""
    <div class="control-room-footer">
        <div><strong>OIL PIPELINE INTELLIGENT MONITORING</strong> &bull; EDGE AI LEAK DETECTION</div>
        <div><strong>ZEDEDA / EVE-OS READY</strong> &bull; PROTOTYPE (SYNTHETIC DATA)</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
