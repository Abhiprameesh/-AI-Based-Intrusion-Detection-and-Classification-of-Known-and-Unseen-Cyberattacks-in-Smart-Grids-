"""
UI Components, Custom Styling, and Plotly Visualizations for Smart Grid IDS Dashboard.
Implements a clean, high-contrast dark navy/charcoal cybersecurity theme.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -----------------------------------------------------------------------------
# Custom CSS Theme Injection
# -----------------------------------------------------------------------------
def apply_custom_css():
    """Injects high-contrast dark navy and charcoal cybersecurity CSS."""
    st.markdown(
        """
        <style>
        /* Base page background */
        .stApp {
            background-color: #0b1120;
            color: #f8fafc;
        }

        /* Top header card */
        .header-container {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px 28px;
            margin-bottom: 24px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
        }
        .header-title {
            color: #38bdf8;
            font-size: 28px;
            font-weight: 800;
            letter-spacing: 1.2px;
            margin: 0;
            padding: 0;
        }
        .header-subtitle {
            color: #94a3b8;
            font-size: 15px;
            font-weight: 500;
            margin-top: 6px;
            margin-bottom: 0;
        }

        /* System status badges */
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 8px;
        }
        .status-active {
            background-color: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid #059669;
        }
        .status-warning {
            background-color: rgba(245, 158, 11, 0.2);
            color: #fbbf24;
            border: 1px solid #d97706;
        }
        .status-alert {
            background-color: rgba(239, 68, 68, 0.2);
            color: #f87171;
            border: 1px solid #dc2626;
        }

        /* KPI Cards */
        .kpi-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 14px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
        }
        .kpi-title {
            color: #94a3b8;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 6px;
        }
        .kpi-value {
            color: #f8fafc;
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .kpi-subtext {
            color: #64748b;
            font-size: 12px;
            font-weight: 500;
        }

        /* Section divider */
        .section-header {
            color: #e2e8f0;
            font-size: 18px;
            font-weight: 700;
            border-bottom: 2px solid #334155;
            padding-bottom: 8px;
            margin-top: 24px;
            margin-bottom: 16px;
        }

        /* Dataframe styling */
        .stDataFrame {
            border: 1px solid #334155;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Renders the top cybersecurity dashboard header and title."""
    st.markdown(
        """
        <div class="header-container">
            <h1 class="header-title">SMART GRID INTRUSION DETECTION SYSTEM</h1>
            <p class="header-subtitle">Three-Stage AI-Based Detection of Known and Unseen Cyberattacks in Smart Grids</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_system_status(status_info: Dict[str, Any]):
    """Renders the system health status indicator strip."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if status_info.get("opendss"):
            st.markdown(
                '<div class="status-badge status-active">● OpenDSS Direct: Connected</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-badge status-alert">● OpenDSS: Unavailable</div>',
                unsafe_allow_html=True,
            )

    with col2:
        if status_info.get("models"):
            st.markdown(
                '<div class="status-badge status-active">● Phase 1 Models: Frozen & Loaded</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-badge status-alert">● AI Models: Missing Artifacts</div>',
                unsafe_allow_html=True,
            )

    with col3:
        if status_info.get("baselines"):
            st.markdown(
                '<div class="status-badge status-active">● Milestone 4 Baseline: Bit-Audited</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-badge status-warning">● Milestone 4 Baseline: Missing</div>',
                unsafe_allow_html=True,
            )

    with col4:
        st.markdown(
            '<div class="status-badge status-active">● Dashboard UI: Ready</div>',
            unsafe_allow_html=True,
        )


def render_kpi_card(title: str, value: str, subtext: str = "", status: str = "normal"):
    """Renders a structured KPI card."""
    color_map = {
        "normal": "#38bdf8",
        "success": "#34d399",
        "warning": "#fbbf24",
        "danger": "#f87171",
    }
    val_color = color_map.get(status, "#f8fafc")

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value" style="color: {val_color};">{value}</div>
            <div class="kpi-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Common Plotly Layout Theme
# -----------------------------------------------------------------------------
DARK_LAYOUT = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#1f2937",
    font=dict(color="#f8fafc", family="sans-serif", size=12),
    xaxis=dict(
        gridcolor="#374151",
        zerolinecolor="#4b5563",
        showline=True,
        linecolor="#4b5563",
    ),
    yaxis=dict(
        gridcolor="#374151",
        zerolinecolor="#4b5563",
        showline=True,
        linecolor="#4b5563",
    ),
    legend=dict(
        bgcolor="rgba(31, 41, 55, 0.8)",
        bordercolor="#4b5563",
        borderwidth=1,
    ),
    margin=dict(l=50, r=30, t=50, b=50),
    hovermode="x unified",
)


# -----------------------------------------------------------------------------
# Tab 1: Grid Monitoring Charts
# -----------------------------------------------------------------------------
def plot_voltage_profiles(features_df: pd.DataFrame, metadata_df: pd.DataFrame) -> go.Figure:
    """Plots nodal Line-to-Neutral voltages with EN 50160 tolerance band."""
    fig = go.Figure()
    steps = metadata_df["step"].values

    # Key voltage features
    voltage_signals = [
        ("Substation-AC-Meter.uln1", "Substation Phase 1 (V)", "#38bdf8"),
        ("PV-Inverter-AC-Meter.uln1", "PV Inverter Phase 1 (V)", "#fbbf24"),
        ("Load-A-AC-Meter.uln1", "Load A Phase 1 (V)", "#34d399"),
        ("Load-B-AC-Meter.uln1", "Load B Phase 1 (V)", "#a78bfa"),
    ]

    for col, name, color in voltage_signals:
        if col in features_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=features_df[col],
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2),
                )
            )

    # Standard nominal band (230 V +/- 5%)
    fig.add_hline(y=241.5, line_dash="dash", line_color="#ef4444", annotation_text="+5% Upper Limit (241.5V)")
    fig.add_hline(y=230.0, line_dash="dot", line_color="#94a3b8", annotation_text="Nominal (230.0V)")
    fig.add_hline(y=218.5, line_dash="dash", line_color="#ef4444", annotation_text="-5% Lower Limit (218.5V)")

    # Attack window highlight
    if "attack_active" in metadata_df.columns and metadata_df["attack_active"].sum() > 0:
        attack_idx = np.where(metadata_df["attack_active"] == 1)[0]
        if len(attack_idx) > 0:
            fig.add_vrect(
                x0=int(attack_idx[0]),
                x1=int(attack_idx[-1]),
                fillcolor="rgba(239, 68, 68, 0.15)",
                layer="below",
                line_width=1,
                line_color="#ef4444",
                annotation_text="Attack Active",
                annotation_position="top left",
            )

    fig.update_layout(
        **DARK_LAYOUT,
        title="Microgrid Nodal Voltage Profiles (Line-to-Neutral)",
        xaxis_title="Simulation Timestep (seconds)",
        yaxis_title="RMS Voltage (V)",
        height=380,
    )
    return fig


def plot_power_balance(metadata_df: pd.DataFrame) -> go.Figure:
    """Plots active power exchange and power balance validation."""
    fig = go.Figure()
    steps = metadata_df["step"].values

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=metadata_df["true_substation_w"] / 1000.0,
            mode="lines",
            name="Substation Net Exchange (kW)",
            line=dict(color="#38bdf8", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=-metadata_df["true_pv_active_w"] / 1000.0,
            mode="lines",
            name="PV Generation (kW)",
            line=dict(color="#fbbf24", width=2),
        )
    )
    total_load = (metadata_df["true_load_a_w"] + metadata_df["true_load_b_w"]) / 1000.0
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=total_load,
            mode="lines",
            name="Total Load Demand (kW)",
            line=dict(color="#f472b6", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=metadata_df["true_bss_active_w"] / 1000.0,
            mode="lines",
            name="Battery BSS Active Power (kW)",
            line=dict(color="#34d399", width=2),
        )
    )

    # Attack window highlight
    if "attack_active" in metadata_df.columns and metadata_df["attack_active"].sum() > 0:
        attack_idx = np.where(metadata_df["attack_active"] == 1)[0]
        if len(attack_idx) > 0:
            fig.add_vrect(
                x0=int(attack_idx[0]),
                x1=int(attack_idx[-1]),
                fillcolor="rgba(239, 68, 68, 0.15)",
                layer="below",
                line_width=1,
                line_color="#ef4444",
            )

    fig.update_layout(
        **DARK_LAYOUT,
        title="Microgrid Active Power Balance (Generation, Storage, Loads, Utility)",
        xaxis_title="Simulation Timestep (seconds)",
        yaxis_title="Active Power (kW)",
        height=380,
    )
    return fig


def plot_battery_soc(metadata_df: pd.DataFrame) -> go.Figure:
    """Plots Battery State-of-Charge (true reference vs reported telemetry)."""
    fig = go.Figure()
    steps = metadata_df["step"].values

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=metadata_df["true_soc_ref"] * 100.0,
            mode="lines",
            name="True Physical SOC (%)",
            line=dict(color="#34d399", width=2.5),
        )
    )

    # If reported SOC was spoofed, show reported trajectory
    if "reported_soc" in metadata_df.columns:
        soc_diff = np.abs(metadata_df["reported_soc"].values - metadata_df["true_soc_ref"].values)
        if np.max(soc_diff) > 1e-4:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=metadata_df["reported_soc"] * 100.0,
                    mode="lines",
                    name="Reported Cyber Telemetry SOC (%)",
                    line=dict(color="#f87171", width=2, dash="dash"),
                )
            )

    fig.update_layout(
        **DARK_LAYOUT,
        title="Battery Storage State of Charge Trajectory",
        xaxis_title="Simulation Timestep (seconds)",
        yaxis_title="Battery SOC (%)",
        height=300,
    )
    return fig


def evaluate_cyber_physical_discrepancy(metadata_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates cyber-physical discrepancies across all available microgrid measurement channels:
    - PV: telemetry_discrepancy_pv_w (or reported_pv_active_w - true_pv_active_w)
    - BSS: telemetry_discrepancy_bss_w (or reported_bss_active_w - true_bss_active_w)
    - Load-B: telemetry_discrepancy_load_b_w (or reported_load_b_w - true_load_b_w)
    - Substation: telemetry_discrepancy_substation_w (or reported_substation_w - true_substation_w)

    Distinguishes three authoritative states:
      1. 'detected': At least one channel has maximum absolute discrepancy >= 1.0 W.
      2. 'no_discrepancy': All available channels have discrepancy < 1.0 W.
      3. 'unavailable': Necessary ground-truth / reported columns are absent.
    """
    if metadata_df is None or len(metadata_df) == 0:
        return {
            "status": "unavailable",
            "channels": {},
            "max_discrepancy_w": 0.0,
            "attack_active_count": 0,
            "attack_start_step": None,
            "attack_end_step": None,
            "target_element": "none",
            "attack_name": "none",
            "is_s5_freeze": False,
        }

    channels: Dict[str, Dict[str, Any]] = {}
    any_channel_available = False

    # 1. PV Inverter Channel
    if "telemetry_discrepancy_pv_w" in metadata_df.columns:
        pv_series = metadata_df["telemetry_discrepancy_pv_w"].astype(float)
        any_channel_available = True
        channels["pv"] = {
            "name": "PV Inverter",
            "series": pv_series,
            "max_abs_w": float(pv_series.abs().max()),
            "has_disc": bool(pv_series.abs().max() >= 1.0),
            "true_col": "true_pv_active_w" if "true_pv_active_w" in metadata_df.columns else None,
            "rep_col": "reported_pv_active_w" if "reported_pv_active_w" in metadata_df.columns else None,
            "negate_kw": True,
            "color_true": "#38bdf8",
            "color_rep": "#ef4444",
        }
    elif "reported_pv_active_w" in metadata_df.columns and "true_pv_active_w" in metadata_df.columns:
        pv_series = (metadata_df["reported_pv_active_w"] - metadata_df["true_pv_active_w"]).astype(float)
        any_channel_available = True
        channels["pv"] = {
            "name": "PV Inverter",
            "series": pv_series,
            "max_abs_w": float(pv_series.abs().max()),
            "has_disc": bool(pv_series.abs().max() >= 1.0),
            "true_col": "true_pv_active_w",
            "rep_col": "reported_pv_active_w",
            "negate_kw": True,
            "color_true": "#38bdf8",
            "color_rep": "#ef4444",
        }

    # 2. Battery Storage (BSS) Channel
    if "telemetry_discrepancy_bss_w" in metadata_df.columns:
        bss_series = metadata_df["telemetry_discrepancy_bss_w"].astype(float)
        any_channel_available = True
        channels["bss"] = {
            "name": "Battery Storage (BSS)",
            "series": bss_series,
            "max_abs_w": float(bss_series.abs().max()),
            "has_disc": bool(bss_series.abs().max() >= 1.0),
            "true_col": "true_bss_active_w" if "true_bss_active_w" in metadata_df.columns else None,
            "rep_col": "reported_bss_active_w" if "reported_bss_active_w" in metadata_df.columns else None,
            "negate_kw": False,
            "color_true": "#34d399",
            "color_rep": "#f59e0b",
        }
    elif "reported_bss_active_w" in metadata_df.columns and "true_bss_active_w" in metadata_df.columns:
        bss_series = (metadata_df["reported_bss_active_w"] - metadata_df["true_bss_active_w"]).astype(float)
        any_channel_available = True
        channels["bss"] = {
            "name": "Battery Storage (BSS)",
            "series": bss_series,
            "max_abs_w": float(bss_series.abs().max()),
            "has_disc": bool(bss_series.abs().max() >= 1.0),
            "true_col": "true_bss_active_w",
            "rep_col": "reported_bss_active_w",
            "negate_kw": False,
            "color_true": "#34d399",
            "color_rep": "#f59e0b",
        }

    # 3. Load-B AC Meter Channel
    if "telemetry_discrepancy_load_b_w" in metadata_df.columns:
        lb_series = metadata_df["telemetry_discrepancy_load_b_w"].astype(float)
        any_channel_available = True
        channels["load_b"] = {
            "name": "Load-B Meter",
            "series": lb_series,
            "max_abs_w": float(lb_series.abs().max()),
            "has_disc": bool(lb_series.abs().max() >= 1.0),
            "true_col": "true_load_b_w" if "true_load_b_w" in metadata_df.columns else None,
            "rep_col": "reported_load_b_w" if "reported_load_b_w" in metadata_df.columns else None,
            "negate_kw": False,
            "color_true": "#38bdf8",
            "color_rep": "#ef4444",
        }
    elif "reported_load_b_w" in metadata_df.columns and "true_load_b_w" in metadata_df.columns:
        lb_series = (metadata_df["reported_load_b_w"] - metadata_df["true_load_b_w"]).astype(float)
        any_channel_available = True
        channels["load_b"] = {
            "name": "Load-B Meter",
            "series": lb_series,
            "max_abs_w": float(lb_series.abs().max()),
            "has_disc": bool(lb_series.abs().max() >= 1.0),
            "true_col": "true_load_b_w",
            "rep_col": "reported_load_b_w",
            "negate_kw": False,
            "color_true": "#38bdf8",
            "color_rep": "#ef4444",
        }

    # 4. Substation AC Meter Channel
    if "telemetry_discrepancy_substation_w" in metadata_df.columns:
        sub_series = metadata_df["telemetry_discrepancy_substation_w"].astype(float)
        any_channel_available = True
        channels["substation"] = {
            "name": "Substation Meter",
            "series": sub_series,
            "max_abs_w": float(sub_series.abs().max()),
            "has_disc": bool(sub_series.abs().max() >= 1.0),
            "true_col": "true_substation_w" if "true_substation_w" in metadata_df.columns else None,
            "rep_col": "reported_substation_w" if "reported_substation_w" in metadata_df.columns else None,
            "negate_kw": False,
            "color_true": "#60a5fa",
            "color_rep": "#fb923c",
        }
    elif "reported_substation_w" in metadata_df.columns and "true_substation_w" in metadata_df.columns:
        sub_series = (metadata_df["reported_substation_w"] - metadata_df["true_substation_w"]).astype(float)
        any_channel_available = True
        channels["substation"] = {
            "name": "Substation Meter",
            "series": sub_series,
            "max_abs_w": float(sub_series.abs().max()),
            "has_disc": bool(sub_series.abs().max() >= 1.0),
            "true_col": "true_substation_w",
            "rep_col": "reported_substation_w",
            "negate_kw": False,
            "color_true": "#60a5fa",
            "color_rep": "#fb923c",
        }

    # Check attack activity interval and metadata
    attack_count = int(metadata_df["attack_active"].sum()) if "attack_active" in metadata_df.columns else 0
    attack_start = None
    attack_end = None
    if attack_count > 0:
        active_steps = metadata_df.loc[metadata_df["attack_active"] == 1, "step"] if "step" in metadata_df.columns else metadata_df.index[metadata_df["attack_active"] == 1]
        if len(active_steps) > 0:
            attack_start = int(active_steps.iloc[0])
            attack_end = int(active_steps.iloc[-1])

    if attack_count > 0:
        active_rows = metadata_df[metadata_df["attack_active"] == 1]
        target_element = str(active_rows["target_element"].iloc[0]) if "target_element" in active_rows.columns and len(active_rows) > 0 else "none"
        attack_name = str(active_rows["attack_name"].iloc[0]) if "attack_name" in active_rows.columns and len(active_rows) > 0 else "none"
        attack_family = str(active_rows["attack_family"].iloc[0]) if "attack_family" in active_rows.columns and len(active_rows) > 0 else "none"
    else:
        target_element = str(metadata_df["target_element"].iloc[0]) if "target_element" in metadata_df.columns and len(metadata_df) > 0 else "none"
        attack_name = str(metadata_df["attack_name"].iloc[0]) if "attack_name" in metadata_df.columns and len(metadata_df) > 0 else "none"
        attack_family = str(metadata_df["attack_family"].iloc[0]) if "attack_family" in metadata_df.columns and len(metadata_df) > 0 else "none"

    scenario_id = str(metadata_df["scenario_id"].iloc[0]) if "scenario_id" in metadata_df.columns and len(metadata_df) > 0 else ""

    is_s5_freeze = bool(
        attack_count > 0
        and (
            "load_b" in attack_name.lower()
            or "freeze" in attack_name.lower()
            or "load-b" in target_element.lower()
            or "telemetry_freeze" in attack_family.lower()
            or "s5" in scenario_id.lower()
        )
    )

    if not any_channel_available:
        status = "unavailable"
        max_disc_w = 0.0
    else:
        max_disc_w = max((ch["max_abs_w"] for ch in channels.values()), default=0.0)
        status = "detected" if max_disc_w >= 1.0 else "no_discrepancy"

    return {
        "status": status,
        "channels": channels,
        "max_discrepancy_w": max_disc_w,
        "attack_active_count": attack_count,
        "attack_start_step": attack_start,
        "attack_end_step": attack_end,
        "target_element": target_element,
        "attack_name": attack_name,
        "is_s5_freeze": is_s5_freeze,
    }


def plot_cyber_physical_discrepancy(metadata_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Plots True Physical State vs Reported Telemetry across all microgrid channels.
    - If numerical discrepancy is detected (>= 1.0 W on any channel): plots manipulated channels
      and highlights the active attack window separately.
    - If S5 Load-B freeze attack is active with 0.0 W discrepancy: plots Load-B telemetry invariance
      with the active freeze attack window (timestep 40 to 110).
    - If no discrepancy is detected and not S5 (e.g. S0 benign baseline): returns None.
    - If discrepancy data is unavailable: returns None.
    """
    audit = evaluate_cyber_physical_discrepancy(metadata_df)
    if audit["status"] == "unavailable":
        return None

    steps = metadata_df["step"].values if "step" in metadata_df.columns else np.arange(len(metadata_df))
    channels = audit["channels"]

    # Case 1: Numerical discrepancy detected
    if audit["status"] == "detected":
        fig = go.Figure()
        for ch_key, ch in channels.items():
            if ch["has_disc"] and ch["true_col"] and ch["rep_col"]:
                sign = -1.0 if ch["negate_kw"] else 1.0
                y_true = sign * metadata_df[ch["true_col"]] / 1000.0
                y_rep = sign * metadata_df[ch["rep_col"]] / 1000.0

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=y_true,
                        mode="lines",
                        name=f"True Physical {ch['name']} (kW)",
                        line=dict(color=ch["color_true"], width=2),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=y_rep,
                        mode="lines",
                        name=f"Reported {ch['name']} Telemetry (kW)",
                        line=dict(color=ch["color_rep"], width=2, dash="dash"),
                    )
                )

        # Highlight active attack interval separately from the discrepancy curve
        if audit["attack_active_count"] > 0 and audit["attack_start_step"] is not None and audit["attack_end_step"] is not None:
            fig.add_vrect(
                x0=audit["attack_start_step"],
                x1=audit["attack_end_step"],
                fillcolor="rgba(239, 68, 68, 0.12)",
                line_width=1,
                line_dash="dash",
                line_color="rgba(239, 68, 68, 0.5)",
                annotation_text=f"Active Attack ({audit['attack_active_count']} steps: t={audit['attack_start_step']}..{audit['attack_end_step']})",
                annotation_position="top left",
                annotation_font=dict(color="#f87171", size=10),
            )

        fig.update_layout(
            **DARK_LAYOUT,
            title="Cyber-Physical Telemetry Discrepancy (True OpenDSS vs Reported Attack Telemetry)",
            xaxis_title="Simulation Timestep (seconds)",
            yaxis_title="Power (kW)",
            height=320,
        )
        return fig

    # Case 2: S5 Load-B Telemetry Freeze (Zero numerical discrepancy on constant physical load)
    if audit["is_s5_freeze"] and "load_b" in channels:
        ch = channels["load_b"]
        if ch["true_col"] and ch["rep_col"]:
            fig = go.Figure()
            y_true = metadata_df[ch["true_col"]] / 1000.0
            y_rep = metadata_df[ch["rep_col"]] / 1000.0

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=y_true,
                    mode="lines",
                    name="True Physical Load-B Power (kW)",
                    line=dict(color="#38bdf8", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=y_rep,
                    mode="lines",
                    name="Reported Load-B Telemetry (kW) [Frozen]",
                    line=dict(color="#ef4444", width=2, dash="dash"),
                )
            )

            if audit["attack_start_step"] is not None and audit["attack_end_step"] is not None:
                fig.add_vrect(
                    x0=audit["attack_start_step"],
                    x1=audit["attack_end_step"],
                    fillcolor="rgba(245, 158, 11, 0.15)",
                    line_width=1,
                    line_dash="dash",
                    line_color="rgba(245, 158, 11, 0.6)",
                    annotation_text=f"Freeze Attack Active ({audit['attack_active_count']} steps: t={audit['attack_start_step']}..{audit['attack_end_step']})",
                    annotation_position="top left",
                    annotation_font=dict(color="#fbbf24", size=10),
                )

            fig.update_layout(
                **DARK_LAYOUT,
                title="Scenario S5: Load-B Telemetry Invariance (Frozen Stale Telemetry on Constant Physical Load)",
                xaxis_title="Simulation Timestep (seconds)",
                yaxis_title="Power (kW)",
                height=320,
            )
            return fig

    # Case 3: Nominal / Benign / Actuator attack without telemetry manipulation
    return None


# -----------------------------------------------------------------------------
# Tab 3: Three-Stage IDS Charts
# -----------------------------------------------------------------------------
def plot_stage1_probabilities(stage1_dict: Dict[str, Any], metadata_df: pd.DataFrame) -> go.Figure:
    """Plots Stage 1 XGBoost attack probabilities with primary and diagnostic thresholds."""
    fig = go.Figure()
    steps = metadata_df["step"].values
    probs = stage1_dict["probabilities"]

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=probs,
            mode="lines",
            name="XGBoost Attack Probability P(Attack)",
            line=dict(color="#38bdf8", width=2.5),
        )
    )

    # Threshold horizontal lines
    fig.add_hline(
        y=stage1_dict["threshold_primary"],
        line_dash="dash",
        line_color="#ef4444",
        line_width=2,
        annotation_text="Primary Threshold tau = 0.50",
    )
    fig.add_hline(
        y=stage1_dict["threshold_diagnostic"],
        line_dash="dot",
        line_color="#f59e0b",
        line_width=2,
        annotation_text="Diagnostic Threshold tau = 0.05",
    )

    # Ground truth attack active window
    if "attack_active" in metadata_df.columns and metadata_df["attack_active"].sum() > 0:
        attack_idx = np.where(metadata_df["attack_active"] == 1)[0]
        if len(attack_idx) > 0:
            fig.add_vrect(
                x0=int(attack_idx[0]),
                x1=int(attack_idx[-1]),
                fillcolor="rgba(239, 68, 68, 0.15)",
                layer="below",
                line_width=1,
                line_color="#ef4444",
                annotation_text="Ground Truth Attack Window",
            )

    fig.update_layout(
        **DARK_LAYOUT,
        title="Stage 1: XGBoost Attack Probability vs Decision Thresholds",
        xaxis_title="Simulation Timestep (seconds)",
        yaxis_title="Predicted Probability P(Attack)",
        height=380,
    )
    fig.update_yaxes(range=[-0.02, max(0.6, float(np.max(probs)) + 0.1)], gridcolor="#374151")
    return fig


def plot_stage2_multiclass(stage2_dict: Dict[str, Any], metadata_df: pd.DataFrame) -> go.Figure:
    """Plots Stage 2 Random Forest multiclass probabilities across the 4 classes."""
    fig = go.Figure()
    steps = metadata_df["step"].values
    probs = stage2_dict["class_probabilities"]
    class_names = stage2_dict["class_names"]
    colors = ["#34d399", "#f87171", "#fbbf24", "#818cf8"]

    for c_idx, (name, color) in enumerate(zip(class_names, colors)):
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=probs[:, c_idx],
                mode="lines",
                name=f"{name} (Class {c_idx})",
                line=dict(color=color, width=2),
                stackgroup="one",
            )
        )

    fig.update_layout(
        **DARK_LAYOUT,
        title="Stage 2: Random Forest Multiclass Probability Distribution",
        xaxis_title="Simulation Timestep (seconds)",
        yaxis_title="Class Probability",
        height=360,
    )
    fig.update_yaxes(range=[0, 1.05], gridcolor="#374151")
    return fig


def plot_stage3_autoencoder(stage3_dict: Dict[str, Any], metadata_df: pd.DataFrame) -> go.Figure:
    """Plots Stage 3 Autoencoder Reconstruction MSE against frozen anomaly thresholds."""
    fig = go.Figure()
    steps = metadata_df["step"].values
    mse = stage3_dict["reconstruction_mse"]

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mse,
            mode="lines",
            name="Reconstruction MSE",
            line=dict(color="#a78bfa", width=2.5),
        )
    )

    # Threshold horizontal lines
    fig.add_hline(
        y=stage3_dict["threshold_recovered"],
        line_dash="dash",
        line_color="#ef4444",
        line_width=2,
        annotation_text="Recovered Threshold tau = 0.3640",
    )

    # Ground truth attack active window
    if "attack_active" in metadata_df.columns and metadata_df["attack_active"].sum() > 0:
        attack_idx = np.where(metadata_df["attack_active"] == 1)[0]
        if len(attack_idx) > 0:
            fig.add_vrect(
                x0=int(attack_idx[0]),
                x1=int(attack_idx[-1]),
                fillcolor="rgba(239, 68, 68, 0.15)",
                layer="below",
                line_width=1,
                line_color="#ef4444",
                annotation_text="Ground Truth Attack Window",
            )

    fig.update_layout(
        **DARK_LAYOUT,
        title="Stage 3: Deep Autoencoder Reconstruction MSE (150 Features)",
        xaxis_title="Simulation Timestep (seconds)",
        yaxis_title="Reconstruction Error (MSE)",
        height=380,
    )
    return fig


def plot_confusion_matrix(cm: Dict[str, int], title: str = "Confusion Matrix") -> go.Figure:
    """Plots a 2x2 confusion matrix heatmap."""
    z = [
        [cm.get("tn", 0), cm.get("fp", 0)],
        [cm.get("fn", 0), cm.get("tp", 0)],
    ]
    x = ["Predicted Normal (0)", "Predicted Attack (1)"]
    y = ["Actual Normal (0)", "Actual Attack (1)"]

    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    x=x[j],
                    y=y[i],
                    text=f"{z[i][j]:,d}",
                    font=dict(color="#f8fafc", size=16, family="sans-serif"),
                    showarrow=False,
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[[0, "#1e293b"], [0.5, "#2563eb"], [1, "#059669"]],
            showscale=False,
        )
    )
    layout = dict(DARK_LAYOUT)
    layout.update(
        title=title,
        annotations=annotations,
        height=260,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    fig.update_layout(layout)
    return fig


# -----------------------------------------------------------------------------
# Tab 4: Research Analysis Visualizations
# -----------------------------------------------------------------------------
def plot_domain_shift_comparison() -> go.Figure:
    """Plots comparative metrics for Tab 4: OpenDSS Conditions C0 through C3."""
    conditions = ["C0 (Baseline)", "C1 (Voltage Align)", "C2 (Reactive Align)", "C3 (Combined Align)"]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Autoencoder Mean MSE (Lower is Closer)",
            "Stage 1 Max Probability (tau=0.50)",
            "Stage 2 Multiclass Accuracy (%)",
        ),
    )

    # 1. Autoencoder MSE
    fig.add_trace(
        go.Bar(
            x=conditions,
            y=[320.72, 322.75, 2.62, 1.36],
            marker_color=["#ef4444", "#ef4444", "#fbbf24", "#34d399"],
            name="Mean MSE",
            text=[f"{v:.2f}" for v in [320.72, 322.75, 2.62, 1.36]],
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    # 2. Stage 1 Max Probability
    fig.add_trace(
        go.Bar(
            x=conditions,
            y=[0.0447, 0.0585, 0.0545, 0.0722],
            marker_color="#38bdf8",
            name="Max Prob",
            text=[f"{v:.4f}" for v in [0.0447, 0.0585, 0.0545, 0.0722]],
            textposition="auto",
        ),
        row=1,
        col=2,
    )

    # 3. Stage 2 Multiclass Accuracy
    fig.add_trace(
        go.Bar(
            x=conditions,
            y=[77.67, 12.94, 76.03, 12.94],
            marker_color=["#34d399", "#ef4444", "#34d399", "#ef4444"],
            name="Accuracy (%)",
            text=[f"{v:.1f}%" for v in [77.67, 12.94, 76.03, 12.94]],
            textposition="auto",
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title="Cross-Domain Performance Across OpenDSS Operating Conditions (Milestone 4.5 Audit)",
        showlegend=False,
        height=380,
    )
    return fig
