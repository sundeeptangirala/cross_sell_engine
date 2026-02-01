"""
tuning_view.py
---------------
Tuning & QA View — Section 10.3 of the spec. DS/Ops only.

Layout:
    Header + access gate note
    Tab 1: Unclassified Recurring Patterns
    Tab 2: Sampling & Feedback
    Tab 3: Drift Monitors
    Tab 4: Config Viewer (taxonomy + thresholds)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.config_loader import load_config, get_merchant_taxonomy
from monitoring.drift_monitor import DriftMonitor


def render_tuning_view(detections: pd.DataFrame, transactions: pd.DataFrame, rpo_objects: list):
    """Renders the full Tuning & QA page."""

    # --- Header ---
    st.markdown("""
        <div class="main-header" style="border: 1px solid rgba(255,255,255,0.15);">
            <h1>⚙️ Tuning & QA <span style="font-size:13px; background:rgba(231,76,60,0.3); padding:3px 10px; border-radius:10px; margin-left:10px;">DS / Ops Only</span></h1>
            <p>Unclassified patterns, reviewer feedback, drift monitors, and live config</p>
        </div>
    """, unsafe_allow_html=True)

    # Access gate note
    st.markdown("""
        <div style="background:#fef9e7; border-left:4px solid #e67e22; padding:10px 16px; border-radius:0 8px 8px 0; margin-bottom:20px; font-size:13px; color:#856404;">
            🔒 <b>Gated View:</b> This page is intended for Data Science and Operations teams only.
            In production, access should be restricted via role-based permissions.
        </div>
    """, unsafe_allow_html=True)

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Unclassified Patterns",
        "📝 Sampling & Feedback",
        "📈 Drift Monitors",
        "📄 Config Viewer",
    ])

    # Precompute: which RPOs were NOT classified into any detection
    detected_keys = set()
    if not detections.empty:
        for _, row in detections.iterrows():
            detected_keys.add((row["customer_id"], row["canonical_merchant"]))

    unclassified_rpos = [
        rpo for rpo in rpo_objects
        if (rpo.customer_id, rpo.merchant_canonical) not in detected_keys
    ]

    # =========================================================================
    # TAB 1: Unclassified Recurring Patterns
    # =========================================================================
    with tab1:
        _render_unclassified_patterns(unclassified_rpos)

    # =========================================================================
    # TAB 2: Sampling & Feedback
    # =========================================================================
    with tab2:
        _render_sampling_feedback(detections)

    # =========================================================================
    # TAB 3: Drift Monitors
    # =========================================================================
    with tab3:
        _render_drift_monitors(detections, transactions)

    # =========================================================================
    # TAB 4: Config Viewer
    # =========================================================================
    with tab4:
        _render_config_viewer()


# =============================================================================
# TAB 1: UNCLASSIFIED PATTERNS
# =============================================================================

def _render_unclassified_patterns(unclassified_rpos: list):
    """
    Shows recurring payment patterns that didn't map to any product type.
    Ranked by customer count equivalent (occurrence_count) and dollar volume.
    These are candidates for new product rules or taxonomy updates.
    """
    st.markdown("""
        <div style="font-size:13px; color:#5a6a7a; margin-bottom:16px; line-height:1.5;">
            Recurring payments detected by the engine that did <b>not</b> match any product interpreter.
            These may represent taxonomy gaps, new product types, or noise. Review weekly.
        </div>
    """, unsafe_allow_html=True)

    if not unclassified_rpos:
        st.success("✅ No unclassified recurring patterns. All detected patterns mapped successfully.")
        return

    # Build a summary DataFrame grouped by merchant
    rows = []
    for rpo in unclassified_rpos:
        rows.append({
            "merchant": rpo.merchant_canonical,
            "category": rpo.category,
            "mcc_code": rpo.mcc_code,
            "cadence_type": rpo.cadence_type,
            "mean_amount": rpo.mean_amount,
            "occurrence_count": rpo.occurrence_count,
            "tenure_months": rpo.tenure_months,
            "cadence_strength": rpo.cadence_strength,
            "customer_id": rpo.customer_id,
        })

    df = pd.DataFrame(rows)

    # Aggregate by merchant: count unique customers, sum volume
    merchant_summary = df.groupby(["merchant", "category", "mcc_code"]).agg(
        customer_count=("customer_id", "nunique"),
        avg_amount=("mean_amount", "mean"),
        total_monthly_volume=("mean_amount", "sum"),
        avg_cadence_strength=("cadence_strength", "mean"),
        avg_tenure_months=("tenure_months", "mean"),
    ).reset_index()

    merchant_summary = merchant_summary.sort_values("total_monthly_volume", ascending=False).reset_index(drop=True)

    # KPI summary
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        st.markdown(f"""
            <div class="kpi-card red">
                <div class="kpi-value">{len(unclassified_rpos):,}</div>
                <div class="kpi-label">Unclassified Patterns</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="kpi-card orange">
                <div class="kpi-value">{merchant_summary['customer_count'].sum():,}</div>
                <div class="kpi-label">Unique Customers Affected</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="kpi-card purple">
                <div class="kpi-value">${merchant_summary['total_monthly_volume'].sum():,.0f}</div>
                <div class="kpi-label">Total Monthly Volume</div>
            </div>
        """, unsafe_allow_html=True)

    # Format and display
    display_df = merchant_summary.copy()
    display_df["avg_amount"] = display_df["avg_amount"].apply(lambda x: f"${x:,.2f}")
    display_df["total_monthly_volume"] = display_df["total_monthly_volume"].apply(lambda x: f"${x:,.0f}")
    display_df["avg_cadence_strength"] = display_df["avg_cadence_strength"].apply(lambda x: f"{x:.2f}")
    display_df["avg_tenure_months"] = display_df["avg_tenure_months"].apply(lambda x: f"{x:.1f} mo")
    display_df.columns = [
        "Merchant", "Category", "MCC Code",
        "Customers", "Avg Amount", "Monthly Volume",
        "Cadence Strength", "Avg Tenure"
    ]

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 2: SAMPLING & FEEDBACK
# =============================================================================

def _render_sampling_feedback(detections: pd.DataFrame):
    """
    Sampling interface: randomly sample detections for manual review.
    Reviewers label them as Correct / Incorrect / Ambiguous.
    Feedback is stored in session state (in production, this would write to a DB).
    """
    st.markdown("""
        <div style="font-size:13px; color:#5a6a7a; margin-bottom:16px; line-height:1.5;">
            Randomly sample detections for manual review. Use labels to flag misclassifications
            that can be fed back to recalibrate thresholds and taxonomy mappings.
        </div>
    """, unsafe_allow_html=True)

    if detections.empty:
        st.warning("No detections available to sample.")
        return

    # Sample size selector
    sample_size = st.slider("Sample Size", min_value=5, max_value=min(50, len(detections)), value=10, step=5)

    # Initialize or re-sample
    if "feedback_sample" not in st.session_state or st.button("🔄 New Sample", key="resample_btn"):
        st.session_state["feedback_sample"] = detections.sample(n=sample_size).reset_index(drop=True)
        st.session_state["feedback_labels"] = {}

    sample_df = st.session_state["feedback_sample"]
    labels = st.session_state["feedback_labels"]

    # Render each sample row
    for idx, row in sample_df.iterrows():
        col_info, col_label = st.columns([3, 1], gap="small")

        with col_info:
            tier_badge = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(row["confidence_tier"], "")
            st.markdown(f"""
                <div style="background:white; border-radius:8px; padding:10px 14px;
                            box-shadow:0 1px 4px rgba(0,0,0,0.06); border:1px solid #edf1f4;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b style="color:#1a2332; font-size:13px;">{row['detected_product_type']}</b>
                        <span class="badge {tier_badge}">{row['confidence_tier']} · {row['confidence_score']:.2f}</span>
                    </div>
                    <div style="font-size:11.5px; color:#5a6a7a; margin-top:4px;">
                        {row['customer_id']} · {row['canonical_merchant']} · {row['recurring_amount_band']}/mo · {row['tenure_months']:.1f} mo tenure
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_label:
            current_label = labels.get(idx, "— Review —")
            label_choice = st.selectbox(
                "",
                options=["— Review —", "✅ Correct", "❌ Incorrect", "⚠️ Ambiguous"],
                index=["— Review —", "✅ Correct", "❌ Incorrect", "⚠️ Ambiguous"].index(current_label) if current_label in ["— Review —", "✅ Correct", "❌ Incorrect", "⚠️ Ambiguous"] else 0,
                key=f"feedback_label_{idx}",
                label_visibility="hidden",
            )
            if label_choice != "— Review —":
                labels[idx] = label_choice

    # Summary of labels
    if labels:
        st.markdown('<div class="section-title" style="margin-top:24px;">Feedback Summary</div>', unsafe_allow_html=True)
        label_counts = pd.Series(labels.values()).value_counts()
        cols = st.columns(3)
        for col, (label, count) in zip(cols, [
            ("✅ Correct", label_counts.get("✅ Correct", 0)),
            ("❌ Incorrect", label_counts.get("❌ Incorrect", 0)),
            ("⚠️ Ambiguous", label_counts.get("⚠️ Ambiguous", 0)),
        ]):
            with col:
                st.metric(label, str(count))

        # Export button
        if st.button("📥 Export Feedback", key="export_feedback"):
            feedback_rows = []
            for idx, label in labels.items():
                row = sample_df.iloc[idx]
                feedback_rows.append({
                    "customer_id": row["customer_id"],
                    "detected_product_type": row["detected_product_type"],
                    "confidence_tier": row["confidence_tier"],
                    "confidence_score": row["confidence_score"],
                    "canonical_merchant": row["canonical_merchant"],
                    "reviewer_label": label,
                })
            feedback_df = pd.DataFrame(feedback_rows)
            st.download_button(
                label="💾 Download Feedback CSV",
                data=feedback_df.to_csv(index=False),
                file_name="feedback_labels.csv",
                mime="text/csv",
                key="download_feedback",
            )


# =============================================================================
# TAB 3: DRIFT MONITORS
# =============================================================================

def _render_drift_monitors(detections: pd.DataFrame, transactions: pd.DataFrame):
    """Runs drift monitoring and displays results with severity-coded alerts."""
    st.markdown("""
        <div style="font-size:13px; color:#5a6a7a; margin-bottom:16px; line-height:1.5;">
            Monitors detection volume, amount distribution shifts (KS test + PSI),
            and unknown merchant ratios. Alerts are severity-coded.
        </div>
    """, unsafe_allow_html=True)

    # Run monitor
    monitor = DriftMonitor()
    report = monitor.run(detections, transactions)

    # Summary KPIs
    col1, col2, col3, col4 = st.columns(4, gap="small")
    summary = report.summary

    with col1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{summary['total_alerts']}</div>
                <div class="kpi-label">Total Alerts</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="kpi-card red">
                <div class="kpi-value">{summary['critical_alerts']}</div>
                <div class="kpi-label">Critical</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="kpi-card orange">
                <div class="kpi-value">{summary['warning_alerts']}</div>
                <div class="kpi-label">Warnings</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{summary['info_alerts']}</div>
                <div class="kpi-label">Info</div>
            </div>
        """, unsafe_allow_html=True)

    # Alerts
    if report.alerts:
        st.markdown('<div class="section-title" style="margin-top:20px;">Alerts</div>', unsafe_allow_html=True)
        for alert in report.alerts:
            css_class = {
                "CRITICAL": "alert-critical",
                "WARNING": "alert-warning",
                "INFO": "alert-info",
            }.get(alert.severity, "alert-info")

            icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(alert.severity, "⚪")

            st.markdown(f"""
                <div class="{css_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <b style="font-size:13px;">{icon} [{alert.alert_type}] {alert.severity}</b>
                        <span style="font-size:10px; color:#7f8c8d;">{alert.product_type}</span>
                    </div>
                    <div style="font-size:12.5px;">{alert.message}</div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No drift alerts detected. All monitors within normal thresholds.")

    # Report metadata
    st.markdown(f"""
        <div style="font-size:11px; color:#95a5a6; margin-top:16px; padding-top:12px; border-top:1px solid #edf1f4;">
            Report generated: {report.run_timestamp} | Baseline: {report.baseline_window} | Comparison: {report.comparison_window}
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 4: CONFIG VIEWER
# =============================================================================

def _render_config_viewer():
    """Read-only view of live taxonomy and interpreter thresholds from config.yaml."""
    st.markdown("""
        <div style="font-size:13px; color:#5a6a7a; margin-bottom:16px; line-height:1.5;">
            Live view of configuration. Changes are made in <b>config/config.yaml</b> — not here.
            This view updates automatically when config changes.
        </div>
    """, unsafe_allow_html=True)

    config = load_config()

    # --- Sub-tabs ---
    cfg_tab1, cfg_tab2 = st.tabs(["🗺️ Merchant Taxonomy", "⚖️ Interpreter Thresholds"])

    with cfg_tab1:
        taxonomy = get_merchant_taxonomy()
        tax_df = pd.DataFrame(taxonomy)
        tax_df.columns = ["MCC Code", "Category", "Product Type", "Match Confidence"]
        tax_df = tax_df.sort_values(["Product Type", "Category"]).reset_index(drop=True)
        st.dataframe(tax_df, use_container_width=True, hide_index=True)

    with cfg_tab2:
        interpreters_config = config["product_interpreters"]

        rows = []
        for product, cfg in interpreters_config.items():
            rows.append({
                "Product Type": product,
                "Cadence": cfg["expected_cadence"],
                "Min Amount": f"${cfg['min_amount']:,.0f}",
                "Max Amount": f"${cfg['max_amount']:,.0f}",
                "Max CV": f"{cfg['max_amount_cv']:.2f}",
                "Min Tenure (mo)": cfg["min_tenure_months"],
                "Expected Channels": ", ".join(cfg["expected_channels"]),
                "Channel Weight": f"{cfg['channel_weight']:.2f}",
            })

        threshold_df = pd.DataFrame(rows)
        st.dataframe(threshold_df, use_container_width=True, hide_index=True)

    # --- Confidence Tier Boundaries ---
    st.markdown('<div class="section-title" style="margin-top:24px;">Confidence Tier Boundaries</div>', unsafe_allow_html=True)
    tiers = config["confidence_tiers"]
    tier_rows = [
        {"Tier": name, "Min Score": f"{bounds['min_score']:.2f}", "Max Score": f"{bounds['max_score']:.2f}"}
        for name, bounds in tiers.items()
    ]
    st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)
