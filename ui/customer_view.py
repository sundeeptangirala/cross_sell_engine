"""
customer_view.py
------------------
Customer View — Section 10.1 of the spec.

Layout:
    Header: Customer search + as-of date
    Left column:  Customer snapshot → Detected products (card grid) → Opportunity notes
    Right column: Explainability → Evidence table → Pattern charts → Confidence breakdown
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_customer_view(detections: pd.DataFrame, transactions: pd.DataFrame):
    """Renders the full Customer View page."""

    # --- Header ---
    st.markdown("""
        <div class="main-header">
            <h1>📋 Customer View</h1>
            <p>Search a customer to inspect detected external financial relationships</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Customer Search ---
    col_search, col_asof = st.columns([2, 1])

    with col_search:
        customer_input = st.text_input(
            "Customer ID",
            placeholder="e.g. CUST000001",
            key="customer_search_input",
        )

    with col_asof:
        # Quick-jump: dropdown of customers with detections
        available_customers = sorted(detections["customer_id"].unique()) if not detections.empty else []
        quick_select = st.selectbox(
            "Or quick-select",
            options=["— Pick a customer —"] + available_customers[:200],
            index=0,
            key="customer_quick_select",
        )

    # Resolve which customer to show
    customer_id = None
    if customer_input and customer_input.strip():
        customer_id = customer_input.strip().upper()
    elif quick_select != "— Pick a customer —":
        customer_id = quick_select

    if customer_id is None:
        st.info("👆 Enter a Customer ID or select one from the dropdown to get started.")
        return

    # --- Validate customer exists ---
    cust_txns = transactions[transactions["customer_id"] == customer_id]
    cust_detections = detections[detections["customer_id"] == customer_id]

    if cust_txns.empty:
        st.warning(f"No transactions found for **{customer_id}**. Check the ID and try again.")
        return

    # --- Two-column layout ---
    left, right = st.columns([1, 1.2], gap="medium")

    # =========================================================================
    # LEFT COLUMN
    # =========================================================================
    with left:
        # --- Customer Snapshot Card ---
        _render_customer_snapshot(customer_id, cust_txns, cust_detections)

        # --- Detected Products ---
        st.markdown('<div class="section-title">Detected External Products</div>', unsafe_allow_html=True)

        if cust_detections.empty:
            st.markdown("""
                <div style="background:#f0f4f8; border-radius:8px; padding:20px; text-align:center; color:#7f8c8d;">
                    No external relationships detected for this customer.
                </div>
            """, unsafe_allow_html=True)
        else:
            # Track which product card is selected for evidence view
            selected_product_idx = st.session_state.get("selected_product_idx", 0)

            for idx, row in cust_detections.iterrows():
                actual_idx = list(cust_detections.index).index(idx)
                is_selected = (actual_idx == selected_product_idx)
                _render_product_card(row, actual_idx, is_selected)

        # --- Opportunity Notes ---
        st.markdown('<div class="section-title" style="margin-top:20px;">Opportunity Notes</div>', unsafe_allow_html=True)
        st.markdown("""
            <div style="background:#f8fafc; border-radius:8px; padding:14px 16px; font-size:13px; color:#34495e; line-height:1.6;">
                <b>💡 Note:</b> These are observed patterns only — no offers or recommendations are generated at this layer.
                Downstream systems decide activation based on eligibility and campaign logic.
            </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # RIGHT COLUMN
    # =========================================================================
    with right:
        if cust_detections.empty:
            st.markdown("""
                <div style="background:#f0f4f8; border-radius:8px; padding:30px; text-align:center; color:#7f8c8d; margin-top:60px;">
                    Select a customer with detected relationships to see details here.
                </div>
            """, unsafe_allow_html=True)
            return

        # Get the selected detection row
        selected_idx = st.session_state.get("selected_product_idx", 0)
        selected_idx = min(selected_idx, len(cust_detections) - 1)
        selected_row = cust_detections.iloc[selected_idx]

        # --- Why We Think This (Explainability) ---
        st.markdown('<div class="section-title">Why We Think This</div>', unsafe_allow_html=True)
        _render_explainability(selected_row)

        # --- Evidence Table ---
        st.markdown('<div class="section-title" style="margin-top:18px;">Evidence Transactions</div>', unsafe_allow_html=True)
        _render_evidence_table(selected_row, transactions)

        # --- Pattern Chart ---
        st.markdown('<div class="section-title" style="margin-top:18px;">Payment Pattern</div>', unsafe_allow_html=True)
        _render_pattern_chart(selected_row, transactions)

        # --- Confidence Breakdown ---
        st.markdown('<div class="section-title" style="margin-top:18px;">Confidence Breakdown</div>', unsafe_allow_html=True)
        _render_confidence_breakdown(selected_row)


# =============================================================================
# HELPER RENDERERS
# =============================================================================

def _render_customer_snapshot(customer_id: str, cust_txns: pd.DataFrame, cust_detections: pd.DataFrame):
    """Renders the customer snapshot card."""
    first_txn = cust_txns["transaction_date"].min()
    last_txn = cust_txns["transaction_date"].max()
    n_detections = len(cust_detections)
    top_categories = cust_txns["category"].value_counts().head(3).index.tolist()

    st.markdown(f"""
        <div style="background:white; border-radius:10px; padding:16px 18px; box-shadow:0 2px 8px rgba(0,0,0,0.06);
                    border:1px solid #edf1f4; margin-bottom:18px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <div>
                    <div style="font-size:18px; font-weight:700; color:#1a2332;">{customer_id}</div>
                    <div style="font-size:12px; color:#7f8c8d;">{len(cust_txns):,} transactions on file</div>
                </div>
                <div style="background:#eef2f7; padding:6px 14px; border-radius:20px; font-size:13px; font-weight:600; color:#2c3e50;">
                    {n_detections} external relationship{"s" if n_detections != 1 else ""} detected
                </div>
            </div>
            <div style="display:flex; gap:24px; font-size:12px; color:#5a6a7a;">
                <div><b style="color:#2c3e50;">Active Since</b><br>{first_txn.strftime('%b %d, %Y')}</div>
                <div><b style="color:#2c3e50;">Last Activity</b><br>{last_txn.strftime('%b %d, %Y')}</div>
                <div><b style="color:#2c3e50;">Top Categories</b><br>{', '.join(top_categories)}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def _render_product_card(row: pd.Series, idx: int, is_selected: bool):
    """Renders a single detected product card."""
    tier = row["confidence_tier"]
    badge_class = f"badge-{tier.lower()}"
    border_style = "border: 2px solid #3498db;" if is_selected else "border: 1px solid #edf1f4;"

    # Click handler via button
    clicked = st.button(
        f"**{row['detected_product_type']}** · {row['recurring_amount_band']}/mo",
        key=f"product_card_{idx}",
        use_container_width=True,
        help=f"Click to view evidence for {row['detected_product_type']}",
    )
    if clicked:
        st.session_state["selected_product_idx"] = idx
        st.rerun()

    st.markdown(f"""
        <div style="background:white; border-radius:8px; padding:10px 14px; margin-top:-8px; margin-bottom:10px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.05); {border_style}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="badge {badge_class}">{tier}</span>
                <span style="font-size:11px; color:#7f8c8d;">Score: {row['confidence_score']:.2f}</span>
            </div>
            <div style="font-size:12px; color:#5a6a7a; margin-top:6px; line-height:1.5;">
                <b>Merchant:</b> {row['canonical_merchant']}<br>
                <b>Since:</b> {row['first_detected_date']} · <b>Tenure:</b> {row['tenure_months']:.1f} mo<br>
                <b>Channel:</b> {row['dominant_channel']}
            </div>
        </div>
    """, unsafe_allow_html=True)


def _render_explainability(row: pd.Series):
    """Renders the 'Why We Think This' reason codes panel."""
    reason_codes = str(row.get("explanation_reason_codes", "")).split(" | ")
    pills_html = "".join(
        f'<span class="reason-pill">{code.strip()}</span>'
        for code in reason_codes if code.strip()
    )
    st.markdown(f"""
        <div style="background:#f8fafc; border-radius:8px; padding:14px 16px; margin-bottom:4px;">
            {pills_html}
        </div>
    """, unsafe_allow_html=True)


def _render_evidence_table(row: pd.Series, transactions: pd.DataFrame):
    """Renders the evidence transaction table, filtered to the selected detection's transactions."""
    txn_refs = str(row.get("evidence_transaction_refs", ""))
    if not txn_refs:
        st.info("No evidence transaction references available.")
        return

    try:
        txn_ids = [int(x) for x in txn_refs.split("|") if x.strip()]
    except ValueError:
        st.info("Could not parse transaction references.")
        return

    evidence_txns = transactions[transactions["transaction_id"].isin(txn_ids)].copy()

    if evidence_txns.empty:
        st.info("Evidence transactions not found in the current dataset.")
        return

    # Format for display
    display_df = evidence_txns[["transaction_id", "transaction_date", "amount", "channel", "cleansed_description"]].copy()
    display_df = display_df.sort_values("transaction_date", ascending=False).reset_index(drop=True)
    display_df["transaction_date"] = display_df["transaction_date"].dt.strftime("%Y-%m-%d")
    display_df["amount"] = display_df["amount"].apply(lambda x: f"${x:,.2f}")
    display_df.columns = ["Txn ID", "Date", "Amount", "Channel", "Merchant"]

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def _render_pattern_chart(row: pd.Series, transactions: pd.DataFrame):
    """Renders an amount-over-time line chart for the selected detection."""
    txn_refs = str(row.get("evidence_transaction_refs", ""))
    if not txn_refs:
        return

    try:
        txn_ids = [int(x) for x in txn_refs.split("|") if x.strip()]
    except ValueError:
        return

    evidence_txns = transactions[transactions["transaction_id"].isin(txn_ids)].copy()
    if evidence_txns.empty or len(evidence_txns) < 2:
        return

    evidence_txns = evidence_txns.sort_values("transaction_date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=evidence_txns["transaction_date"],
        y=evidence_txns["amount"],
        mode="lines+markers",
        line=dict(color="#3498db", width=2.5),
        marker=dict(size=7, color="#3498db", line=dict(width=2, color="white")),
        name="Payment Amount",
    ))

    # Mean line
    mean_amt = evidence_txns["amount"].mean()
    fig.add_hline(
        y=mean_amt, line_dash="dash", line_color="#e74c3c", line_width=1.5,
        annotation_text=f"Mean: ${mean_amt:,.0f}",
        annotation_position="top right",
        annotation_font_size=11,
    )

    fig.update_layout(
        height=220,
        margin=dict(l=40, r=20, t=10, b=30),
        plot_bgcolor="white",
        paper_bgcolor="#f8fafc",
        xaxis=dict(showgrid=False, title_text="", tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#edf1f4", title_text="", tickprefix="$", tickfont=dict(size=10)),
        hovermode="x unified",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_confidence_breakdown(row: pd.Series):
    """Renders a signal checklist showing which factors contributed to confidence."""
    checks = []

    # Cadence
    codes_str = str(row.get("explanation_reason_codes", ""))
    checks.append(("Monthly Cadence", "CADENCE_MONTHLY" in codes_str))
    checks.append(("Amount Stability", "AMOUNT_STABLE" in codes_str or "AMOUNT_VARIABLE" in codes_str))
    checks.append(("Sufficient Tenure", f"{row['tenure_months']:.1f} mo" if row["tenure_months"] >= 2 else "< 2 months"))
    checks.append(("Taxonomy Match", "TAXONOMY_MATCH" in codes_str))
    checks.append(("Channel Alignment", "CHANNEL_MATCH" in codes_str))

    rows_html = ""
    for label, value in checks:
        if value is True or (isinstance(value, str) and value):
            icon = "✅"
            color = "#27ae60"
            val_text = value if isinstance(value, str) else "Matched"
        else:
            icon = "⚪"
            color = "#95a5a6"
            val_text = "—"

        rows_html += f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:6px 0; border-bottom:1px solid #f0f0f0;">
                <div style="font-size:13px; color:#2c3e50;">{icon} {label}</div>
                <div style="font-size:12px; color:{color}; font-weight:500;">{val_text}</div>
            </div>
        """

    st.markdown(f"""
        <div style="background:#f8fafc; border-radius:8px; padding:12px 16px;">
            {rows_html}
            <div style="display:flex; justify-content:space-between; padding-top:8px; margin-top:4px;">
                <div style="font-size:12px; font-weight:600; color:#1a2332;">Composite Score</div>
                <div style="font-size:14px; font-weight:700; color:#3498db;">{row['confidence_score']:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
