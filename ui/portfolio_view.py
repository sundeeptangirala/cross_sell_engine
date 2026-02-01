"""
portfolio_view.py
-------------------
Portfolio View — Section 10.2 of the spec.

Layout:
    Header
    KPI row (4 cards)
    Charts row: Product gap distribution | Top external servicers/issuers
    Detection trends over time
    Drill-down table (filterable)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Color palette — consistent across all charts
PRODUCT_COLORS = {
    "External Mortgage":     "#3498db",
    "External Auto Loan":    "#e67e22",
    "External Credit Card":  "#27ae60",
    "Student Loan":          "#9b59b6",
    "Rent":                  "#e74c3c",
    "Insurance":             "#1abc9c",
}


def render_portfolio_view(detections: pd.DataFrame, transactions: pd.DataFrame):
    """Renders the full Portfolio View page."""

    # --- Header ---
    st.markdown("""
        <div class="main-header">
            <h1>📊 Portfolio View</h1>
            <p>Aggregate view of detected external relationships across all customers</p>
        </div>
    """, unsafe_allow_html=True)

    if detections.empty:
        st.warning("No detections available. Check your confidence filter or time window.")
        return

    # --- KPI Row ---
    _render_kpis(detections, transactions)

    # --- Charts Row ---
    col_gap, col_top = st.columns([1, 1], gap="medium")

    with col_gap:
        _render_product_gap_distribution(detections)

    with col_top:
        _render_top_servicers(detections)

    # --- Detection Trends ---
    st.markdown('<div class="section-title" style="margin-top:28px;">Detection Trends Over Time</div>', unsafe_allow_html=True)
    _render_detection_trends(detections)

    # --- Drill-Down Table ---
    st.markdown('<div class="section-title" style="margin-top:28px;">Drill-Down Table</div>', unsafe_allow_html=True)
    _render_drilldown_table(detections)


# =============================================================================
# KPIs
# =============================================================================

def _render_kpis(detections: pd.DataFrame, transactions: pd.DataFrame):
    """Renders the 4 KPI cards."""
    total_customers = transactions["customer_id"].nunique()
    customers_with_detections = detections["customer_id"].nunique()
    high_confidence_mortgage = len(detections[
        (detections["detected_product_type"] == "External Mortgage") &
        (detections["confidence_tier"] == "High")
    ])
    high_confidence_auto = len(detections[
        (detections["detected_product_type"] == "External Auto Loan") &
        (detections["confidence_tier"] == "High")
    ])
    high_confidence_cc = len(detections[
        (detections["detected_product_type"] == "External Credit Card") &
        (detections["confidence_tier"] == "High")
    ])
    unknown_ratio = 0.0  # Placeholder — would be computed from taxonomy gaps

    cols = st.columns(4, gap="small")

    kpis = [
        ("Customers with External Relationships", f"{customers_with_detections:,}",
         f"of {total_customers:,} total ({customers_with_detections/total_customers*100:.1f}%)", ""),
        ("External Mortgages (High)", f"{high_confidence_mortgage:,}",
         "High-confidence detections", "green"),
        ("External Auto Loans (High)", f"{high_confidence_auto:,}",
         "High-confidence detections", "orange"),
        ("External Credit Cards (High)", f"{high_confidence_cc:,}",
         "High-confidence detections", "purple"),
    ]

    for col, (label, value, sub, color_class) in zip(cols, kpis):
        with col:
            st.markdown(f"""
                <div class="kpi-card {color_class}">
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-label">{label}</div>
                    <div style="font-size:11px; color:#95a5a6; margin-top:4px;">{sub}</div>
                </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PRODUCT GAP DISTRIBUTION
# =============================================================================

def _render_product_gap_distribution(detections: pd.DataFrame):
    """Horizontal bar chart of detection counts by product type."""
    st.markdown('<div class="section-title">Product Gap Distribution</div>', unsafe_allow_html=True)

    product_counts = detections["detected_product_type"].value_counts()
    products = product_counts.index.tolist()
    counts = product_counts.values.tolist()
    colors = [PRODUCT_COLORS.get(p, "#95a5a6") for p in products]

    fig = go.Figure(go.Bar(
        x=counts,
        y=products,
        orientation="h",
        marker_color=colors,
        text=[f"{c:,}" for c in counts],
        textposition="outside",
        textfont=dict(size=12, color="#2c3e50"),
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=140, r=50, t=10, b=20),
        plot_bgcolor="white",
        paper_bgcolor="#f8fafc",
        xaxis=dict(showgrid=True, gridcolor="#edf1f4", title_text="", tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, title_text="", tickfont=dict(size=12, color="#2c3e50")),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# TOP SERVICERS / ISSUERS
# =============================================================================

def _render_top_servicers(detections: pd.DataFrame):
    """Pie chart of top external servicers/issuers by detection count."""
    st.markdown('<div class="section-title">Top External Servicers / Issuers</div>', unsafe_allow_html=True)

    top_merchants = detections["canonical_merchant"].value_counts().head(8)

    fig = go.Figure(go.Pie(
        labels=top_merchants.index.tolist(),
        values=top_merchants.values.tolist(),
        hole=0.45,
        marker_colors=["#3498db", "#e67e22", "#27ae60", "#9b59b6",
                        "#e74c3c", "#1abc9c", "#f39c12", "#2980b9"],
        textinfo="label+percent",
        textfont=dict(size=11),
        hoverinfo="label+value+percent",
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#f8fafc",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# DETECTION TRENDS OVER TIME
# =============================================================================

def _render_detection_trends(detections: pd.DataFrame):
    """Line chart showing detection volume over time, stacked by product type."""
    df = detections.copy()
    df["last_detected_date"] = pd.to_datetime(df["last_detected_date"])
    df["month"] = df["last_detected_date"].dt.to_period("M").astype(str)

    # Pivot: rows = months, columns = product types
    pivot = df.groupby(["month", "detected_product_type"]).size().unstack(fill_value=0)
    pivot = pivot.sort_index()

    fig = go.Figure()

    for product in pivot.columns:
        fig.add_trace(go.Scatter(
            x=pivot.index.tolist(),
            y=pivot[product].values,
            name=product,
            mode="lines+markers",
            line=dict(color=PRODUCT_COLORS.get(product, "#95a5a6"), width=2.5),
            marker=dict(size=5),
            stackgroup="one",
            fillcolor=PRODUCT_COLORS.get(product, "#95a5a6"),
            opacity=0.3,
        ))

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=10, b=40),
        plot_bgcolor="white",
        paper_bgcolor="#f8fafc",
        xaxis=dict(showgrid=False, title_text="", tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#edf1f4", title_text="Detections", title_font=dict(size=11), tickfont=dict(size=10)),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.25,
            xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# DRILL-DOWN TABLE
# =============================================================================

def _render_drilldown_table(detections: pd.DataFrame):
    """Filterable drill-down table with all detections."""

    # Filter controls
    col_product, col_tier, col_search = st.columns([2, 1.5, 2], gap="small")

    with col_product:
        product_filter = st.multiselect(
            "Product Type",
            options=sorted(detections["detected_product_type"].unique()),
            default=[],
            placeholder="All products",
            key="portfolio_product_filter",
        )

    with col_tier:
        tier_filter = st.multiselect(
            "Confidence Tier",
            options=["High", "Medium", "Low"],
            default=[],
            placeholder="All tiers",
            key="portfolio_tier_filter",
        )

    with col_search:
        search_text = st.text_input(
            "Search Customer or Merchant",
            placeholder="e.g. CUST000123 or Wells Fargo",
            key="portfolio_search",
        )

    # Apply filters
    filtered = detections.copy()

    if product_filter:
        filtered = filtered[filtered["detected_product_type"].isin(product_filter)]
    if tier_filter:
        filtered = filtered[filtered["confidence_tier"].isin(tier_filter)]
    if search_text:
        search_lower = search_text.lower()
        filtered = filtered[
            filtered["customer_id"].str.lower().str.contains(search_lower) |
            filtered["canonical_merchant"].str.lower().str.contains(search_lower)
        ]

    # Format for display
    display_df = filtered[[
        "customer_id", "detected_product_type", "confidence_tier",
        "confidence_score", "tenure_months", "recurring_amount_band",
        "canonical_merchant", "dominant_channel"
    ]].copy()

    display_df["confidence_score"] = display_df["confidence_score"].apply(lambda x: f"{x:.2f}")
    display_df["tenure_months"] = display_df["tenure_months"].apply(lambda x: f"{x:.1f} mo")
    display_df.columns = [
        "Customer ID", "Detected Product", "Confidence", "Score",
        "Tenure", "Amount Band", "Merchant", "Channel"
    ]

    # Row count info
    st.markdown(
        f'<div style="font-size:12px; color:#7f8c8d; margin-bottom:8px;">'
        f'Showing <b>{len(display_df):,}</b> of <b>{len(detections):,}</b> detections</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        display_df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        height=min(400, max(200, len(display_df) * 35 + 40)),
    )
