"""
app.py
-------
Streamlit application entry point for the Cross-Sell Intelligence Engine.

Run from the project root:
    streamlit run ui/app.py

Architecture:
    - Global state (transactions, detections, RPOs) is cached via st.session_state
      and loaded once on first run.
    - Sidebar handles navigation and global filters (time window, confidence).
    - Each view is a separate module for maintainability.
"""

import sys
import os
import streamlit as st
import pandas as pd

# Ensure project root is on path regardless of where streamlit is invoked
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pipeline import CrossSellPipeline
from core.recurring_payment_detector import RecurringPaymentDetector
from ui.customer_view import render_customer_view
from ui.portfolio_view import render_portfolio_view
from ui.tuning_view import render_tuning_view


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Cross-Sell Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Global font and background */
    .stApp {
        font-family: 'Segoe UI', system-ui, sans-serif;
        background-color: #f4f6f9;
    }

    /* Sidebar styling */
    .css-1d798d5, [data-testid="stSidebar"] {
        background-color: #1a2332 !important;
    }
    [data-testid="stSidebar"] * {
        color: #c8d6e5 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a2332 0%, #2c3e50 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .main-header h1 {
        margin: 0;
        font-size: 24px;
        font-weight: 600;
        letter-spacing: -0.3px;
    }
    .main-header p {
        margin: 4px 0 0 0;
        opacity: 0.7;
        font-size: 13px;
    }

    /* KPI Cards */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #3498db;
        text-align: left;
    }
    .kpi-card.green { border-left-color: #27ae60; }
    .kpi-card.orange { border-left-color: #e67e22; }
    .kpi-card.purple { border-left-color: #8e44ad; }
    .kpi-card.red    { border-left-color: #e74c3c; }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #1a2332;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 12px;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Confidence badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .badge-high    { background: #d4edda; color: #155724; }
    .badge-medium  { background: #fff3cd; color: #856404; }
    .badge-low     { background: #f8d7da; color: #721c24; }

    /* Product detection cards */
    .product-card {
        background: white;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 12px;
        border: 1px solid #edf1f4;
    }
    .product-card:hover {
        box-shadow: 0 4px 14px rgba(0,0,0,0.10);
        border-color: #3498db;
    }
    .product-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .product-card-header h4 {
        margin: 0;
        color: #1a2332;
        font-size: 15px;
    }

    /* Evidence table */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Section dividers */
    .section-title {
        font-size: 14px;
        font-weight: 600;
        color: #1a2332;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        padding-bottom: 8px;
        border-bottom: 2px solid #edf1f4;
        margin-bottom: 12px;
    }

    /* Nav buttons in sidebar */
    .nav-btn {
        display: block;
        width: 100%;
        padding: 10px 16px;
        margin: 4px 0;
        border: none;
        border-radius: 8px;
        background: transparent;
        color: #c8d6e5;
        text-align: left;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s;
    }
    .nav-btn:hover {
        background: rgba(255,255,255,0.08);
        color: #ffffff;
    }
    .nav-btn.active {
        background: rgba(52, 152, 219, 0.2);
        color: #5dade2;
        font-weight: 600;
    }

    /* Reason code pills */
    .reason-pill {
        display: inline-block;
        background: #eef2f7;
        color: #2c3e50;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11.5px;
        font-family: 'Consolas', monospace;
        margin: 2px;
    }

    /* Alerts */
    .alert-critical {
        background: #fdecea;
        border-left: 4px solid #e74c3c;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
    }
    .alert-warning {
        background: #fef9e7;
        border-left: 4px solid #e67e22;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
    }
    .alert-info {
        background: #eaf2f8;
        border-left: 4px solid #3498db;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING & CACHING
# =============================================================================

@st.cache_data(show_spinner="Loading transactions...")
def load_transactions(input_path: str) -> pd.DataFrame:
    """Loads and caches the transaction CSV."""
    df = pd.read_csv(input_path)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


@st.cache_data(show_spinner="Running detection pipeline...")
def run_pipeline(_transactions_hash, transactions: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Runs the full pipeline and caches the result. Hash param forces re-run on data change."""
    pipeline = CrossSellPipeline(lookback_days=lookback_days)
    return pipeline.run(transactions)


@st.cache_data(show_spinner="Detecting recurring patterns...")
def run_detection_only(_transactions_hash, transactions: pd.DataFrame, lookback_days: int) -> list:
    """Runs Stage 1 only (recurring payment objects) for Tuning & QA view."""
    detector = RecurringPaymentDetector()
    return detector.detect(transactions, lookback_days=lookback_days)


def initialize_data():
    """
    Ensures transaction data and detection results are loaded into session state.
    Only runs the pipeline once — subsequent navigations reuse cached results.
    """
    if "transactions" not in st.session_state:
        input_path = os.path.join(PROJECT_ROOT, "cross_sell_sample_data.csv")
        if not os.path.exists(input_path):
            st.error(f"❌ Sample data not found at: {input_path}")
            st.stop()
        st.session_state["transactions"] = load_transactions(input_path)
        st.session_state["input_path"] = input_path

    if "detections" not in st.session_state or st.session_state.get("_lookback") != st.session_state.get("lookback_days", 365):
        lookback = st.session_state.get("lookback_days", 365)
        txn_hash = str(len(st.session_state["transactions"]))  # Simple cache key
        st.session_state["detections"] = run_pipeline(txn_hash, st.session_state["transactions"], lookback)
        st.session_state["_lookback"] = lookback

    if "rpo_objects" not in st.session_state or st.session_state.get("_rpo_lookback") != st.session_state.get("lookback_days", 365):
        lookback = st.session_state.get("lookback_days", 365)
        txn_hash = str(len(st.session_state["transactions"]))
        st.session_state["rpo_objects"] = run_detection_only(txn_hash, st.session_state["transactions"], lookback)
        st.session_state["_rpo_lookback"] = lookback


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Renders the sidebar navigation and global controls."""

    st.sidebar.markdown("""
        <div style="padding: 10px 0 20px 0; text-align: center;">
            <div style="font-size: 22px; font-weight: 700; color: #fff; letter-spacing: -0.5px;">🔍 Cross-Sell</div>
            <div style="font-size: 11px; color: #7f8c8d; margin-top: 2px;">Intelligence Engine v1</div>
        </div>
    """, unsafe_allow_html=True)

    # --- Navigation ---
    st.sidebar.markdown("<div style='font-size:10px; color:#5a6a7a; text-transform:uppercase; letter-spacing:1px; padding: 8px 0 4px 0;'>Navigation</div>", unsafe_allow_html=True)

    pages = {
        "📋  Customer View": "customer",
        "📊  Portfolio View": "portfolio",
        "⚙️  Tuning & QA": "tuning",
    }

    current_page = st.session_state.get("current_page", "customer")

    for label, key in pages.items():
        active_class = "active" if current_page == key else ""
        if st.sidebar.button(label, key=f"nav_{key}", use_container_width=True,
                             help={"customer": "Search and inspect individual customers",
                                   "portfolio": "Portfolio-level KPIs and trends",
                                   "tuning": "DS/Ops: unclassified patterns and drift monitors"}[key]):
            st.session_state["current_page"] = key
            st.rerun()

    # --- Global Controls ---
    st.sidebar.markdown("<hr style='border-color:#2c3e50; margin: 20px 0 12px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='font-size:10px; color:#5a6a7a; text-transform:uppercase; letter-spacing:1px; padding-bottom:6px;'>Global Controls</div>", unsafe_allow_html=True)

    # Time window
    lookback_options = {
        "90 days": 90,
        "180 days": 180,
        "365 days": 365,
    }
    selected_lookback_label = st.sidebar.selectbox(
        "Time Window",
        options=list(lookback_options.keys()),
        index=2,  # Default: 365 days
    )
    new_lookback = lookback_options[selected_lookback_label]
    if st.session_state.get("lookback_days") != new_lookback:
        st.session_state["lookback_days"] = new_lookback
        # Clear cached detections to force re-run
        st.session_state.pop("detections", None)
        st.session_state.pop("rpo_objects", None)
        st.rerun()

    # Confidence filter
    confidence_options = ["High + Medium", "High Only", "All (incl. Low)"]
    selected_confidence = st.sidebar.selectbox(
        "Confidence Filter",
        options=confidence_options,
        index=0,  # Default: High + Medium
    )
    st.session_state["confidence_filter"] = selected_confidence

    # --- Data info footer ---
    st.sidebar.markdown("<hr style='border-color:#2c3e50; margin: 20px 0 12px 0;'>", unsafe_allow_html=True)
    if "transactions" in st.session_state:
        txns = st.session_state["transactions"]
        st.sidebar.markdown(f"""
            <div style='font-size:11px; color:#5a6a7a; line-height:1.6;'>
                <b style='color:#8a9bb0;'>Dataset</b><br>
                {len(txns):,} transactions<br>
                {txns['customer_id'].nunique():,} customers<br>
                {txns['transaction_date'].min().strftime('%b %Y')} – {txns['transaction_date'].max().strftime('%b %Y')}
            </div>
        """, unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Load data on first visit
    initialize_data()

    # Sidebar
    render_sidebar()

    # Apply confidence filter to detections
    detections = st.session_state["detections"].copy()
    confidence_filter = st.session_state.get("confidence_filter", "High + Medium")

    if confidence_filter == "High Only":
        detections = detections[detections["confidence_tier"] == "High"]
    elif confidence_filter == "High + Medium":
        detections = detections[detections["confidence_tier"].isin(["High", "Medium"])]
    # "All" = no filter

    st.session_state["filtered_detections"] = detections

    # Route to the selected page
    page = st.session_state.get("current_page", "customer")

    if page == "customer":
        render_customer_view(
            detections=detections,
            transactions=st.session_state["transactions"],
        )
    elif page == "portfolio":
        render_portfolio_view(
            detections=detections,
            transactions=st.session_state["transactions"],
        )
    elif page == "tuning":
        render_tuning_view(
            detections=detections,
            transactions=st.session_state["transactions"],
            rpo_objects=st.session_state["rpo_objects"],
        )


if __name__ == "__main__":
    main()
