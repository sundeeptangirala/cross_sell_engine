"""
test_engine.py
---------------
Comprehensive test suite for the cross-sell detection engine.

Run from the project root:
    python -m pytest tests/test_engine.py -v

Tests are organized by layer:
    - Config & Taxonomy
    - Recurring Payment Detector
    - Product Interpreters
    - Full Pipeline (integration)
    - Drift Monitor
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config_loader import load_config, get_product_interpreter_config, get_all_product_types, reset_config
from core.taxonomy import TaxonomyLookup
from core.models import RecurringPaymentObject
from core.recurring_payment_detector import RecurringPaymentDetector
from interpreters.product_interpreters import get_all_interpreters, MortgageInterpreter
from pipeline import CrossSellPipeline
from monitoring.drift_monitor import DriftMonitor


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def reset_config_cache():
    """Clears config cache before each test for isolation."""
    reset_config()
    yield
    reset_config()


def _make_recurring_txns(
    customer_id: str = "CUST000001",
    merchant: str = "Wells Fargo Home Mortgage",
    mcc_code: int = 6159,
    mcc_desc: str = "Federal & Federally-Sponsored Credit Agencies",
    category: str = "Mortgage",
    base_amount: float = 2500.0,
    n_months: int = 10,
    channel: str = "ACH",
    amount_cv: float = 0.03,
    start_date: datetime = datetime(2024, 3, 1),
    start_txn_id: int = 1,
) -> pd.DataFrame:
    """Helper: generates a clean monthly recurring transaction series."""
    rows = []
    for i in range(n_months):
        month = start_date.month + i
        year = start_date.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        day = min(start_date.day, 28)
        date = datetime(year, month, day)
        amount = round(base_amount * (1 + np.random.normal(0, amount_cv)), 2)
        rows.append({
            "transaction_id": start_txn_id + i,
            "customer_id": customer_id,
            "transaction_date": date,
            "amount": max(amount, 10.0),
            "channel": channel,
            "raw_description": merchant.upper() + " PMT",
            "cleansed_description": merchant,
            "mcc_code": mcc_code,
            "mcc_description": mcc_desc,
            "category": category,
        })
    return pd.DataFrame(rows)


def _make_rpo(
    customer_id="CUST000001",
    merchant="Wells Fargo Home Mortgage",
    mcc_code=6159,
    category="Mortgage",
    cadence_type="monthly",
    cadence_strength=0.85,
    mean_amount=2500.0,
    amount_cv=0.03,
    tenure_months=9.0,
    occurrence_count=10,
    dominant_channel="ACH",
) -> RecurringPaymentObject:
    """Helper: creates a RecurringPaymentObject directly for interpreter tests."""
    return RecurringPaymentObject(
        customer_id=customer_id,
        merchant_canonical=merchant,
        mcc_code=mcc_code,
        mcc_description="Test MCC Desc",
        category=category,
        cadence_type=cadence_type,
        cadence_strength=cadence_strength,
        mean_amount=mean_amount,
        median_amount=mean_amount,
        amount_cv=amount_cv,
        amount_iqr=50.0,
        tenure_months=tenure_months,
        occurrence_count=occurrence_count,
        first_seen=datetime(2024, 3, 1),
        last_seen=datetime(2024, 12, 1),
        channel_distribution={"ACH": 0.7, "Bill Pay": 0.3},
        dominant_channel=dominant_channel,
        transaction_ids=list(range(1, occurrence_count + 1)),
    )


# =============================================================================
# CONFIG & TAXONOMY TESTS
# =============================================================================

class TestConfig:
    def test_config_loads_successfully(self):
        config = load_config()
        assert "recurring_detection" in config
        assert "merchant_taxonomy" in config
        assert "product_interpreters" in config
        assert "confidence_tiers" in config

    def test_all_product_types_present(self):
        products = get_all_product_types()
        expected = {"External Mortgage", "External Auto Loan", "External Credit Card",
                    "Student Loan", "Rent", "Insurance"}
        assert set(products) == expected

    def test_interpreter_config_has_required_keys(self):
        required_keys = {
            "expected_cadence", "min_amount", "max_amount",
            "max_amount_cv", "min_tenure_months", "expected_channels", "channel_weight"
        }
        for product in get_all_product_types():
            cfg = get_product_interpreter_config(product)
            assert required_keys.issubset(cfg.keys()), f"Missing keys for {product}"

    def test_missing_product_type_raises(self):
        with pytest.raises(KeyError):
            get_product_interpreter_config("Nonexistent Product")


class TestTaxonomy:
    def test_taxonomy_loads(self):
        taxonomy = TaxonomyLookup()
        assert len(taxonomy) > 0

    def test_known_lookup_succeeds(self):
        taxonomy = TaxonomyLookup()
        result = taxonomy.lookup(6159, "Mortgage")
        assert result is not None
        assert result["product_type"] == "External Mortgage"

    def test_unknown_lookup_returns_none(self):
        taxonomy = TaxonomyLookup()
        result = taxonomy.lookup(9999, "Unknown Category")
        assert result is None

    def test_get_product_type_shortcut(self):
        taxonomy = TaxonomyLookup()
        assert taxonomy.get_product_type(6512, "Rent") == "Rent"
        assert taxonomy.get_product_type(9999, "Fake") is None

    def test_all_product_types_present_in_taxonomy(self):
        taxonomy = TaxonomyLookup()
        products = taxonomy.get_all_product_types()
        expected = {"External Mortgage", "External Auto Loan", "External Credit Card",
                    "Student Loan", "Rent", "Insurance"}
        assert expected.issubset(products)


# =============================================================================
# RECURRING PAYMENT DETECTOR TESTS
# =============================================================================

class TestRecurringPaymentDetector:
    def test_detects_monthly_recurring(self):
        txns = _make_recurring_txns(n_months=10)
        detector = RecurringPaymentDetector()
        results = detector.detect(txns)
        assert len(results) == 1
        assert results[0].cadence_type == "monthly"
        assert results[0].cadence_strength > 0.5

    def test_filters_out_insufficient_occurrences(self):
        # Only 2 transactions — below min_occurrences (3)
        txns = _make_recurring_txns(n_months=2)
        detector = RecurringPaymentDetector()
        results = detector.detect(txns)
        assert len(results) == 0

    def test_amount_statistics_correct(self):
        txns = _make_recurring_txns(base_amount=1500.0, amount_cv=0.0, n_months=6)
        detector = RecurringPaymentDetector()
        results = detector.detect(txns)
        assert len(results) == 1
        # With CV=0, all amounts should be exactly 1500
        assert results[0].mean_amount == pytest.approx(1500.0, abs=1.0)
        assert results[0].amount_cv == pytest.approx(0.0, abs=0.01)

    def test_tenure_calculated_correctly(self):
        start = datetime(2024, 1, 1)
        txns = _make_recurring_txns(n_months=6, start_date=start)
        detector = RecurringPaymentDetector()
        results = detector.detect(txns)
        assert len(results) == 1
        # 5 months between first and last payment (months 1–6)
        assert results[0].tenure_months == pytest.approx(5.0, abs=0.5)

    def test_channel_distribution_captured(self):
        # Mix of ACH and Bill Pay
        txns = _make_recurring_txns(n_months=10, channel="ACH")
        # Manually flip some to Bill Pay
        txns.loc[txns.index[:3], "channel"] = "Bill Pay"

        detector = RecurringPaymentDetector()
        results = detector.detect(txns)
        assert len(results) == 1
        assert "ACH" in results[0].channel_distribution
        assert "Bill Pay" in results[0].channel_distribution

    def test_multiple_customers_independent(self):
        txns1 = _make_recurring_txns(customer_id="CUST000001", start_txn_id=1)
        txns2 = _make_recurring_txns(
            customer_id="CUST000002",
            merchant="Chase Home Finance",
            start_txn_id=100,
            base_amount=3000.0,
        )
        txns = pd.concat([txns1, txns2], ignore_index=True)

        detector = RecurringPaymentDetector()
        results = detector.detect(txns)
        assert len(results) == 2
        customer_ids = {r.customer_id for r in results}
        assert customer_ids == {"CUST000001", "CUST000002"}

    def test_irregular_transactions_filtered(self):
        """Random, non-recurring transactions should not produce RPOs."""
        np.random.seed(42)
        rows = []
        for i in range(20):
            rows.append({
                "transaction_id": i + 1,
                "customer_id": "CUST000001",
                "transaction_date": datetime(2024, 1, 1) + timedelta(days=int(np.random.uniform(1, 300))),
                "amount": round(np.random.uniform(10, 200), 2),
                "channel": "Card",
                "raw_description": "AMAZON.COM #123",
                "cleansed_description": "Amazon.com",
                "mcc_code": 5999,
                "mcc_description": "Retail Stores, NEC",
                "category": "Retail",
            })
        txns = pd.DataFrame(rows).sort_values("transaction_date").reset_index(drop=True)

        detector = RecurringPaymentDetector()
        results = detector.detect(txns)
        # Irregular noise should either not appear or have very low strength
        for r in results:
            assert r.cadence_strength < 0.5 or r.cadence_type == "irregular"

    def test_missing_columns_raises(self):
        bad_df = pd.DataFrame({"customer_id": ["CUST1"], "amount": [100.0]})
        detector = RecurringPaymentDetector()
        with pytest.raises(ValueError, match="Missing required columns"):
            detector.detect(bad_df)


# =============================================================================
# PRODUCT INTERPRETER TESTS
# =============================================================================

class TestProductInterpreters:
    def test_all_interpreters_instantiate(self):
        interpreters = get_all_interpreters()
        assert len(interpreters) == 6
        product_types = {i.product_type for i in interpreters}
        expected = {"External Mortgage", "External Auto Loan", "External Credit Card",
                    "Student Loan", "Rent", "Insurance"}
        assert product_types == expected

    def test_mortgage_interpreter_detects_valid_rpo(self):
        rpo = _make_rpo(
            mcc_code=6159, category="Mortgage",
            mean_amount=2500.0, amount_cv=0.03,
            cadence_strength=0.85, tenure_months=9.0,
        )
        interpreter = MortgageInterpreter()
        result = interpreter.interpret(rpo)
        assert result is not None
        assert result.detected_product_type == "External Mortgage"
        assert result.confidence_score > 0.5
        assert result.confidence_tier in {"High", "Medium", "Low"}

    def test_mortgage_interpreter_rejects_wrong_taxonomy(self):
        """RPO with auto loan MCC should not match mortgage interpreter."""
        rpo = _make_rpo(
            mcc_code=6141, category="Auto Loan",
            mean_amount=500.0, amount_cv=0.02,
        )
        interpreter = MortgageInterpreter()
        result = interpreter.interpret(rpo)
        assert result is None

    def test_mortgage_interpreter_rejects_amount_out_of_range(self):
        """Amount below mortgage minimum ($500) → hard gate failure."""
        rpo = _make_rpo(
            mcc_code=6159, category="Mortgage",
            mean_amount=50.0,  # Way too low for a mortgage
            amount_cv=0.03,
        )
        interpreter = MortgageInterpreter()
        result = interpreter.interpret(rpo)
        assert result is None

    def test_mortgage_interpreter_rejects_wrong_cadence(self):
        """Weekly cadence should not match monthly-expected mortgage."""
        rpo = _make_rpo(
            mcc_code=6159, category="Mortgage",
            mean_amount=2500.0, cadence_type="weekly",
        )
        interpreter = MortgageInterpreter()
        result = interpreter.interpret(rpo)
        assert result is None

    def test_mortgage_interpreter_rejects_insufficient_tenure(self):
        """Tenure below minimum (2 months) → hard gate failure."""
        rpo = _make_rpo(
            mcc_code=6159, category="Mortgage",
            mean_amount=2500.0, tenure_months=1.0,
        )
        interpreter = MortgageInterpreter()
        result = interpreter.interpret(rpo)
        assert result is None

    def test_credit_card_interpreter_tolerates_high_cv(self):
        """CC payments can have CV up to 0.30 — should still detect."""
        rpo = _make_rpo(
            mcc_code=6099, category="Credit Card Payment",
            merchant="Citi Card Payment",
            mean_amount=1500.0, amount_cv=0.25,
            cadence_strength=0.80, tenure_months=8.0,
        )
        interpreters = get_all_interpreters()
        cc_interpreter = next(i for i in interpreters if i.product_type == "External Credit Card")
        result = cc_interpreter.interpret(rpo)
        assert result is not None
        assert result.detected_product_type == "External Credit Card"

    def test_reason_codes_populated(self):
        """Every detection should have explanation reason codes."""
        rpo = _make_rpo(
            mcc_code=6159, category="Mortgage",
            mean_amount=2500.0, amount_cv=0.03,
            cadence_strength=0.85, tenure_months=9.0,
        )
        interpreter = MortgageInterpreter()
        result = interpreter.interpret(rpo)
        assert result is not None
        assert len(result.explanation_reason_codes) >= 3
        # Should contain key signal codes
        codes_str = " ".join(result.explanation_reason_codes)
        assert "CADENCE" in codes_str
        assert "AMOUNT" in codes_str
        assert "TAXONOMY" in codes_str

    def test_amount_band_formatting(self):
        """Amount bands should be human-readable strings."""
        interpreter = MortgageInterpreter()
        assert interpreter._format_amount_band(75.0) == "$0–$100"
        assert interpreter._format_amount_band(750.0) == "$500–$1,000"
        assert interpreter._format_amount_band(1500.0) == "$1,000–$2,000"
        assert interpreter._format_amount_band(6000.0) == "$5,000+"


# =============================================================================
# FULL PIPELINE INTEGRATION TESTS
# =============================================================================

class TestPipeline:
    def test_pipeline_runs_end_to_end(self):
        """Full pipeline on synthetic mortgage data should produce output."""
        txns = _make_recurring_txns(n_months=10, base_amount=2500.0, amount_cv=0.03)
        pipeline = CrossSellPipeline()
        output = pipeline.run(txns)

        assert isinstance(output, pd.DataFrame)
        assert len(output) == 1
        assert output.iloc[0]["detected_product_type"] == "External Mortgage"
        assert output.iloc[0]["confidence_tier"] in {"High", "Medium", "Low"}

    def test_pipeline_output_schema(self):
        """Output DataFrame must match the spec's §11 schema."""
        txns = _make_recurring_txns(n_months=10)
        pipeline = CrossSellPipeline()
        output = pipeline.run(txns)

        expected_cols = {
            "customer_id", "detected_product_type", "confidence_tier",
            "confidence_score", "first_detected_date", "last_detected_date",
            "tenure_months", "recurring_amount_band", "mean_amount",
            "canonical_merchant", "dominant_channel",
            "evidence_transaction_refs", "explanation_reason_codes",
        }
        assert expected_cols.issubset(set(output.columns))

    def test_pipeline_with_mixed_products(self):
        """Multiple product types in one run should all be detected."""
        mortgage_txns = _make_recurring_txns(
            customer_id="CUST000001", merchant="Wells Fargo Home Mortgage",
            mcc_code=6159, category="Mortgage",
            base_amount=2500.0, amount_cv=0.03, start_txn_id=1,
        )
        auto_txns = _make_recurring_txns(
            customer_id="CUST000002", merchant="Capital One Auto Finance",
            mcc_code=6141, category="Auto Loan",
            base_amount=450.0, amount_cv=0.02, start_txn_id=100,
        )
        rent_txns = _make_recurring_txns(
            customer_id="CUST000003", merchant="Greystar Property Mgmt",
            mcc_code=6512, category="Rent",
            base_amount=1800.0, amount_cv=0.04, start_txn_id=200,
        )
        txns = pd.concat([mortgage_txns, auto_txns, rent_txns], ignore_index=True)

        pipeline = CrossSellPipeline()
        output = pipeline.run(txns)

        assert len(output) == 3
        detected_types = set(output["detected_product_type"].values)
        assert "External Mortgage" in detected_types
        assert "External Auto Loan" in detected_types
        assert "Rent" in detected_types

    def test_pipeline_empty_input(self):
        """Empty input should return empty DataFrame with correct schema."""
        empty_df = pd.DataFrame(columns=[
            "transaction_id", "customer_id", "transaction_date", "amount",
            "channel", "raw_description", "cleansed_description",
            "mcc_code", "mcc_description", "category"
        ])
        pipeline = CrossSellPipeline()
        output = pipeline.run(empty_df)
        assert len(output) == 0

    def test_pipeline_noise_only_produces_no_detections(self):
        """Pure retail/dining noise should not trigger any product detection."""
        np.random.seed(42)
        rows = []
        for i in range(50):
            rows.append({
                "transaction_id": i + 1,
                "customer_id": "CUST000001",
                "transaction_date": datetime(2024, 1, 1) + timedelta(days=i * 7),
                "amount": round(np.random.uniform(10, 100), 2),
                "channel": "Card",
                "raw_description": "AMAZON.COM STORE",
                "cleansed_description": "Amazon.com",
                "mcc_code": 5999,
                "mcc_description": "Retail Stores, NEC",
                "category": "Retail",
            })
        txns = pd.DataFrame(rows)

        pipeline = CrossSellPipeline()
        output = pipeline.run(txns)
        assert len(output) == 0


# =============================================================================
# DRIFT MONITOR TESTS
# =============================================================================

class TestDriftMonitor:
    def test_drift_monitor_runs_without_error(self):
        """Smoke test: monitor should run and return a DriftReport."""
        # Create synthetic detections
        detections = pd.DataFrame({
            "detected_product_type": ["External Mortgage"] * 20,
            "confidence_score": np.random.uniform(0.6, 0.95, 20),
            "mean_amount": np.random.uniform(2000, 3000, 20),
            "last_detected_date": [
                (datetime.now() - timedelta(days=np.random.randint(0, 120))).strftime("%Y-%m-%d")
                for _ in range(20)
            ],
            "canonical_merchant": ["Wells Fargo Home Mortgage"] * 20,
        })

        transactions = pd.DataFrame({
            "transaction_date": [
                (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(100)
            ],
            "cleansed_description": ["Wells Fargo Home Mortgage"] * 100,
            "category": ["Mortgage"] * 100,
        })

        monitor = DriftMonitor()
        report = monitor.run(detections, transactions)

        assert report is not None
        assert "total_alerts" in report.summary
        assert isinstance(report.alerts, list)

    def test_psi_computation(self):
        """PSI should be ~0 for identical distributions, >0 for shifted ones."""
        monitor = DriftMonitor()

        # Identical distributions → PSI ≈ 0
        baseline = np.random.normal(100, 10, 500)
        same = np.random.normal(100, 10, 500)
        psi_same = monitor._compute_psi(baseline, same)
        assert psi_same < 0.1  # Should be very small

        # Shifted distribution → PSI > 0
        shifted = np.random.normal(150, 10, 500)
        psi_shifted = monitor._compute_psi(baseline, shifted)
        assert psi_shifted > psi_same  # Should be meaningfully larger


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
