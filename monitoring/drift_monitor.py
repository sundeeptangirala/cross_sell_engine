"""
drift_monitor.py
------------------
Drift monitoring for the cross-sell detection engine.

Implements three monitoring dimensions from the spec (Section 9C):
    1. Detection volume trends — are we detecting more/less than expected?
    2. Amount distribution shifts — has the payment amount distribution changed?
    3. Unknown merchant ratio — are we seeing merchants we can't classify?

Methods:
    - KS test (Kolmogorov-Smirnov): Detects distributional shifts in amounts
      between a baseline window and a recent window.
    - PSI (Population Stability Index): Quantifies how much a distribution
      has shifted. Industry standard thresholds: <0.1 = stable, 0.1–0.25 = minor
      shift, >0.25 = major shift.

All thresholds and window sizes come from config.yaml.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats
from typing import List

from config.config_loader import get_drift_monitoring_config


@dataclass
class DriftAlert:
    """A single drift detection alert."""
    alert_type: str                  # "VOLUME" | "AMOUNT_DISTRIBUTION" | "UNKNOWN_MERCHANT"
    severity: str                    # "INFO" | "WARNING" | "CRITICAL"
    product_type: str                # Which product (or "ALL")
    metric_name: str                 # e.g. "detection_count", "ks_statistic"
    metric_value: float
    threshold: float
    message: str
    detected_at: str = ""            # ISO timestamp


@dataclass
class DriftReport:
    """Full drift monitoring report — one per run."""
    run_timestamp: str
    baseline_window: str             # e.g. "2024-02-01 to 2024-11-01"
    comparison_window: str           # e.g. "2024-11-01 to 2024-12-01"
    alerts: List[DriftAlert] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


class DriftMonitor:
    """
    Monitors detection output for distributional drift.

    Usage:
        monitor = DriftMonitor()
        report = monitor.run(detections_df, transactions_df)
    """

    def __init__(self):
        self.config = get_drift_monitoring_config()
        self.ks_alpha = self.config["ks_alpha"]
        self.psi_minor = self.config["psi_minor"]
        self.psi_major = self.config["psi_major"]
        self.baseline_days = self.config["baseline_days"]
        self.comparison_days = self.config["comparison_days"]

    # -------------------------------------------------------------------------
    # PUBLIC INTERFACE
    # -------------------------------------------------------------------------

    def run(self, detections_df: pd.DataFrame, transactions_df: pd.DataFrame) -> DriftReport:
        """
        Run full drift monitoring suite.

        Args:
            detections_df: DataFrame of DetectionResults (output of the engine).
                Must have columns: detected_product_type, confidence_score,
                mean_amount, last_detected_date, canonical_merchant.
            transactions_df: Raw transactions DataFrame (for unknown merchant ratio).
                Must have columns: transaction_date, cleansed_description, category.

        Returns:
            DriftReport with all alerts and summary metrics.
        """
        now = pd.Timestamp.now()
        comparison_start = now - pd.Timedelta(days=self.comparison_days)
        baseline_start = comparison_start - pd.Timedelta(days=self.baseline_days)

        alerts: List[DriftAlert] = []

        # --- 1. Volume drift ---
        alerts.extend(self._check_volume_drift(detections_df, baseline_start, comparison_start, now))

        # --- 2. Amount distribution drift (per product) ---
        alerts.extend(self._check_amount_drift(detections_df, baseline_start, comparison_start, now))

        # --- 3. Unknown merchant ratio ---
        alerts.extend(self._check_unknown_merchant_ratio(transactions_df, baseline_start, comparison_start, now))

        # --- Summary ---
        summary = {
            "total_alerts": len(alerts),
            "critical_alerts": sum(1 for a in alerts if a.severity == "CRITICAL"),
            "warning_alerts": sum(1 for a in alerts if a.severity == "WARNING"),
            "info_alerts": sum(1 for a in alerts if a.severity == "INFO"),
        }

        return DriftReport(
            run_timestamp=now.isoformat(),
            baseline_window=f"{baseline_start.date()} to {comparison_start.date()}",
            comparison_window=f"{comparison_start.date()} to {now.date()}",
            alerts=alerts,
            summary=summary,
        )

    # -------------------------------------------------------------------------
    # INTERNAL: VOLUME DRIFT
    # -------------------------------------------------------------------------

    def _check_volume_drift(
        self, df: pd.DataFrame, baseline_start, comparison_start, now
    ) -> List[DriftAlert]:
        """
        Checks if detection volume has shifted significantly between windows.
        Uses a simple ratio comparison: if volume changed by >50% in either
        direction, flag it.
        """
        alerts = []

        if "last_detected_date" not in df.columns:
            return alerts

        df = df.copy()
        df["last_detected_date"] = pd.to_datetime(df["last_detected_date"])

        baseline_mask = (df["last_detected_date"] >= baseline_start) & (df["last_detected_date"] < comparison_start)
        comparison_mask = (df["last_detected_date"] >= comparison_start) & (df["last_detected_date"] <= now)

        baseline_count = baseline_mask.sum()
        comparison_count = comparison_mask.sum()

        if baseline_count == 0:
            return alerts  # No baseline to compare against

        # Normalize to per-day rate for fair comparison across different window sizes
        baseline_rate = baseline_count / self.baseline_days
        comparison_rate = comparison_count / self.comparison_days

        ratio = comparison_rate / baseline_rate if baseline_rate > 0 else 0

        if ratio > 1.5 or ratio < 0.5:
            severity = "CRITICAL" if (ratio > 2.0 or ratio < 0.33) else "WARNING"
            alerts.append(DriftAlert(
                alert_type="VOLUME",
                severity=severity,
                product_type="ALL",
                metric_name="detection_volume_ratio",
                metric_value=round(ratio, 3),
                threshold=1.5,
                message=(
                    f"Detection volume changed by {((ratio - 1) * 100):+.0f}%. "
                    f"Baseline rate: {baseline_rate:.1f}/day, "
                    f"Current rate: {comparison_rate:.1f}/day."
                ),
                detected_at=now.isoformat(),
            ))

        return alerts

    # -------------------------------------------------------------------------
    # INTERNAL: AMOUNT DISTRIBUTION DRIFT
    # -------------------------------------------------------------------------

    def _check_amount_drift(
        self, df: pd.DataFrame, baseline_start, comparison_start, now
    ) -> List[DriftAlert]:
        """
        Per-product KS test + PSI on detection amounts between baseline and comparison windows.
        """
        alerts = []
        required_cols = {"last_detected_date", "detected_product_type", "mean_amount"}
        if not required_cols.issubset(df.columns):
            return alerts

        df = df.copy()
        df["last_detected_date"] = pd.to_datetime(df["last_detected_date"])

        baseline_mask = (df["last_detected_date"] >= baseline_start) & (df["last_detected_date"] < comparison_start)
        comparison_mask = (df["last_detected_date"] >= comparison_start) & (df["last_detected_date"] <= now)

        for product_type in df["detected_product_type"].unique():
            product_mask = df["detected_product_type"] == product_type
            baseline_amounts = df.loc[baseline_mask & product_mask, "mean_amount"].values
            comparison_amounts = df.loc[comparison_mask & product_mask, "mean_amount"].values

            # Need minimum samples for meaningful tests
            if len(baseline_amounts) < 10 or len(comparison_amounts) < 5:
                continue

            # --- KS Test ---
            ks_stat, ks_pvalue = stats.ks_2samp(baseline_amounts, comparison_amounts)
            if ks_pvalue < self.ks_alpha:
                alerts.append(DriftAlert(
                    alert_type="AMOUNT_DISTRIBUTION",
                    severity="WARNING",
                    product_type=product_type,
                    metric_name="ks_p_value",
                    metric_value=round(ks_pvalue, 4),
                    threshold=self.ks_alpha,
                    message=(
                        f"Amount distribution shift detected for {product_type}. "
                        f"KS statistic={ks_stat:.3f}, p-value={ks_pvalue:.4f}."
                    ),
                    detected_at=now.isoformat(),
                ))

            # --- PSI ---
            psi = self._compute_psi(baseline_amounts, comparison_amounts)
            if psi > self.psi_minor:
                severity = "CRITICAL" if psi > self.psi_major else "WARNING"
                alerts.append(DriftAlert(
                    alert_type="AMOUNT_DISTRIBUTION",
                    severity=severity,
                    product_type=product_type,
                    metric_name="psi",
                    metric_value=round(psi, 4),
                    threshold=self.psi_major if severity == "CRITICAL" else self.psi_minor,
                    message=(
                        f"PSI={psi:.3f} for {product_type}. "
                        f"({'Major' if severity == 'CRITICAL' else 'Minor'} distribution shift.)"
                    ),
                    detected_at=now.isoformat(),
                ))

        return alerts

    # -------------------------------------------------------------------------
    # INTERNAL: UNKNOWN MERCHANT RATIO
    # -------------------------------------------------------------------------

    def _check_unknown_merchant_ratio(
        self, transactions_df: pd.DataFrame, baseline_start, comparison_start, now
    ) -> List[DriftAlert]:
        """
        Checks the ratio of transactions in financial categories that didn't
        map to any product detection. A rising unknown ratio can indicate
        new merchants or taxonomy gaps.
        """
        alerts = []
        required_cols = {"transaction_date", "category"}
        if not required_cols.issubset(transactions_df.columns):
            return alerts

        df = transactions_df.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])

        # Financial categories that SHOULD map to products
        financial_cats = {
            "Mortgage", "Auto Loan", "Credit Card Payment",
            "Student Loan", "Rent", "Insurance"
        }

        for window_name, start, end in [
            ("baseline", baseline_start, comparison_start),
            ("comparison", comparison_start, now),
        ]:
            mask = (df["transaction_date"] >= start) & (df["transaction_date"] <= end)
            window_df = df[mask]

            if window_df.empty:
                continue

            financial_mask = window_df["category"].isin(financial_cats)
            total_financial = financial_mask.sum()
            if total_financial == 0:
                continue

        # Simple check: if there are financial-category transactions that
        # aren't in our taxonomy, that's an unknown merchant signal.
        # We flag this as INFO level — it's a data hygiene signal, not an emergency.
        comparison_mask = (df["transaction_date"] >= comparison_start) & (df["transaction_date"] <= now)
        comparison_df = df[comparison_mask]
        if not comparison_df.empty:
            financial_mask = comparison_df["category"].isin(financial_cats)
            financial_count = financial_mask.sum()
            # This is a placeholder ratio — in production, you'd cross-reference
            # against actual detections to find truly unclassified merchants.
            if financial_count > 0:
                alerts.append(DriftAlert(
                    alert_type="UNKNOWN_MERCHANT",
                    severity="INFO",
                    product_type="ALL",
                    metric_name="financial_category_txn_count",
                    metric_value=float(financial_count),
                    threshold=0.0,
                    message=(
                        f"Found {financial_count} financial-category transactions in comparison window. "
                        f"Cross-reference against detections to identify unclassified merchants."
                    ),
                    detected_at=now.isoformat(),
                ))

        return alerts

    # -------------------------------------------------------------------------
    # INTERNAL: PSI CALCULATION
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_psi(baseline: np.ndarray, comparison: np.ndarray, n_bins: int = 10) -> float:
        """
        Computes Population Stability Index between two distributions.

        PSI = Σ (P_actual - P_expected) * ln(P_actual / P_expected)

        Uses the baseline distribution to define bin edges, then maps both
        distributions into those bins.
        """
        # Define bin edges from baseline
        bin_edges = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
        # Ensure unique edges (can collapse if data has low cardinality)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            return 0.0  # Not enough variation to compute PSI

        # Bin both distributions
        baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
        comparison_counts, _ = np.histogram(comparison, bins=bin_edges)

        # Add small epsilon to avoid log(0) and division by zero
        eps = 1e-6
        baseline_freq = (baseline_counts + eps) / (baseline_counts.sum() + eps * len(baseline_counts))
        comparison_freq = (comparison_counts + eps) / (comparison_counts.sum() + eps * len(comparison_counts))

        # PSI formula
        psi = np.sum((comparison_freq - baseline_freq) * np.log(comparison_freq / baseline_freq))

        return float(psi)
