"""
recurring_payment_detector.py
------------------------------
Product-agnostic recurring payment detection engine.

This is the shared foundation layer (Section 7.1 of the spec). It does NOT
know or care about product types. It only answers one question:

    "For this customer + merchant, is there a recurring payment pattern?"

Output: a RecurringPaymentObject per qualifying group. These objects are
then consumed by product interpreters for classification and confidence scoring.

Design decisions:
    - Grouping key is (customer_id, cleansed_description). This clusters
      payments to the same merchant regardless of MCC/category variations.
    - Cadence detection uses inter-transaction gap analysis, not fixed
      calendar binning. This handles irregular pay periods correctly.
    - All thresholds and tolerances are read from config.yaml.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List

from core.models import RecurringPaymentObject
from config.config_loader import get_recurring_detection_config


class RecurringPaymentDetector:
    """
    Detects recurring payment patterns in transaction data.

    Usage:
        detector = RecurringPaymentDetector()
        recurring_objects = detector.detect(transactions_df)
    """

    def __init__(self):
        self.config = get_recurring_detection_config()
        self.min_occurrences = self.config["min_occurrences"]
        self.cadence_tolerances = self.config["cadence_tolerances"]
        self.cadence_scoring_weights = self.config["cadence_scoring"]

    # -------------------------------------------------------------------------
    # PUBLIC INTERFACE
    # -------------------------------------------------------------------------

    def detect(self, transactions: pd.DataFrame, lookback_days: int | None = None) -> List[RecurringPaymentObject]:
        """
        Run recurring payment detection on a transactions DataFrame.

        Args:
            transactions: DataFrame with columns:
                customer_id, transaction_date, amount, channel,
                cleansed_description, mcc_code, mcc_description, category,
                transaction_id
            lookback_days: Override the default lookback window. If None,
                uses config default.

        Returns:
            List of RecurringPaymentObject, one per qualifying
            (customer, merchant) group.
        """
        df = self._prepare(transactions, lookback_days)

        if df.empty:
            return []

        # Group by customer + merchant and process each group
        grouped = df.groupby(["customer_id", "cleansed_description"])
        results: List[RecurringPaymentObject] = []

        for (customer_id, merchant), group in grouped:
            # Filter: minimum occurrences gate
            if len(group) < self.min_occurrences:
                continue

            rpo = self._build_recurring_payment_object(customer_id, merchant, group)
            if rpo is not None:
                results.append(rpo)

        return results

    # -------------------------------------------------------------------------
    # INTERNAL: DATA PREPARATION
    # -------------------------------------------------------------------------

    def _prepare(self, transactions: pd.DataFrame, lookback_days: int | None) -> pd.DataFrame:
        """
        Validates input, parses dates, and applies the lookback window filter.
        """
        required_cols = [
            "customer_id", "transaction_date", "amount", "channel",
            "cleansed_description", "mcc_code", "mcc_description",
            "category", "transaction_id"
        ]
        missing = [c for c in required_cols if c not in transactions.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = transactions.copy()

        # Parse dates if not already datetime
        if not pd.api.types.is_datetime64_any_dtype(df["transaction_date"]):
            df["transaction_date"] = pd.to_datetime(df["transaction_date"])

        # Apply lookback window
        if lookback_days is None:
            lookback_days = self.config["default_lookback_days"]

        cutoff = df["transaction_date"].max() - pd.Timedelta(days=lookback_days)
        df = df[df["transaction_date"] >= cutoff].copy()

        # Sort by customer + date for gap calculations
        df = df.sort_values(["customer_id", "cleansed_description", "transaction_date"]).reset_index(drop=True)

        return df

    # -------------------------------------------------------------------------
    # INTERNAL: RECURRING PAYMENT OBJECT CONSTRUCTION
    # -------------------------------------------------------------------------

    def _build_recurring_payment_object(
        self, customer_id: str, merchant: str, group: pd.DataFrame
    ) -> RecurringPaymentObject | None:
        """
        Builds a RecurringPaymentObject from a single (customer, merchant) group.

        Returns None if the group does not meet cadence criteria (i.e. the
        payments exist but are not sufficiently regular to be "recurring").
        """
        # --- Cadence detection ---
        cadence_type, cadence_strength = self._detect_cadence(group)

        # If cadence is irregular and strength is very low, skip entirely.
        # This filters out truly random one-off transactions.
        if cadence_type == "irregular" and cadence_strength < 0.3:
            return None

        # --- Amount statistics ---
        amounts = group["amount"].values
        mean_amt = float(np.mean(amounts))
        median_amt = float(np.median(amounts))
        std_amt = float(np.std(amounts, ddof=1)) if len(amounts) > 1 else 0.0
        amount_cv = (std_amt / mean_amt) if mean_amt > 0 else 0.0
        q1, q3 = float(np.percentile(amounts, 25)), float(np.percentile(amounts, 75))
        amount_iqr = q3 - q1

        # --- Tenure ---
        first_seen = group["transaction_date"].min()
        last_seen = group["transaction_date"].max()
        tenure_months = (last_seen - first_seen).days / 30.44  # Avg days per month

        # --- Channel distribution ---
        channel_dist = group["channel"].value_counts(normalize=True).to_dict()
        dominant_channel = group["channel"].mode().iloc[0]

        # --- Metadata (use first row for MCC/category — consistent within group) ---
        first_row = group.iloc[0]

        return RecurringPaymentObject(
            customer_id=customer_id,
            merchant_canonical=merchant,
            mcc_code=int(first_row["mcc_code"]),
            mcc_description=str(first_row["mcc_description"]),
            category=str(first_row["category"]),
            cadence_type=cadence_type,
            cadence_strength=round(cadence_strength, 4),
            mean_amount=round(mean_amt, 2),
            median_amount=round(median_amt, 2),
            amount_cv=round(amount_cv, 4),
            amount_iqr=round(amount_iqr, 2),
            tenure_months=round(tenure_months, 2),
            occurrence_count=len(group),
            first_seen=first_seen.to_pydatetime() if hasattr(first_seen, "to_pydatetime") else first_seen,
            last_seen=last_seen.to_pydatetime() if hasattr(last_seen, "to_pydatetime") else last_seen,
            channel_distribution=channel_dist,
            dominant_channel=dominant_channel,
            transaction_ids=group["transaction_id"].tolist(),
        )

    # -------------------------------------------------------------------------
    # INTERNAL: CADENCE DETECTION
    # -------------------------------------------------------------------------

    def _detect_cadence(self, group: pd.DataFrame) -> tuple[str, float]:
        """
        Determines the cadence type and strength score for a transaction group.

        Logic:
            1. Compute all inter-transaction gaps in days.
            2. Test each known cadence type (monthly, biweekly, weekly) against
               tolerance windows from config.
            3. Pick the cadence with the best fit. If none fit well, label "irregular".
            4. Compute a strength score (0.0–1.0) based on gap consistency,
               occurrence count, and tenure.

        Returns:
            Tuple of (cadence_type, cadence_strength_score).
        """
        dates = group["transaction_date"].sort_values().values
        gaps = np.diff(dates).astype("timedelta64[D]").astype(float)

        if len(gaps) == 0:
            return ("irregular", 0.0)

        # --- Test each cadence type ---
        best_cadence = "irregular"
        best_fit_ratio = 0.0

        for cadence_name, tolerance in self.cadence_tolerances.items():
            min_gap = tolerance["min_gap_days"]
            max_gap = tolerance["max_gap_days"]

            # What fraction of gaps fall within this cadence's tolerance window?
            in_window = np.sum((gaps >= min_gap) & (gaps <= max_gap))
            fit_ratio = in_window / len(gaps)

            # Require at least 60% of gaps to match to qualify as this cadence
            if fit_ratio >= 0.60 and fit_ratio > best_fit_ratio:
                best_fit_ratio = fit_ratio
                best_cadence = cadence_name

        # --- Compute strength score ---
        strength = self._compute_cadence_strength(gaps, best_cadence, best_fit_ratio, group)

        return (best_cadence, strength)

    def _compute_cadence_strength(
        self, gaps: np.ndarray, cadence_type: str, fit_ratio: float, group: pd.DataFrame
    ) -> float:
        """
        Computes a 0.0–1.0 strength score using the weighted formula from config:
            - gap_consistency_weight: How regular are the gaps? (fit_ratio)
            - occurrence_count_weight: Normalized occurrence count.
            - tenure_weight: Normalized tenure.

        Each component is scaled to [0, 1] before weighting.
        """
        w = self.cadence_scoring_weights

        # Component 1: Gap consistency (fit_ratio is already 0–1)
        gap_consistency = fit_ratio if cadence_type != "irregular" else 0.0

        # Component 2: Occurrence count, normalized.
        # 3 occurrences = minimum (score ~0.2), 12+ = full score.
        occ = len(group)
        occ_score = min((occ - self.min_occurrences) / (12 - self.min_occurrences), 1.0)
        occ_score = max(occ_score, 0.0)

        # Component 3: Tenure in months, normalized.
        # 2 months = minimum (~0.2), 12+ months = full score.
        dates = group["transaction_date"].sort_values().values
        tenure_days = (dates[-1] - dates[0]).astype("timedelta64[D]").astype(float)
        tenure_months = tenure_days / 30.44
        tenure_score = min((tenure_months - 2) / (12 - 2), 1.0)
        tenure_score = max(tenure_score, 0.0)

        # Weighted composite
        strength = (
            w["gap_consistency_weight"] * gap_consistency
            + w["occurrence_count_weight"] * occ_score
            + w["tenure_weight"] * tenure_score
        )

        return round(min(strength, 1.0), 4)
