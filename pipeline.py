"""
pipeline.py
------------
Main orchestration layer. Wires together:
    1. RecurringPaymentDetector  →  produces RecurringPaymentObjects
    2. Product Interpreters      →  classifies RPOs into DetectionResults
    3. Output serialization      →  writes customer-level output to CSV

This is the single entry point for running the engine. Everything else
is internal machinery.

Usage:
    from pipeline import CrossSellPipeline

    pipeline = CrossSellPipeline()
    results_df = pipeline.run(transactions_df)
"""

import pandas as pd
import logging
from typing import List
from datetime import datetime

from core.models import RecurringPaymentObject, DetectionResult
from core.recurring_payment_detector import RecurringPaymentDetector
from interpreters.product_interpreters import get_all_interpreters
from config.config_loader import load_config

logger = logging.getLogger(__name__)


class CrossSellPipeline:
    """
    End-to-end cross-sell detection pipeline.

    Orchestrates detection → interpretation → output without exposing
    internal objects to callers.
    """

    def __init__(self, lookback_days: int | None = None):
        """
        Args:
            lookback_days: Override default lookback window from config.
        """
        self.config = load_config()
        self.lookback_days = lookback_days
        self.detector = RecurringPaymentDetector()
        self.interpreters = get_all_interpreters()

        logger.info(
            f"Pipeline initialized. "
            f"Interpreters: {[i.product_type for i in self.interpreters]}. "
            f"Lookback: {lookback_days or self.config['recurring_detection']['default_lookback_days']} days."
        )

    # -------------------------------------------------------------------------
    # PUBLIC INTERFACE
    # -------------------------------------------------------------------------

    def run(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full detection pipeline.

        Args:
            transactions: DataFrame with required columns (see detector).

        Returns:
            DataFrame of detection results, one row per detected external
            relationship. Columns match the spec's Customer-Level Output (§11).
        """
        logger.info(f"Pipeline starting. Input: {len(transactions):,} transactions.")

        # --- Stage 1: Recurring payment detection ---
        rpobjects = self.detector.detect(transactions, lookback_days=self.lookback_days)
        logger.info(f"Stage 1 complete. Recurring payment objects: {len(rpobjects):,}.")

        # --- Stage 2: Product interpretation ---
        detections = self._run_interpreters(rpobjects)
        logger.info(f"Stage 2 complete. Detections: {len(detections):,}.")

        # --- Stage 3: Serialize to DataFrame ---
        output_df = self._serialize_detections(detections)
        logger.info(f"Pipeline complete. Output rows: {len(output_df):,}.")

        return output_df

    def run_detection_only(self, transactions: pd.DataFrame) -> List[RecurringPaymentObject]:
        """
        Run only Stage 1 (recurring payment detection). Useful for debugging
        or for the Tuning & QA view.
        """
        return self.detector.detect(transactions, lookback_days=self.lookback_days)

    # -------------------------------------------------------------------------
    # INTERNAL: INTERPRETATION
    # -------------------------------------------------------------------------

    def _run_interpreters(self, rpobjects: List[RecurringPaymentObject]) -> List[DetectionResult]:
        """
        Runs all product interpreters against all recurring payment objects.

        Each RPO is tested against every interpreter. An RPO can match at most
        one product type (the taxonomy gate ensures this — each (mcc, category)
        maps to exactly one product_type). But we run all interpreters for safety
        and future flexibility.
        """
        detections: List[DetectionResult] = []

        for rpo in rpobjects:
            for interpreter in self.interpreters:
                result = interpreter.interpret(rpo)
                if result is not None:
                    detections.append(result)
                    # An RPO should only match one product — break after first match
                    break

        return detections

    # -------------------------------------------------------------------------
    # INTERNAL: OUTPUT SERIALIZATION
    # -------------------------------------------------------------------------

    def _serialize_detections(self, detections: List[DetectionResult]) -> pd.DataFrame:
        """
        Converts DetectionResult objects to a flat DataFrame matching the
        spec's Customer-Level Output schema (§11).
        """
        if not detections:
            return pd.DataFrame(columns=[
                "customer_id", "detected_product_type", "confidence_tier",
                "confidence_score", "first_detected_date", "last_detected_date",
                "tenure_months", "recurring_amount_band", "mean_amount",
                "canonical_merchant", "dominant_channel",
                "evidence_transaction_refs", "explanation_reason_codes",
            ])

        rows = []
        for d in detections:
            rows.append({
                "customer_id": d.customer_id,
                "detected_product_type": d.detected_product_type,
                "confidence_tier": d.confidence_tier,
                "confidence_score": d.confidence_score,
                "first_detected_date": d.first_detected_date.strftime("%Y-%m-%d") if hasattr(d.first_detected_date, "strftime") else str(d.first_detected_date),
                "last_detected_date": d.last_detected_date.strftime("%Y-%m-%d") if hasattr(d.last_detected_date, "strftime") else str(d.last_detected_date),
                "tenure_months": d.tenure_months,
                "recurring_amount_band": d.recurring_amount_band,
                "mean_amount": d.mean_amount,
                "canonical_merchant": d.canonical_merchant,
                "dominant_channel": d.dominant_channel,
                "evidence_transaction_refs": "|".join(str(x) for x in d.evidence_transaction_refs),
                "explanation_reason_codes": " | ".join(d.explanation_reason_codes),
            })

        df = pd.DataFrame(rows)

        # Sort: customer → confidence score descending
        df = df.sort_values(
            ["customer_id", "confidence_score"],
            ascending=[True, False]
        ).reset_index(drop=True)

        return df
