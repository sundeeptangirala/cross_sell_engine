"""
base_interpreter.py
---------------------
Abstract base class for all product interpreters.

Each concrete interpreter (mortgage, auto loan, etc.) inherits from this.
Shared logic — confidence tier mapping, amount banding, reason code
generation — lives here so it's never duplicated.

Concrete interpreters only need to implement:
    - _compute_signal_score(): product-specific scoring logic
    - _get_reason_codes(): product-specific explainability strings
"""

from abc import ABC, abstractmethod
from datetime import datetime

from core.models import RecurringPaymentObject, DetectionResult
from core.taxonomy import TaxonomyLookup
from config.config_loader import get_product_interpreter_config, get_confidence_tiers


class BaseProductInterpreter(ABC):
    """
    Abstract base for product interpreters.

    Subclasses implement _compute_signal_score() and _get_reason_codes().
    This class handles taxonomy matching, confidence tier assignment,
    and DetectionResult construction.
    """

    def __init__(self, product_type: str):
        self.product_type = product_type
        self.config = get_product_interpreter_config(product_type)
        self.taxonomy = TaxonomyLookup()
        self.tier_boundaries = get_confidence_tiers()

    # -------------------------------------------------------------------------
    # PUBLIC INTERFACE
    # -------------------------------------------------------------------------

    def interpret(self, rpo: RecurringPaymentObject) -> DetectionResult | None:
        """
        Attempt to classify a RecurringPaymentObject as this product type.

        Returns:
            DetectionResult if the RPO matches this product, None otherwise.
        """
        # Step 1: Taxonomy gate — does this RPO's (mcc, category) map to us?
        taxonomy_match = self.taxonomy.lookup(rpo.mcc_code, rpo.category)
        if taxonomy_match is None or taxonomy_match["product_type"] != self.product_type:
            return None  # Not our product type

        taxonomy_confidence = taxonomy_match["match_confidence"]

        # Step 2: Signal scoring — product-specific checks
        signal_score = self._compute_signal_score(rpo)
        if signal_score is None:
            return None  # Failed a hard gate (e.g. amount out of range)

        # Step 3: Composite score
        # Weighted blend: 70% signal score, 30% taxonomy match confidence
        composite_score = round(0.70 * signal_score + 0.30 * taxonomy_confidence, 4)

        # Step 4: Confidence tier
        confidence_tier = self._assign_tier(composite_score)

        # Step 5: Reason codes
        reason_codes = self._get_reason_codes(rpo, signal_score, taxonomy_confidence)

        # Step 6: Build result
        return DetectionResult(
            customer_id=rpo.customer_id,
            detected_product_type=self.product_type,
            confidence_score=composite_score,
            confidence_tier=confidence_tier,
            first_detected_date=rpo.first_seen,
            last_detected_date=rpo.last_seen,
            tenure_months=rpo.tenure_months,
            recurring_amount_band=self._format_amount_band(rpo.mean_amount),
            mean_amount=rpo.mean_amount,
            canonical_merchant=rpo.merchant_canonical,
            dominant_channel=rpo.dominant_channel,
            evidence_transaction_refs=rpo.transaction_ids,
            explanation_reason_codes=reason_codes,
            detected_at=datetime.now(),
        )

    # -------------------------------------------------------------------------
    # ABSTRACT METHODS — Implement in each product interpreter
    # -------------------------------------------------------------------------

    @abstractmethod
    def _compute_signal_score(self, rpo: RecurringPaymentObject) -> float | None:
        """
        Compute a 0.0–1.0 signal score for this RPO against this product's rules.

        Returns:
            Float score if the RPO passes all hard gates, None if it fails
            a hard gate (e.g. amount completely out of range).
        """
        ...

    @abstractmethod
    def _get_reason_codes(
        self, rpo: RecurringPaymentObject, signal_score: float, taxonomy_confidence: float
    ) -> list[str]:
        """
        Generate human-readable explanation strings for this detection.
        These surface in the UI's explainability panel.
        """
        ...

    # -------------------------------------------------------------------------
    # SHARED HELPERS
    # -------------------------------------------------------------------------

    def _assign_tier(self, score: float) -> str:
        """Maps a composite score to a confidence tier string."""
        for tier_name, bounds in self.tier_boundaries.items():
            if bounds["min_score"] <= score < bounds["max_score"]:
                return tier_name
        # Edge case: score == 1.0 exactly
        return "High"

    def _check_amount_in_range(self, amount: float) -> bool:
        """Hard gate: is the amount within this product's expected range?"""
        return self.config["min_amount"] <= amount <= self.config["max_amount"]

    def _check_cv_within_limit(self, cv: float) -> bool:
        """Soft check: is amount variability within expected bounds?"""
        return cv <= self.config["max_amount_cv"]

    def _check_tenure(self, tenure_months: float) -> bool:
        """Hard gate: has the pattern been observed long enough?"""
        return tenure_months >= self.config["min_tenure_months"]

    def _check_cadence(self, cadence_type: str) -> bool:
        """Hard gate: does the cadence type match expectations?"""
        return cadence_type == self.config["expected_cadence"]

    def _compute_channel_score(self, rpo: RecurringPaymentObject) -> float:
        """
        Scores channel alignment. 1.0 if dominant channel is in expected set,
        partial credit if some expected channels appear in distribution.
        """
        expected = set(self.config["expected_channels"])
        if rpo.dominant_channel in expected:
            return 1.0
        # Partial: what fraction of transactions used expected channels?
        expected_share = sum(
            rpo.channel_distribution.get(ch, 0.0) for ch in expected
        )
        return expected_share

    @staticmethod
    def _format_amount_band(amount: float) -> str:
        """Formats a mean amount into a human-readable band string."""
        bands = [
            (0, 100, "$0–$100"),
            (100, 250, "$100–$250"),
            (250, 500, "$250–$500"),
            (500, 1000, "$500–$1,000"),
            (1000, 2000, "$1,000–$2,000"),
            (2000, 3500, "$2,000–$3,500"),
            (3500, 5000, "$3,500–$5,000"),
            (5000, float("inf"), "$5,000+"),
        ]
        for low, high, label in bands:
            if low <= amount < high:
                return label
        return "$5,000+"
