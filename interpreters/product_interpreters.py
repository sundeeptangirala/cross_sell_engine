"""
product_interpreters.py
-------------------------
Concrete product interpreters. One class per detected product type.

Each interpreter applies product-specific scoring logic to a
RecurringPaymentObject. The scoring follows a consistent pattern:

    1. Hard gates: If the RPO fails any hard gate, return None (no detection).
    2. Component scores: Score each signal dimension (cadence, amount stability,
       tenure, channel) independently on a 0–1 scale.
    3. Weighted composite: Blend component scores into a single signal score.

Thresholds and weights come from config.yaml — only the scoring structure
lives in code.
"""

from core.models import RecurringPaymentObject
from interpreters.base_interpreter import BaseProductInterpreter


# =============================================================================
# EXTERNAL MORTGAGE
# =============================================================================
class MortgageInterpreter(BaseProductInterpreter):
    """
    Detects external mortgage payments.

    Key signals: monthly cadence, high amount (>$500), very low variability
    (CV < 0.08), multi-month tenure. Mortgage payments are among the most
    stable recurring patterns in transaction data.
    """

    def __init__(self):
        super().__init__("External Mortgage")

    def _compute_signal_score(self, rpo: RecurringPaymentObject) -> float | None:
        # --- Hard gates ---
        if not self._check_amount_in_range(rpo.mean_amount):
            return None
        if not self._check_cadence(rpo.cadence_type):
            return None
        if not self._check_tenure(rpo.tenure_months):
            return None

        # --- Component scores (each 0–1) ---
        # Cadence strength: direct from RPO
        cadence_score = rpo.cadence_strength

        # Amount stability: CV-based. Mortgage CV should be very low.
        # Score = 1.0 at CV=0, linearly decays to 0 at max_amount_cv * 2
        max_cv = self.config["max_amount_cv"]
        stability_score = max(1.0 - (rpo.amount_cv / (max_cv * 2)), 0.0)

        # Tenure: normalized. Full score at 12 months, linear from min.
        min_tenure = self.config["min_tenure_months"]
        tenure_score = min((rpo.tenure_months - min_tenure) / (12 - min_tenure), 1.0)

        # Channel alignment
        channel_score = self._compute_channel_score(rpo)

        # --- Weighted composite ---
        # Mortgage: cadence and stability are the strongest signals
        channel_w = self.config["channel_weight"]
        remaining = 1.0 - channel_w
        signal_score = (
            remaining * 0.40 * cadence_score
            + remaining * 0.35 * stability_score
            + remaining * 0.25 * tenure_score
            + channel_w * channel_score
        )

        return round(signal_score, 4)

    def _get_reason_codes(self, rpo, signal_score, taxonomy_confidence) -> list[str]:
        codes = []
        codes.append(f"CADENCE_MONTHLY (strength={rpo.cadence_strength:.2f})")
        codes.append(f"AMOUNT_STABLE (CV={rpo.amount_cv:.3f}, mean=${rpo.mean_amount:,.2f})")
        codes.append(f"TENURE_{rpo.tenure_months:.1f}_MONTHS")
        codes.append(f"TAXONOMY_MATCH (confidence={taxonomy_confidence:.2f})")
        if rpo.dominant_channel in self.config["expected_channels"]:
            codes.append(f"CHANNEL_MATCH ({rpo.dominant_channel})")
        return codes


# =============================================================================
# EXTERNAL AUTO LOAN
# =============================================================================
class AutoLoanInterpreter(BaseProductInterpreter):
    """
    Detects external auto loan payments.

    Key signals: monthly cadence, moderate amount ($100–$2,000), very low
    variability (fixed-rate loans have CV near 0). Similar structure to
    mortgage but different amount range.
    """

    def __init__(self):
        super().__init__("External Auto Loan")

    def _compute_signal_score(self, rpo: RecurringPaymentObject) -> float | None:
        if not self._check_amount_in_range(rpo.mean_amount):
            return None
        if not self._check_cadence(rpo.cadence_type):
            return None
        if not self._check_tenure(rpo.tenure_months):
            return None

        cadence_score = rpo.cadence_strength

        max_cv = self.config["max_amount_cv"]
        stability_score = max(1.0 - (rpo.amount_cv / (max_cv * 2)), 0.0)

        min_tenure = self.config["min_tenure_months"]
        tenure_score = min((rpo.tenure_months - min_tenure) / (12 - min_tenure), 1.0)

        channel_score = self._compute_channel_score(rpo)

        channel_w = self.config["channel_weight"]
        remaining = 1.0 - channel_w
        signal_score = (
            remaining * 0.40 * cadence_score
            + remaining * 0.35 * stability_score
            + remaining * 0.25 * tenure_score
            + channel_w * channel_score
        )

        return round(signal_score, 4)

    def _get_reason_codes(self, rpo, signal_score, taxonomy_confidence) -> list[str]:
        codes = []
        codes.append(f"CADENCE_MONTHLY (strength={rpo.cadence_strength:.2f})")
        codes.append(f"AMOUNT_STABLE (CV={rpo.amount_cv:.3f}, mean=${rpo.mean_amount:,.2f})")
        codes.append(f"TENURE_{rpo.tenure_months:.1f}_MONTHS")
        codes.append(f"TAXONOMY_MATCH (confidence={taxonomy_confidence:.2f})")
        if rpo.dominant_channel in self.config["expected_channels"]:
            codes.append(f"CHANNEL_MATCH ({rpo.dominant_channel})")
        return codes


# =============================================================================
# EXTERNAL CREDIT CARD
# =============================================================================
class CreditCardInterpreter(BaseProductInterpreter):
    """
    Detects external credit card payments.

    Key signals: monthly cadence, variable amounts (min pay vs full pay
    creates higher CV). This is the most "noisy" product — CV tolerance
    is set much higher (0.30) than loans.
    """

    def __init__(self):
        super().__init__("External Credit Card")

    def _compute_signal_score(self, rpo: RecurringPaymentObject) -> float | None:
        if not self._check_amount_in_range(rpo.mean_amount):
            return None
        if not self._check_cadence(rpo.cadence_type):
            return None
        if not self._check_tenure(rpo.tenure_months):
            return None

        cadence_score = rpo.cadence_strength

        # CC has higher CV tolerance — use a wider decay curve
        max_cv = self.config["max_amount_cv"]
        stability_score = max(1.0 - (rpo.amount_cv / (max_cv * 1.5)), 0.0)

        min_tenure = self.config["min_tenure_months"]
        tenure_score = min((rpo.tenure_months - min_tenure) / (12 - min_tenure), 1.0)

        channel_score = self._compute_channel_score(rpo)

        # CC: cadence is the strongest signal (amount varies too much to weight heavily)
        channel_w = self.config["channel_weight"]
        remaining = 1.0 - channel_w
        signal_score = (
            remaining * 0.50 * cadence_score
            + remaining * 0.20 * stability_score
            + remaining * 0.30 * tenure_score
            + channel_w * channel_score
        )

        return round(signal_score, 4)

    def _get_reason_codes(self, rpo, signal_score, taxonomy_confidence) -> list[str]:
        codes = []
        codes.append(f"CADENCE_MONTHLY (strength={rpo.cadence_strength:.2f})")
        codes.append(f"AMOUNT_VARIABLE (CV={rpo.amount_cv:.3f}, mean=${rpo.mean_amount:,.2f})")
        codes.append(f"TENURE_{rpo.tenure_months:.1f}_MONTHS")
        codes.append(f"TAXONOMY_MATCH (confidence={taxonomy_confidence:.2f})")
        if rpo.dominant_channel in self.config["expected_channels"]:
            codes.append(f"CHANNEL_MATCH ({rpo.dominant_channel})")
        return codes


# =============================================================================
# STUDENT LOAN
# =============================================================================
class StudentLoanInterpreter(BaseProductInterpreter):
    """
    Detects student loan payments.

    Key signals: monthly cadence, moderate fixed amount ($100–$2,500),
    very low variability (federal loans are fixed). Similar to auto loan
    structurally.
    """

    def __init__(self):
        super().__init__("Student Loan")

    def _compute_signal_score(self, rpo: RecurringPaymentObject) -> float | None:
        if not self._check_amount_in_range(rpo.mean_amount):
            return None
        if not self._check_cadence(rpo.cadence_type):
            return None
        if not self._check_tenure(rpo.tenure_months):
            return None

        cadence_score = rpo.cadence_strength

        max_cv = self.config["max_amount_cv"]
        stability_score = max(1.0 - (rpo.amount_cv / (max_cv * 2)), 0.0)

        min_tenure = self.config["min_tenure_months"]
        tenure_score = min((rpo.tenure_months - min_tenure) / (12 - min_tenure), 1.0)

        channel_score = self._compute_channel_score(rpo)

        channel_w = self.config["channel_weight"]
        remaining = 1.0 - channel_w
        signal_score = (
            remaining * 0.40 * cadence_score
            + remaining * 0.35 * stability_score
            + remaining * 0.25 * tenure_score
            + channel_w * channel_score
        )

        return round(signal_score, 4)

    def _get_reason_codes(self, rpo, signal_score, taxonomy_confidence) -> list[str]:
        codes = []
        codes.append(f"CADENCE_MONTHLY (strength={rpo.cadence_strength:.2f})")
        codes.append(f"AMOUNT_STABLE (CV={rpo.amount_cv:.3f}, mean=${rpo.mean_amount:,.2f})")
        codes.append(f"TENURE_{rpo.tenure_months:.1f}_MONTHS")
        codes.append(f"TAXONOMY_MATCH (confidence={taxonomy_confidence:.2f})")
        if rpo.dominant_channel in self.config["expected_channels"]:
            codes.append(f"CHANNEL_MATCH ({rpo.dominant_channel})")
        return codes


# =============================================================================
# RENT
# =============================================================================
class RentInterpreter(BaseProductInterpreter):
    """
    Detects external rent payments.

    Key signals: monthly cadence, moderate-to-high amount ($300–$8,000),
    low variability. Rent is distinctive because Check is a significant
    channel (unlike most other products).
    """

    def __init__(self):
        super().__init__("Rent")

    def _compute_signal_score(self, rpo: RecurringPaymentObject) -> float | None:
        if not self._check_amount_in_range(rpo.mean_amount):
            return None
        if not self._check_cadence(rpo.cadence_type):
            return None
        if not self._check_tenure(rpo.tenure_months):
            return None

        cadence_score = rpo.cadence_strength

        max_cv = self.config["max_amount_cv"]
        stability_score = max(1.0 - (rpo.amount_cv / (max_cv * 2)), 0.0)

        min_tenure = self.config["min_tenure_months"]
        tenure_score = min((rpo.tenure_months - min_tenure) / (12 - min_tenure), 1.0)

        channel_score = self._compute_channel_score(rpo)

        channel_w = self.config["channel_weight"]
        remaining = 1.0 - channel_w
        signal_score = (
            remaining * 0.40 * cadence_score
            + remaining * 0.30 * stability_score
            + remaining * 0.30 * tenure_score
            + channel_w * channel_score
        )

        return round(signal_score, 4)

    def _get_reason_codes(self, rpo, signal_score, taxonomy_confidence) -> list[str]:
        codes = []
        codes.append(f"CADENCE_MONTHLY (strength={rpo.cadence_strength:.2f})")
        codes.append(f"AMOUNT_STABLE (CV={rpo.amount_cv:.3f}, mean=${rpo.mean_amount:,.2f})")
        codes.append(f"TENURE_{rpo.tenure_months:.1f}_MONTHS")
        codes.append(f"TAXONOMY_MATCH (confidence={taxonomy_confidence:.2f})")
        if rpo.dominant_channel in self.config["expected_channels"]:
            codes.append(f"CHANNEL_MATCH ({rpo.dominant_channel})")
        return codes


# =============================================================================
# INSURANCE
# =============================================================================
class InsuranceInterpreter(BaseProductInterpreter):
    """
    Detects insurance premium payments.

    Key signals: monthly cadence, lower fixed amount ($40–$1,500), low
    variability. Insurance premiums are very stable but lower dollar value
    than loans — this can create overlap with other recurring small payments.
    """

    def __init__(self):
        super().__init__("Insurance")

    def _compute_signal_score(self, rpo: RecurringPaymentObject) -> float | None:
        if not self._check_amount_in_range(rpo.mean_amount):
            return None
        if not self._check_cadence(rpo.cadence_type):
            return None
        if not self._check_tenure(rpo.tenure_months):
            return None

        cadence_score = rpo.cadence_strength

        max_cv = self.config["max_amount_cv"]
        stability_score = max(1.0 - (rpo.amount_cv / (max_cv * 2)), 0.0)

        min_tenure = self.config["min_tenure_months"]
        tenure_score = min((rpo.tenure_months - min_tenure) / (12 - min_tenure), 1.0)

        channel_score = self._compute_channel_score(rpo)

        channel_w = self.config["channel_weight"]
        remaining = 1.0 - channel_w
        signal_score = (
            remaining * 0.35 * cadence_score
            + remaining * 0.35 * stability_score
            + remaining * 0.30 * tenure_score
            + channel_w * channel_score
        )

        return round(signal_score, 4)

    def _get_reason_codes(self, rpo, signal_score, taxonomy_confidence) -> list[str]:
        codes = []
        codes.append(f"CADENCE_MONTHLY (strength={rpo.cadence_strength:.2f})")
        codes.append(f"AMOUNT_STABLE (CV={rpo.amount_cv:.3f}, mean=${rpo.mean_amount:,.2f})")
        codes.append(f"TENURE_{rpo.tenure_months:.1f}_MONTHS")
        codes.append(f"TAXONOMY_MATCH (confidence={taxonomy_confidence:.2f})")
        if rpo.dominant_channel in self.config["expected_channels"]:
            codes.append(f"CHANNEL_MATCH ({rpo.dominant_channel})")
        return codes


# =============================================================================
# INTERPRETER REGISTRY
# =============================================================================
# Single source of truth for all active interpreters.
# To add a new product: create the class above, add it here.

INTERPRETER_REGISTRY: dict[str, type[BaseProductInterpreter]] = {
    "External Mortgage": MortgageInterpreter,
    "External Auto Loan": AutoLoanInterpreter,
    "External Credit Card": CreditCardInterpreter,
    "Student Loan": StudentLoanInterpreter,
    "Rent": RentInterpreter,
    "Insurance": InsuranceInterpreter,
}


def get_all_interpreters() -> list[BaseProductInterpreter]:
    """Instantiates and returns all registered interpreters."""
    return [cls() for cls in INTERPRETER_REGISTRY.values()]
