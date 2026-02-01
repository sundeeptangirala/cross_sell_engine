"""
models.py
----------
Core domain models. These are the typed contracts between engine layers.

- RecurringPaymentObject: Output of the detection layer. Product-agnostic.
  This is the reusable primitive that all product interpreters consume.

- DetectionResult: Output of a product interpreter. Customer-facing result
  with confidence tier, evidence references, and explanation codes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class RecurringPaymentObject:
    """
    Product-agnostic recurring payment detection output.

    Produced by RecurringPaymentDetector for each (customer, merchant) group
    that meets the minimum recurrence criteria. Consumed by all product
    interpreters as input.
    """

    # Identity
    customer_id: str
    merchant_canonical: str          # Cleansed/normalized merchant name
    mcc_code: int
    mcc_description: str
    category: str

    # Cadence
    cadence_type: str                # "monthly" | "biweekly" | "weekly" | "irregular"
    cadence_strength: float          # 0.0 – 1.0. How regular the payment pattern is.

    # Amount statistics
    mean_amount: float
    median_amount: float
    amount_cv: float                 # Coefficient of variation (std / mean). Lower = more stable.
    amount_iqr: float                # Interquartile range.

    # Tenure & frequency
    tenure_months: float             # Months between first and last observation.
    occurrence_count: int            # Total number of transactions in the window.
    first_seen: datetime
    last_seen: datetime

    # Channel
    channel_distribution: dict       # e.g. {"ACH": 0.7, "Bill Pay": 0.3}
    dominant_channel: str            # Channel with highest share.

    # Evidence
    transaction_ids: list[int] = field(default_factory=list)  # Supporting txn IDs


@dataclass
class DetectionResult:
    """
    Product interpreter output. One per detected external relationship.

    This is what gets surfaced to business users (after confidence filtering)
    and written to the customer-level output table.
    """

    # Identity
    customer_id: str
    detected_product_type: str       # e.g. "External Mortgage"

    # Confidence
    confidence_score: float          # Raw composite score 0.0–1.0
    confidence_tier: str             # "High" | "Medium" | "Low"

    # Timing
    first_detected_date: datetime
    last_detected_date: datetime
    tenure_months: float

    # Amount
    recurring_amount_band: str       # Human-readable band, e.g. "$2,000–$2,500"
    mean_amount: float

    # Source
    canonical_merchant: str
    dominant_channel: str

    # Evidence & explainability
    evidence_transaction_refs: list[int] = field(default_factory=list)
    explanation_reason_codes: list[str] = field(default_factory=list)

    # Metadata
    detected_at: datetime = field(default_factory=datetime.now)
