"""
taxonomy.py
------------
Merchant taxonomy lookup layer.

Loads the merchant_taxonomy table from config.yaml and builds a fast
lookup index keyed on (mcc_code, category). This is the bridge between
raw transaction fields and the product classification space.

Taxonomy updates happen in config.yaml — no code changes required.
"""

from typing import Optional, Dict, Tuple
from config.config_loader import get_merchant_taxonomy


class TaxonomyLookup:
    """
    Fast lookup from (mcc_code, category) → product_type + match_confidence.

    Built once at init from the config taxonomy table. Thread-safe for reads.
    """

    def __init__(self):
        self._index: Dict[Tuple[int, str], Dict] = {}
        self._load_taxonomy()

    def _load_taxonomy(self) -> None:
        """Builds the lookup index from config."""
        for entry in get_merchant_taxonomy():
            key = (entry["mcc_code"], entry["category"])
            # If duplicate keys exist, keep the higher confidence entry
            if key in self._index:
                if entry["match_confidence"] > self._index[key]["match_confidence"]:
                    self._index[key] = entry
            else:
                self._index[key] = entry

    def lookup(self, mcc_code: int, category: str) -> Optional[Dict]:
        """
        Look up product type for a given (mcc_code, category) pair.

        Args:
            mcc_code: MCC code from the transaction.
            category: Category label from the transaction.

        Returns:
            Dict with product_type and match_confidence, or None if no match.
        """
        return self._index.get((mcc_code, category))

    def get_product_type(self, mcc_code: int, category: str) -> Optional[str]:
        """Shortcut: returns just the product_type string or None."""
        result = self.lookup(mcc_code, category)
        return result["product_type"] if result else None

    def get_match_confidence(self, mcc_code: int, category: str) -> float:
        """Returns the taxonomy match confidence, or 0.0 if no match."""
        result = self.lookup(mcc_code, category)
        return result["match_confidence"] if result else 0.0

    def get_all_product_types(self) -> set[str]:
        """Returns all product types present in the taxonomy."""
        return {entry["product_type"] for entry in self._index.values()}

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return f"TaxonomyLookup(entries={len(self)}, products={self.get_all_product_types()})"
