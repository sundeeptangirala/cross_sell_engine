"""
main.py
--------
Entry point for the Cross-Sell Intelligence Engine.

Reads the sample transaction data, runs the full detection pipeline,
and writes output to the outputs/ folder.

Usage (from the project root):
    python main.py

    # With optional arguments:
    python main.py --input path/to/transactions.csv
    python main.py --lookback 180
    python main.py --min-confidence Medium
"""

import sys
import os
import argparse
import logging
import pandas as pd
from datetime import datetime

# Ensure project root is on path (for VS Code runs from any working directory)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from pipeline import CrossSellPipeline
from monitoring.drift_monitor import DriftMonitor


# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-Sell Intelligence Engine — Detect external financial relationships."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input transactions CSV. Defaults to cross_sell_sample_data.csv in project root."
    )
    parser.add_argument(
        "--lookback", type=int, default=None,
        help="Lookback window in days. Defaults to config value (365)."
    )
    parser.add_argument(
        "--min-confidence", type=str, default="Medium",
        choices=["High", "Medium", "Low"],
        help="Minimum confidence tier to include in output. Default: Medium (filters out Low)."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Defaults to outputs/ in project root."
    )
    parser.add_argument(
        "--run-drift-monitor", action="store_true", default=False,
        help="Also run drift monitoring and output a drift report."
    )
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    # --- Resolve paths ---
    input_path = args.input or os.path.join(PROJECT_ROOT, "cross_sell_sample_data.csv")
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # --- Load transactions ---
    logger.info(f"Loading transactions from: {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    transactions = pd.read_csv(input_path)
    logger.info(f"Loaded {len(transactions):,} transactions, {transactions['customer_id'].nunique():,} customers.")

    # --- Run pipeline ---
    logger.info("Initializing pipeline...")
    pipeline = CrossSellPipeline(lookback_days=args.lookback)

    logger.info("Running detection pipeline...")
    detections = pipeline.run(transactions)
    logger.info(f"Raw detections: {len(detections):,} relationships detected.")

    # --- Apply confidence filter ---
    tier_order = {"High": 3, "Medium": 2, "Low": 1}
    min_tier_value = tier_order[args.min_confidence]
    filtered = detections[
        detections["confidence_tier"].map(tier_order) >= min_tier_value
    ].copy()
    logger.info(
        f"After filtering (>= {args.min_confidence}): {len(filtered):,} detections. "
        f"Filtered out: {len(detections) - len(filtered):,} Low-confidence detections."
    )

    # --- Output: Detection Results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detections_path = os.path.join(output_dir, f"detections_{timestamp}.csv")
    filtered.to_csv(detections_path, index=False)
    logger.info(f"Detections saved to: {detections_path}")

    # --- Print summary ---
    _print_summary(filtered)

    # --- Optional: Drift Monitoring ---
    if args.run_drift_monitor:
        logger.info("\nRunning drift monitor...")
        monitor = DriftMonitor()
        report = monitor.run(detections, transactions)

        logger.info(f"Drift Report: {report.summary}")
        for alert in report.alerts:
            level = {"CRITICAL": logging.ERROR, "WARNING": logging.WARNING}.get(alert.severity, logging.INFO)
            logger.log(level, f"[{alert.alert_type}] {alert.severity}: {alert.message}")

        # Save drift report
        drift_path = os.path.join(output_dir, f"drift_report_{timestamp}.csv")
        if report.alerts:
            drift_rows = [
                {
                    "alert_type": a.alert_type,
                    "severity": a.severity,
                    "product_type": a.product_type,
                    "metric_name": a.metric_name,
                    "metric_value": a.metric_value,
                    "threshold": a.threshold,
                    "message": a.message,
                    "detected_at": a.detected_at,
                }
                for a in report.alerts
            ]
            pd.DataFrame(drift_rows).to_csv(drift_path, index=False)
            logger.info(f"Drift report saved to: {drift_path}")
        else:
            logger.info("No drift alerts detected.")


def _print_summary(df: pd.DataFrame):
    """Prints a clean summary table to the console."""
    if df.empty:
        print("\n  No detections to display.\n")
        return

    print("\n" + "=" * 80)
    print("  CROSS-SELL DETECTION SUMMARY")
    print("=" * 80)

    # By product type
    print("\n  Detections by Product Type:")
    print("  " + "-" * 60)
    for product in df["detected_product_type"].unique():
        subset = df[df["detected_product_type"] == product]
        high = (subset["confidence_tier"] == "High").sum()
        med = (subset["confidence_tier"] == "Medium").sum()
        print(f"    {product:30s}  {len(subset):>5,} detections  (High: {high}, Medium: {med})")

    # By confidence tier
    print(f"\n  Confidence Mix:")
    print("  " + "-" * 60)
    for tier in ["High", "Medium", "Low"]:
        count = (df["confidence_tier"] == tier).sum()
        pct = (count / len(df) * 100) if len(df) > 0 else 0
        print(f"    {tier:10s}  {count:>5,}  ({pct:.1f}%)")

    # Coverage
    customers_with_detections = df["customer_id"].nunique()
    print(f"\n  Customers with detected external relationships: {customers_with_detections:,}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
