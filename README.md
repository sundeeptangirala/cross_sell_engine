# Cross-Sell Intelligence Engine

Production-grade system for detecting external financial relationships from transaction data.

## Overview

This engine analyzes customer transaction patterns to identify external products (mortgages, auto loans, credit cards, student loans, rent, insurance) held at other institutions. Built per the product specification in `Cross-sell_Intelligence_Using_Transaction_Data.docx`.

## Architecture

- **Product-agnostic detection layer**: Identifies recurring payment patterns regardless of product type
- **Product interpreters**: Lightweight scoring and classification per product (6 interpreters)
- **Config-driven**: All thresholds, taxonomy mappings, and amount bands live in `config/config.yaml`
- **Evidence-based explainability**: Every detection includes reason codes and supporting transactions

## Project Structure

```
cross_sell_engine/
├── config/              # Configuration (thresholds, taxonomy)
├── core/                # Detection engine (product-agnostic)
├── interpreters/        # Product-specific scoring logic
├── monitoring/          # Drift detection (KS test, PSI)
├── ui/                  # Streamlit interface (3 views)
├── tests/               # Test suite (32 tests)
├── main.py              # CLI entry point
└── pipeline.py          # Orchestrator
```

## Installation

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run CLI (outputs to outputs/)
python main.py

# Run UI
streamlit run ui/app.py
```

### Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. New app → Repository: `your-username/cross-sell-engine` → Main file: `ui/app.py`
4. Deploy

## Usage

### Command Line

```bash
# Run with defaults (365-day lookback, Medium+ confidence)
python main.py

# Custom lookback window
python main.py --lookback 180

# Include low-confidence detections
python main.py --min-confidence Low

# Run with drift monitoring
python main.py --run-drift-monitor
```

### Streamlit UI

Three views:

1. **Customer View** — Search customers, see detected relationships, evidence, and explainability
2. **Portfolio View** — KPIs, product gap distribution, trends, drill-down table
3. **Tuning & QA** — Unclassified patterns, sampling/feedback, drift monitors, config viewer (DS/Ops only)

## Configuration

All volatile inputs live in `config/config.yaml`:

- **Merchant taxonomy**: (mcc_code, category) → product_type mappings
- **Product thresholds**: Amount ranges, CV limits, tenure minimums per product
- **Confidence tiers**: Score boundaries for High/Medium/Low
- **Drift monitoring**: KS alpha, PSI thresholds, baseline windows

Update config.yaml — no code changes needed for threshold tuning.

## Testing

```bash
# Run full test suite
python -m pytest tests/test_engine.py -v

# Or manually (if pytest unavailable)
python tests/test_engine.py
```

32 tests covering config, taxonomy, detector, interpreters, pipeline, and drift monitoring.

## Data

Includes 100K sample transaction records (`cross_sell_sample_data.csv`):
- 4,978 customers
- 12 months of activity (Feb 2024 – Jan 2025)
- Product signals: mortgages, auto loans, credit cards, student loans, rent, insurance
- Noise: retail, groceries, dining, travel, utilities

## Performance

- Detection runtime: ~8 seconds on 100K records
- Detections: ~6,080 external relationships across ~3,767 customers
- Confidence mix: 96.5% High, 3.5% Medium, 0% Low (default filter excludes Low)

## Change Management

Per spec Section 9:

- **Stable logic** (in code): Cadence detection, stability metrics, tenure calculations
- **Volatile inputs** (in config): Merchant aliases, thresholds, amount bands
- **New rule discovery**: Unclassified recurring patterns ranked weekly in Tuning & QA view
- **Drift monitoring**: KS tests, PSI, volume trends

## License

Proprietary — internal use only.
