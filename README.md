# ChronosCast: Sales Forecasting & Inventory Planning

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

ChronosCast is a production-grade time-series forecasting and inventory planning engine. Built to handle highly volatile and sparse retail data, this repository implements a comprehensive, end-to-end analytical pipeline based on industry-standard methodologies (e.g., PwC forecasting architectures). 

The platform utilizes a hybrid blend of statistical models, machine learning, and deep learning—unified by a fault-tolerant, dynamically weighted ensemble—to generate highly accurate SKU-level demand forecasts.

---

## 🎯 Architecture & Methodology

This project solves the "intermittent demand" problem common in retail data through a rigid 5-step analytical architecture:

1. **Sparsity-Driven Granularity Selection**
   Granular data (Article x Site) is often too sparse to forecast mathematically. The engine dynamically evaluates the hierarchical sparsity of the data and automatically selects the most granular, mathematically viable aggregation level (`Category x Brand x Site`).

2. **Data Cleansing & Winsorization**
   Before modeling, time-series data is protected against extreme anomalies (e.g., data-entry errors, pandemic hoarding) using Interquartile Range (IQR) Winsorization to cap outliers.

3. **Hybrid Model Training & Cross-Validation**
   The engine concurrently trains 6 discrete models:
   * **Statistical:** ARIMA, SARIMA, ARIMAX, ETS
   * **Machine Learning:** Facebook Prophet
   * **Deep Learning:** PyTorch LSTM
   
   To prevent data leakage, models are evaluated using a strict, expanding-window temporal holdout mechanism.

4. **Fault-Tolerant Ensembling**
   Instead of relying on a single model, the engine utilizes an inverse-RMSE weighted ensemble. The ensemble dynamically routes around model failures (e.g., strict seasonal convergence failures in SARIMA) and suppresses the noise of weaker models.

5. **Top-Down Volume Decomposition**
   Aggregate forecasts are mathematically reconciled back down to the granular SKU level using 18-month historical contribution ratios, ensuring zero volume loss.

---

## 📊 Performance Metrics

On the final validation fold against the `Category x Brand x Site` grain, the architecture yielded the following RMSE (Root Mean Square Error) results:

* **Baseline (Seasonal Naive):** 5.15
* **Prophet:** 4.99
* **ETS (Best Standalone):** 3.83 (25.6% improvement over baseline)
* **ChronosCast Ensemble:** **3.33 (35.3% improvement over baseline)**

The ensemble successfully stabilized volatile DL predictions and generated a highly actionable error margin of ~3.3 units per month.

---

## 🏗️ Repository Structure

```text
chronoscast/
├── data/
│   ├── raw/             # Immutable source files (transactions, dimensions, exogenous)
├── docs/                # Project documentation and baseline schemas
├── src/
│   └── chronoscast/
│       ├── config.py             # Centralized parameters and file paths
│       ├── data_ingestion.py     # Pandas multi-sheet loaders
│       ├── data_preparation.py   # Cleansing, IQR capping, and merging
│       ├── eda.py                # Deterministic hierarchical sparsity analyzer
│       ├── forecasting_data.py   # Target grain zero-filling and feature preservation
│       ├── forecasting_models.py # Unified interfaces for stats, ML, and DL models
│       ├── validation.py         # Expanding window chronological splitting
│       ├── ensemble_model.py     # Dynamic inverse-RMSE weighting
│       └── decomposition.py      # Top-down volume allocation
├── main.py                       # The main orchestration entry point
└── tests/                        # Pytest suite verifying pipeline logic
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.11+
* `uv` or `pip` for dependency management

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   uv sync
   ```

### Execution
To run the entire end-to-end analytical pipeline (Ingestion ➔ Outlier Capping ➔ Grain Selection ➔ Model Training ➔ Validation ➔ Ensembling ➔ Decomposition):

```bash
uv run python main.py
```

The final decomposed forecasts will be saved to your configured output path.
