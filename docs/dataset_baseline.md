# Stage 0: Dataset Baseline & Project Context

## 1. Project Context
ChronosCast is a Sales Forecasting and Inventory Planning platform for an electronics major in South-East Asia. The business faces challenges in inventory planning due to poor visibility into future sales. 
The objective is to establish a rigorous, production-grade forecasting system capable of predicting future demand (quantity of articles sold per shop per month) approximately 2-4 months into the future. 
This repository establishes the foundation of that system using the provided synthetic datasets and the core methodology outlined in the PwC solution documentation.

## 2. Source Material Summary
The project follows a standard 5-step methodological backbone established by the PwC solution materials:
1. **Data Loading and EDA**
2. **Processing and Sparsity**
3. **Model Building** (using statistical models like ARIMA/SARIMA/ETS/ARIMAX and ML models like Prophet/LSTM)
4. **Ensembling** (combining outputs from multiple models)
5. **Forecasting and Write Back**

A key component of this methodology involves a **top-down decomposition** logic. Because granular demand (Article × Site) is sparse, the system will forecast at a higher aggregation level (e.g., Category × Site) where the data is continuous. These aggregate forecasts are then disaggregated down to the granular level by applying historical sales contribution ratios over a recent window (e.g., 18 months).

## 3. Dataset Inventory
The raw source layer is immutable and consists of the following 6 CSV files:

| Dataset | Rows | Cols | Primary Role |
|---|---|---|---|
| `salesTran.csv` | 10,477 | 14 | Primary fact table. Records daily transactional demand. |
| `dimProduct.csv` | 24 | 6 | Dimension table defining article hierarchies and categorization. |
| `dimProfitCenter.csv` | 8 | 11 | Dimension table for organizational profit centers. |
| `dimShowRoom.csv` | 9 | 7 | Dimension table mapping physical stores (sites). |
| `weatherIndex.csv` | 43 | 4 | Exogenous features (temperature, humidity) at a monthly level. |
| `economicIndex.csv` | 4 | 4 | Exogenous features (CPI, GDP growth) at an annual level. |

## 4. Schema Summary
- **salesTran**: `billingDate`, `articleNo`, `articleDescription`, `profitCenter`, `generalNamePro`, `brand`, `product`, `site`, `division`, `zone`, `territory`, `salesTypeDesc`, `billQty`, `basePrice`
- **dimProduct**: `profitCenter`, `brand`, `category`, `articleNo`, `articleDescription`, `product`
- **dimProfitCenter**: `id`, `refId`, `profitCenter`, `validToDate`, `city`, `countryKey`, `companyCode`, `companyName`, `validFrom`, `generalNameProfitCenter`, `createdOn`
- **dimShowRoom**: `id`, `refId`, `showRoom`, `division`, `zone`, `territory`, `plantDescription`
- **weatherIndex**: `month`, `city`, `temperature`, `humidity`
- **economicIndex**: `year`, `country`, `cpi`, `gdp_growth`

## 5. Data Types
- **Dates/Times**: `billingDate` (salesTran) and `month` (weatherIndex) are stored as raw integers (e.g., 20220101) and must be explicitly parsed downstream. `year` (economicIndex) is also an integer.
- **Identifiers**: Most textual categorical fields (brand, category, site, profitCenter) correctly load as strings. `articleNo` loads as an integer.
- **Numerics**: Quantities (`billQty`) are integers, and prices/rates (`basePrice`, `temperature`, `cpi`, `gdp_growth`) are floats.

## 6. Temporal Coverage
- **Historical Sales (`salesTran`)**: 2022-01-01 to 2024-07-31 (~31 months)
- **Weather (`weatherIndex`)**: 2022-01-01 to 2025-07-01 (~42 months)
- **Economic (`economicIndex`)**: 2022 to 2025 (4 years)
*Note: The exogenous datasets extend approximately 1 year beyond the historical sales data. This lookahead is structurally necessary for feeding future values into multivariate forecasting models.*

## 7. Data Quality
- **Missing Values**: 0 nulls across all datasets.
- **Duplicates**: `salesTran.csv` contains 1 duplicate row. All other files have 0 duplicates.

## 8. Referential Integrity
- `salesTran.articleNo` → `dimProduct.articleNo` (100% match)
- `salesTran.profitCenter` → `dimProfitCenter.profitCenter` (100% match)
- `salesTran.site` → `dimShowRoom.refId` (100% match, note that it must join to `refId` and NOT `showRoom`).

## 9. Important Semantic Findings
- **Target Variable**: The target for the forecasting algorithm is `billqty` (sales quantity), not `basePrice`.
- **Product Terminology Conflict**: In `salesTran`, the column `product` conceptually represents a broad category. However, in `dimProduct`, the column `product` represents a brand-line label. These two columns represent distinctly different entities and must be handled with care during downstream merges to avoid namespace collisions.

## 10. Preliminary Sparsity
- **Article × Site (Daily)**: 95.17% missing days.
- **Article × Site (Monthly)**: 37.87% missing months.
- **ProfitCenter × Site (Monthly)**: 8.02% missing months.
*Observation*: The data clearly demonstrates the business problem outlined in the case study. The granular Article × Site matrix is highly sparse, while higher aggregations like ProfitCenter × Site provide a much more continuous historical signal.

## 11. Risks / Ambiguities
- **Temporal Alignment**: `economicIndex` is provided at an annual level, while `weatherIndex` is monthly, and `salesTran` is daily. Building the "Monthly Master Dataset" will require explicitly broadcasting/forward-filling the annual data into 12 distinct months.
- **Data Integrity**: The raw CSV duplicate in the transaction table must be handled exclusively in the ingestion/preparation downstream, as raw files are immutable.

## 12. Stage 0 Conclusion
The project source data is intact, well-understood, and correctly matches the structure described by the case study. The repository structure is successfully baselined with Python initialized. The project is fully ready to proceed to Stage 1.

## 13. Recommended Stage 1 Scope
- Initialize the `src/` modules (`config.py`, `data_ingestion.py`, `data_preparation.py`, `utils.py`).
- Implement the `DataIngestor` class to load the raw CSVs.
- Implement the `DataPreparer` class to clean the data (handle duplicates, parse dates) and merge the dimensions and exogenous variables into a cohesive Monthly Master Dataset.
- Write tests in `tests/` validating this specific logic.
