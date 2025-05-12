# README: Stock Prediction System

## Overview

This project consists of an ETL pipeline, a machine learning model, and a live predictor module for stock price forecasting. The system relies on configuration and mapping files to customize data processing and model execution.

## Configuration Files

The system includes multiple configuration files that allow users to adjust parameters without modifying the source code.

### 1. ETL Configuration (`etl/config.json`)

This file contains settings for the ETL (Extract, Transform, Load) pipeline.

#### **Key Parameters:**

- **tickers**: List of stock tickers used for data extraction (`AAPL`, `GOOG`, `AMZN`, `MSFT`, `NVDA`).
- **data\_folder**: Directory where raw stock data is stored (`etl/stock_data`).
- **clean\_folder**: Directory where cleaned data is stored (`etl/clean`).
- **output\_folder**: Directory where processed stock data is saved (`etl/stock_data`).
- **Cleaning Methods**: The cleaning methods specify how missing or incorrect data is handled in different financial datasets:
  - **Fill**: Missing values are replaced with an appropriate estimation or interpolation.
  - **Zero**: Missing values are replaced with zero, assuming they represent the absence of a transaction.
  - **Drop**: Columns that are not relevant or have excessive missing data are removed entirely.
- **share\_prices\_cleaning\_methods**: Defines data cleaning methods for share prices.
- **income\_cleaning\_methods**: Defines data cleaning rules for income statements.
- **balance\_cleaning\_methods**: Defines data cleaning rules for balance sheets.
- **cashflow\_cleaning\_methods**: Defines data cleaning rules for cash flow statements.
- **company\_cleaning\_methods**: Defines data cleaning rules for company metadata.

#### **How to Use:**

Modify the values in `config.json` to customize data sources, cleaning methods, and output directories. The ETL pipeline will read this file when executed (`etl/run_etl.py`).

### 2. Machine Learning Configuration (`ml/config_ml.json`)

This file contains settings for training and running machine learning models.

#### **Key Parameters:**

- **tickers**: List of stock tickers used for training (`AAPL`, `GOOG`, `AMZN`, `MSFT`, `NVDA`).
- **data_directory**: Directory where cleaned data is stored (`etl/clean`).
- **models_directory**: Directory where trained models are stored (`ml/models`).
- **scalers_directory**: Directory where scalers are stored (`ml/scalers`).
- **features_directory**: Directory where feature selection data is stored (`ml/features`).
- **alpha**: Regularization strength parameter (default: `0.01`).
- **l1_ratio**: Ratio between L1 and L2 regularization in Elastic Net models (default: `0.5`).
- **top_features**: Number of top features to use in feature selection (default: `10`).
- **test_size**: Proportion of the dataset used for testing (default: `0.2`).
- **random_state**: Random seed for reproducibility (default: `42`).
- **cv**: Number of cross-validation folds (default: `5`).
- **scoring**: Metric used for model evaluation (default: `f1`).
- **hyperparameter_grid**: Defines hyperparameter tuning options for model selection:
  - `n_estimators`: `[50, 100, 200]`
  - `max_depth`: `[null, 10, 20, 30]`
  - `min_samples_split`: `[2, 5, 10]`
  - `min_samples_leaf`: `[1, 2, 4]`
  - `bootstrap`: `[true, false]`
- **log_file**: File for logging machine learning processes (`ml/ml.log`).
- **columns_to_drop**: List of columns dropped before model training (`Date`, `Fiscal Year`, `Fiscal Period`, `Report Date`, `Publish Date`, `Restated Date`).

#### **How to Use:**

Adjust `config_ml.json` to modify the machine learning model’s behavior. The script `ml/run_ml.py` will read this file when training or making predictions.

## Mapping File

### 3. Live Predictor Mapping (`livepredictor/mapping.json`)

This file maps financial and stock market variables to standardized names used in the system. The reason for this mapping is that when downloading historical data from the SimFin API, some of column names in the raw data differ from those used in the live API. Mapping ensures uniformity and compatibility across data processing steps.

#### **How to Use:**

Modify `mapping.json` to add or update stock tickers and financial term mappings. The live predictor (`livepredictor/Livepredictor.py`) will read this file when processing financial data.

## Running the System

### 1. Running the ETL Pipeline

```bash
python etl/run_etl.py
```

### 2. Training the Machine Learning Model

```bash
python ml/run_ml.py
```

### 3. Running Live Stock Predictions

```bash
streamlit run NeuroMarket.py
```

Ensure that the configuration and mapping files are correctly set up before executing the scripts. Modifications to these files allow customization without altering code.

## Contact

For any issues or questions, please refer to the project documentation or contact the development team.

