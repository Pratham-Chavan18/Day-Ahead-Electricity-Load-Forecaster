# Day-Ahead Electricity Load Forecaster

An end-to-end machine learning pipeline that forecasts hourly household electricity consumption 24 hours into the future using gradient boosted decision trees (**XGBoost**).

---

## 📌 Project Overview & Objectives

Smart meters collect high-frequency electricity usage data. However, RAW minute-level data is noisy, volatile, and prone to missing values. This project constructs an industrial-grade data processing and forecasting pipeline designed to:
1. **Clean & Handle Missing Data**: Parse raw minute-level reads (~2.07 million rows over ~4 years) from the **UCI Individual Household Electric Power Consumption** dataset.
2. **Resample to Hourly Blocks**: Apply domain-specific mathematical aggregations (summing energy metrics, averaging steady-state voltage/intensity metrics) to smooth out minute-to-minute noise and establish a stable 24-hour forecasting horizon.
3. **Prevent Data Leakage**: Enforce strict chronological train/test splitting (training on pre-2010 data, testing on 2010 data) and drop concurrent instantaneous metrics during feature engineering.
4. **Engineer Advanced Time-Series Features**: Capture daily and weekly seasonality via cyclical Sine/Cosine encoding, historical same-hour lag features, and rolling trend statistics.
5. **Beat Naive Baselines**: Validate model performance against the industry standard *"Tomorrow looks like Today"* (24-hour lag) baseline using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

---

## 📊 Evaluation & Model Performance

The evaluation was conducted on a strictly isolated, chronological test set starting **January 1, 2010** (`len(test_set) = 7,639 hours`).

| Model | MAE (kW-h / hr) | RMSE (kW-h / hr) | Error Reduction vs Baseline |
| :--- | :---: | :---: | :---: |
| **Naive Baseline (*24h Lag*)** | `33.5708` | `49.3970` | *Reference Baseline* |
| **XGBoost Regressor** | **`27.3785`** | **`37.0858`** | **+18.4% MAE / +24.9% RMSE** |

### Key Findings
* **Massive Error Reduction**: The XGBoost model successfully cuts down forecasting error by over **18.4% MAE** and **24.9% RMSE** compared to the naive baseline.
* **Diurnal Accuracy**: Cyclical time encoding (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`) combined with historical same-hour averages (`same_hr_mean_7d`) enables the tree estimator to anticipate sharp evening consumption spikes and early-morning baseline dips with high precision.

---

## 📈 Visual Proof of Performance

The generated output chart (`forecast_results.png`) showcases model accuracy across three visual dimensions:
1. **Time-Series Sample Window**: Actual vs Naive Baseline vs XGBoost forecast across a representative 10-day window in January 2010.
2. **Residual Distribution**: Error histogram confirming tighter clustering around `0` error for XGBoost compared to the baseline spread.
3. **Metric Summary Bar Chart**: Visual comparison of MAE and RMSE scores.

![Forecast Results](forecast_results.png)

---

## 🧠 Real-World Business & Utility Context

### 1. What would you change if you had to forecast for hundreds of thousands of meters at once instead of one?

Scaling from a single household meter to **100,000+ smart meters** introduces massive computational, storage, and architectural shifts. Moving from single-meter scripting to enterprise MLOps requires three core changes:

#### A. Distributed Data Engineering & Storage
* **Out-of-Core Processing**: In-memory Pandas arrays fail when handling billions of rows across thousands of meters. I would migrate data preprocessing to **Apache Spark (PySpark)** or **Polars** on a distributed compute cluster.
* **Storage Architecture**: Raw smart meter telemetries would be ingested into columnar cloud data lakes (e.g., **Delta Lake**, **Snowflake**, or **AWS S3 + Athena**) or specialized time-series databases (**TimescaleDB** or **ClickHouse**) partitioned by `meter_id` and timestamp.

#### B. Global Forecasting Models vs Local Models
* **Global Model Paradigm**: Maintaining 100,000 individual local XGBoost models creates severe operational debt and training bottlenecks. Instead, I would train a unified **Global Forecasting Model** (such as **LightGBM**, **Temporal Fusion Transformer (TFT)**, or **DeepAR**) across all meters simultaneously.
* **Static Metadata Injection**: To allow a single global model to capture distinct household behaviors, static features—such as `meter_id` target encodings, geographic zone, tariff class, square footage, and appliance presence—are concatenated with dynamic lags. This allows the global model to cross-learn general grid seasonality while customizing baselines for individual homes.

#### C. Automated MLOps & Orchestration
* **Batch Inference & Pipeline Automation**: Containerize workflows via **Docker** and orchestrate nightly batch pipelines via **Apache Airflow** or **Kubeflow**.
* **Monitoring & Drift Detection**: Implement continuous monitoring (**Evidently AI** or **Prometheus**) to detect concept drift (e.g., seasonal weather shifts, EV adoption) and trigger automated model retraining.

---

### 2. Do you think a model like this is used in practice by utilities, or would something simpler win?

In utility operations, the choice between complex ML models (like XGBoost/Transformers) and simpler heuristics depends entirely on the **aggregation hierarchy** and the **Return on Investment (ROI)** of marginal accuracy:

#### A. Grid & Substation Level (Complex Models Dominate)
At the aggregated transmission, distribution, or substation level, complex machine learning models (**XGBoost**, **LSTMs**, **Gradient Boosted Trees**) are the undisputed industry standard.
* **Financial Impact**: When forecasting load for an entire city or power grid, even a **0.5% – 1.0% reduction in forecasting error (MAPE/MAE)** translates to millions of dollars saved annually in peak-demand generation, spinning reserves, and wholesale energy market balancing costs.
* **Compute Validation**: Cloud infrastructure and GPU training costs are completely negligible compared to the massive financial and grid-stability savings achieved by superior accuracy.

#### B. Individual Edge / Smart Meter Level (Simpler Models Win)
If the utility requires forecasts at the individual residential meter level—particularly when executed locally on **Edge devices** or smart meters—simpler lightweight models win.
* **Compute & Resource Constraints**: Millions of residential meters lack the memory and computational capacity to execute heavy gradient boosted trees or neural network ensembles locally.
* **Optimal ROI**: For individual homes, lightweight heuristics—such as **Dynamic Exponential Rolling Averages**, **Holt-Winters Smoothing**, or **SARIMA**—provide "good enough" accuracy at a fraction of the computational and storage cost. The noise at an individual household level (e.g., random vacation days, irregular cooking times) is often irreducible by complex models, making simpler heuristics far more cost-effective.

---

## 🛠️ Project Structure & Setup

```text
electricity-load-forecaster/
│
├── data/
│   └── household_power_consumption.txt    # Raw UCI dataset (2.07M rows)
│
├── main.py                                # Core end-to-end pipeline script
├── forecast_results.png                   # Output visualization chart
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
```

### Quickstart Guide

1. **Clone & Navigate**:
   ```bash
   git clone <repo_url>
   cd electricity-load-forecaster
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**:
   Place the unzipped `household_power_consumption.txt` file into the `data/` directory.
   *(Download link: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption))*

4. **Execute Pipeline**:
   ```bash
   python main.py
   ```
   *The script will load the data, engineer features, train the XGBoost model, print final metrics, and generate `forecast_results.png` automatically.*
