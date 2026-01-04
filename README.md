# Dynamic Pricing Pipeline using Apache Airflow


## 1. Project Overview

Dynamic pricing refers to the systematic adjustment of prices based on historical transactions, customer behavior, and temporal patterns. In real-world data-driven organizations, such strategies are implemented through **automated data engineering pipelines** rather than manual analysis.

This project presents a **complete end-to-end Data Engineering pipeline** orchestrated using **Apache Airflow**. The pipeline ingests raw e-commerce data, performs data cleaning and feature engineering, trains a machine learning model, and automatically generates analytical outputs.

All components are designed to answer **five predefined research questions (RQ1–RQ5)** in a reproducible and systematic manner.

**Primary focus areas:**
- Pipeline design and orchestration  
- Automation and reproducibility  
- Clear mapping between research questions and generated outputs  

---

## 2. Research Questions

### RQ1 – Effectiveness of Automated Feature Engineering

**Question:**  
How does automated feature engineering influence predictive performance in a dynamic pricing pipeline?

**Generated Outputs:**
- `RQ1_Fig1.pdf`
- `RQ1_Table1.xlsx`
- `RQ1_Fig1_test.pdf`
- `RQ1_Table1_test.xlsx`

---

### RQ2 – Generalization Across Data Segments

**Question:**  
Does the trained model generalize consistently across different data segments?

**Generated Outputs:**
- `RQ2_Fig1.pdf`
- `RQ2_Table1.xlsx`

---

### RQ3 – Model Comparison and Explainability

**Question:**  
How do different regression models compare in terms of predictive accuracy and training time?

**Models Compared:**
- Linear Regression  
- Random Forest Regressor  
- XGBoost Regressor  

**Generated Outputs:**
- `RQ3_Fig1.pdf`
- `RQ3_Fig2.pdf`
- `RQ3_Table1.xlsx`

---

### RQ4 – Fairness and Bias Analysis

**Question:**  
Are there observable differences in prediction errors across different data segments that may indicate bias?

**Generated Outputs:**
- `RQ4_Fig1.pdf`
- `RQ4_Table1.xlsx`

---

### RQ5 – Stability and Risk Assessment

**Question:**  
How stable are the model predictions across different value ranges?

**Generated Outputs:**
- `RQ5_Fig1.pdf`
- `RQ5_Table1.xlsx`

---

## 3. Dataset Description

- **Source:** Kaggle  
- **Dataset:** E-commerce Order Dataset  
- **Link:**  
  https://www.kaggle.com/datasets/bytadit/ecommerce-order-dataset  

The dataset contains transactional information including:
- Orders  
- Customers  
- Payments  
- Products  

Raw datasets are **not committed to the repository** due to size considerations. Once placed locally, the pipeline can be executed to reproduce all results.

---

## 4. Pipeline Architecture (ETL Flow)

### 4.1 Data Ingestion and Joining (`01_ingest_join.py`)
- Reads multiple raw CSV files for training and testing splits  
- Input files include customers, orders, order items, payments, and products  
- Data is joined using keys such as `order_id`, `customer_id`, and `product_id`  

**Output files:**
- `model_train_joined.csv`
- `model_test_joined.csv`

---

### 4.2 Data Cleaning and Feature Engineering (`02_clean_features.py`)
- **Missing value handling:**
  - Numerical features → median imputation  
  - Categorical features → `"unknown"`  
- **Temporal feature extraction:**
  - Purchase hour  
  - Day of week  
  - Month  
- **Derived feature creation:**
  - Shipping cost as a percentage of price  

**Output files:**
- `model_train_features.csv`
- `model_test_features.csv`

---

### 4.3 Model Training (`03_train.py`)
- Trains an **XGBoost Regressor** using a Scikit-learn Pipeline  
- **Preprocessing includes:**
  - One-Hot Encoding for categorical variables  
  - Numerical passthrough  

**Target variable:**
- `payment_value`

**Generated artifacts:**
- Trained model → `data/processed/models/model.joblib`  
- Training metrics → `data/processed/models/train_metrics.csv`

---

### 4.4 Research Output Generation (`04_make_outputs.py`)
- Generates figures and tables corresponding to **RQ1–RQ5**  
- Outputs are saved automatically to:
  - `outputs/figures/`
  - `outputs/tables/`

---
### 5  Repository Structure and Directory Explanation

* 5.1 Set up the environment

```bash
git clone https://github.com/powerbyadu/dynamic-pricing-airflow-rq1-rq5.git
cd dynamic-pricing-airflow-rq1-rq5
```

```bash
python3 -m venv airflow_venv
source airflow_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```



* 5.2 Place the dataset
  ```text
    data/
  ├── train/
  │   ├── df_Customers_train.csv
  │   ├── df_Orders_train.csv
  │   ├── df_OrderItems_train.csv
  │   ├── df_Payments_train.csv
  │   └── df_Products_train.csv
  └── test/
      ├── df_Customers_test.csv
      ├── df_Orders_test.csv
      ├── df_OrderItems_test.csv
      ├── df_Payments_test.csv
      └── df_Products_test.csv
  ```

* 5.3 Execute pipeline scripts in order
  * Data ingestion and joining
    ```bash
    python src/01_ingest_join.py --split train
    python src/01_ingest_join.py --split test
    ```

  * Data cleaning and feature engineering
    ```bash
    python src/02_clean_features.py --split train
    python src/02_clean_features.py --split test
    ```

  * Model training
    ```bash
    python src/03_train.py
    ```

  * Generate research outputs
    ```bash
    python src/04_make_outputs.py --split train
    python src/04_make_outputs.py --split test
    ```
  

* 5.4 Repository Structure
```text
dynamic-pricing-airflow-rq1-rq5/
├── dags/
│ └── dynamic_pricing_dag.py
├── src/
│ ├── 01_ingest_join.py
│ ├── 02_clean_features.py
│ ├── 03_train.py
│ └── 04_make_outputs.py
├── outputs/
│ ├── figures/
│ └── tables/
├── requirements.txt
├── start_env.sh
└── README.md
```

* 5.5 Folder purpose explanation:

- `dags/` – Contains the Apache Airflow DAG definition that orchestrates the pipeline  
- `src/` – Contains Python scripts for each pipeline stage (ingestion, processing, training, outputs)  
- `outputs/` – Stores all generated figures and tables for RQ1–RQ5  
- `requirements.txt` – Lists all Python dependencies  
- `start_env.sh` – Optional helper script for environment setup  
- `README.md` – Project documentation  

---

## 6. Environment Specifications

- **Python Version:** 3.11.9  
- **Operating System:** WSL (Ubuntu on Windows)

---

## 7. Reproducibility Instructions

This section describes the exact steps required to reproduce the pipeline execution and regenerate the analytical outputs using Apache Airflow.

* 7.1 Repository Setup
  Clone the repository and navigate to the project directory:

  ```bash
  git clone https://github.com/powerbyadu/dynamic-pricing-airflow-rq1-rq5.git
  cd dynamic-pricing-airflow-rq1-rq5
  ```

* 7.2 Python Environment Setup
  Create and activate a virtual environment to isolate dependencies:
  
  ```bash
  python3 -m venv airflow_venv
  source airflow_venv/bin/activate
  ```

  Upgrade pip and install required packages:

  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

* 7.3 Data Placement
```text
 data/
 ├── train/
 │   ├── df_Customers_train.csv
 │   ├── df_Orders_train.csv
 │   ├── df_OrderItems_train.csv
 │   ├── df_Payments_train.csv
 │   └── df_Products_train.csv
 └── test/
     ├── df_Customers_test.csv
     ├── df_Orders_test.csv
     ├── df_OrderItems_test.csv
     ├── df_Payments_test.csv
     └── df_Products_test.csv
```
* 7.4 Airflow Initialization
  Set the Airflow home directory within the project and initialize the metadata database:

  ```bash
  export AIRFLOW_HOME="$(pwd)/airflow_home"
  mkdir -p "$AIRFLOW_HOME"
  airflow db init
  ```

* 7.5 Starting Airflow Services
  Start the Airflow webserver and scheduler in separate terminals:

 * Termina 1
   ```bash
   airflow webserver
   ```
   
 * Terminal 2
   ```bash
   airflow scheduler
   ```
  
* 7.7 Executing the Pipeline
  Open the Airflow web interface:

  ```bash
  http://localhost:8080
  ```

* 7.6 Notes on Environment Configuration
  The Airflow DAG references a project directory path corresponding to the local development
  environment. When running the project in a different directory, this path can be updated in:

  ```bash
  dags/dynamic_pricing_dag.py
  ```

  This configuration change does not affect the pipeline logic or analytical results.










