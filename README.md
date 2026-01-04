1. Project Overview
Dynamic pricing refers to the systematic adjustment of prices based on historical transactions, customer behavior, and temporal patterns. In real-world data-driven organizations, such pricing strategies are implemented through automated data engineering pipelines rather than manual analysis.
This project presents a complete end-to-end Data Engineering pipeline orchestrated using Apache Airflow. The pipeline ingests raw e-commerce data, performs data cleaning and feature engineering, trains a machine learning model, and automatically generates analytical outputs.
All components are designed to answer five predefined research questions (RQ1–RQ5) in a reproducible and systematic manner.
The emphasis of this project is on:
Pipeline design and orchestration
Automation and reproducibility
Clear mapping between research questions and generated outputs
3. Research Questions
RQ1 – Effectiveness of Automated Feature Engineering
Question:
How does automated feature engineering influence predictive performance in a dynamic pricing pipeline?
Generated Outputs:
RQ1_Fig1.pdf
RQ1_Table1.xlsx
RQ1_Fig1_test.pdf
RQ1_Table1_test.xlsx





RQ2 – Generalization Across Data Segments
Question:
Does the trained model generalize consistently across different data segments?
Generated Outputs:
RQ2_Fig1.pdf
RQ2_Table1.xlsx

RQ3 – Model Comparison and Explainability
Question:
How do different regression models compare in terms of predictive accuracy and training time?
Models Compared:
Linear Regression
Random Forest Regressor
XGBoost Regressor
Generated Outputs:
RQ3_Fig1.pdf
RQ3_Fig2.pdf
RQ3_Table1.xlsx

RQ4 – Fairness and Bias Analysis
Question:
Are there observable differences in prediction errors across different data segments that may indicate bias?
Generated Outputs:
RQ4_Fig1.pdf
RQ4_Table1.xlsx

RQ5 – Stability and Risk Assessment
Question:
How stable are the model predictions across different value ranges?
Generated Outputs:
RQ5_Fig1.pdf
RQ5_Table1.xlsx

4. Dataset Description
Source: Kaggle
Dataset: E-commerce Order Dataset
Link:
https://www.kaggle.com/datasets/bytadit/ecommerce-order-dataset
The dataset contains transactional information including:
Orders
Customers
Payments
Products
Raw datasets are not committed to the repository due to size considerations. Once placed locally, the pipeline can be executed to reproduce all results.

5. Pipeline Architecture (ETL Flow)
5.1 Data Ingestion and Joining (01_ingest_join.py)
Reads multiple raw CSV files for training and testing splits
Input files include customers, orders, order items, payments, and products
Data is joined using keys such as order_id, customer_id, and product_id
Output files:
model_train_joined.csv
model_test_joined.csv

5.2 Data Cleaning and Feature Engineering (02_clean_features.py)
Missing value handling:
Numerical features → median imputation
Categorical features → "unknown"
Temporal feature extraction:
Purchase hour
Day of week
Month
Derived feature creation:
Shipping cost as a percentage of price
Output files:
model_train_features.csv
model_test_features.csv

5.3 Model Training (03_train.py)
Trains an XGBoost Regressor using a Scikit-learn Pipeline
Preprocessing includes:
One-Hot Encoding for categorical variables
Numerical passthrough
Target variable:
payment_value
Generated artifacts:
Trained model → data/processed/models/model.joblib
Training metrics → data/processed/models/train_metrics.csv

5.4 Research Output Generation (04_make_outputs.py)
Generates figures and tables corresponding to RQ1–RQ5
Implements:
Temporal feature impact analysis
Model comparison
Feature importance extraction
Stability analysis across segments
Outputs are saved automatically to:
outputs/figures/
outputs/tables/

6. Machine Learning Model
Primary Model: XGBoost Regressor
Rationale for Selection:
Strong performance on structured tabular data
Ability to model non-linear relationships
Computational efficiency suitable for pipeline execution
Supports interpretability via feature importance
XGBoost serves as the primary model, while additional models are evaluated for comparative analysis in RQ3.

7. Repository Structure
dynamic-pricing-airflow-rq1-rq5/
│
├── dags/
│   └── dynamic_pricing_dag.py
│
├── src/
│   ├── 01_ingest_join.py
│   ├── 02_clean_features.py
│   ├── 03_train.py
│   └── 04_make_outputs.py
│
├── outputs/
│   ├── figures/
│   └── tables/
│
├── requirements.txt
├── start_env.sh
└── README.md


8. Environment Specifications
Python Version: 3.11.9
Operating System: WSL (Ubuntu on Windows)

9. Reproducibility Instructions
9.1 Setup
git clone https://github.com/powerbyadu/dynamic-pricing-airflow-rq1-rq5.git
cd dynamic-pricing-airflow-rq1-rq5
python3 -m venv airflow_venv
source airflow_venv/bin/activate
pip install -r requirements.txt

9.2 Airflow Initialization
export AIRFLOW_HOME="$(pwd)/airflow_home"
mkdir -p "$AIRFLOW_HOME"
airflow db init

9.3 Run Airflow
airflow webserver
airflow scheduler

Trigger the DAG manually via the Airflow UI:
DAG ID: dynamic_pricing_pipeline
URL: http://localhost:8080

10. Airflow Pipeline Logic
Training Flow
Ingest → Feature Engineering → Model Training → Output Generation

Testing Flow
Ingest → Feature Engineering → Output Generation

This separation ensures consistent evaluation and avoids data leakage.

11. Note on Airflow Project Path
The Airflow DAG currently references a fixed project directory path corresponding to the local development environment:
/home/yadu/dynamic-pricing-airflow

When executing the project in a different directory structure, this path can be updated in:
dags/dynamic_pricing_dag.py

This configuration detail does not affect the data processing logic or experimental results.

12. Common Issues and Troubleshooting
DAG not visible in Airflow UI:
Verify AIRFLOW_HOME and restart scheduler/webserver.
Missing data files:
Ensure required CSV files are present in data/train and data/test.
Dependency errors:
Reinstall packages using requirements.txt.
Path-related errors:
Update the project directory path in the DAG file if necessary.

13. Academic Declaration
This repository was developed exclusively for academic submission as part of a Data Engineering course.
All figures and tables are generated programmatically, and no manual modification of results was performed.
