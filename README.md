# Dynamic Pricing Pipeline with Apache Airflow

This project implements an end-to-end **Dynamic Pricing Machine Learning Pipeline** orchestrated using **Apache Airflow**.  
The pipeline covers **data ingestion, feature engineering, model training, inference, and research output generation** aligned with **RQ1–RQ5**.

The primary objective is to demonstrate **data engineering and ML pipeline orchestration**, not just model accuracy.

---

## 📌 Project Structure

dynamic-pricing-airflow/
│
├── airflow_home/ # Airflow metadata (not committed)
│
├── data/
│ ├── train/ # Raw training CSVs (not committed)
│ ├── test/ # Raw test CSVs (not committed)
│ └── processed/ # Joined & feature-engineered data
│
├── src/
│ ├── 01_ingest_join.py # Data ingestion & table joins
│ ├── 02_clean_features.py # Feature engineering
│ ├── 03_train.py # Model training & persistence
│ └── 04_make_outputs.py # RQ1–RQ5 analysis & outputs
│
├── dags/
│ └── dynamic_pricing_dag.py # Airflow DAG definition
│
├── outputs/
│ ├── figures/ # Generated PDF figures (RQ1–RQ5)
│ └── tables/ # Generated Excel tables (RQ1–RQ5)
│
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md

yaml
Copy code

---

## 🔁 Airflow DAG Overview

**DAG Name:** `dynamic_pricing_pipeline`

### Pipeline Flow
ingest_train → features_train → train_model → outputs_train
ingest_test → features_test → outputs_test
train_model → outputs_test

gherkin
Copy code

### Key Design Decisions
- **Train and test pipelines are separated**
- **Model training occurs once**
- **Test inference waits for trained model**
- Prevents data leakage
- Mirrors real-world ML deployment pipelines

---

## 📊 Research Questions Implemented

| RQ | Description | Outputs |
|----|------------|---------|
| RQ1 | Baseline model performance | MAE figure & table |
| RQ2 | Impact of temporal features | Baseline vs temporal comparison |
| RQ3 | Model comparison | Linear vs RF vs XGBoost |
| RQ4 | Feature importance | Top contributing features |
| RQ5 | Model stability | MAE across price segments |

---

## 📂 Output Artifacts

Generated automatically and stored as files:

### Figures
outputs/figures/
├── RQ1_Fig1.pdf
├── RQ2_Fig1.pdf
├── RQ3_Fig1.pdf
├── RQ3_Fig2.pdf
├── RQ4_Fig1.pdf
└── RQ5_Fig1.pdf

clean
Copy code

### Tables
outputs/tables/
├── RQ1_Table1.xlsx
├── RQ2_Table1.xlsx
├── RQ3_Table1.xlsx
├── RQ4_Table1.xlsx
└── RQ5_Table1.xlsx

yaml
Copy code

---

## 🚀 How to Run (WSL / Ubuntu)

### 1️⃣ Activate environment
```bash
source airflow_venv/bin/activate
export AIRFLOW_HOME=~/dynamic-pricing-airflow/airflow_home
2️⃣ Initialize Airflow
bash
Copy code
airflow db init
3️⃣ Start Airflow
bash
Copy code
airflow webserver --port 8080
airflow scheduler
4️⃣ Trigger the DAG
Open browser:

dts
Copy code
http://localhost:8080
Trigger dynamic_pricing_pipeline

🧪 Manual Execution (Optional)
bash
Copy code
python3 src/01_ingest_join.py --split train
python3 src/02_clean_features.py --split train
python3 src/03_train.py
python3 src/04_make_outputs.py --split train
🎓 Academic Note
This project emphasizes:

Data engineering robustness

ML lifecycle orchestration

Reproducible research outputs

Industry-standard workflow design

Rather than maximizing accuracy alone, the focus is on pipeline correctness and reliability.