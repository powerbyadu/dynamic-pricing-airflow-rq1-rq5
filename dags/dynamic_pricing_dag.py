from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/home/yadu/dynamic-pricing-airflow"

def cd(cmd: str) -> str:
    return f'cd "{PROJECT_DIR}" && {cmd}'

with DAG(
    dag_id="dynamic_pricing_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["dynamic-pricing"],
) as dag:

    ingest_train = BashOperator(task_id="ingest_train", bash_command=cd("python3 src/01_ingest_join.py --split train"))
    feat_train   = BashOperator(task_id="features_train", bash_command=cd("python3 src/02_clean_features.py --split train"))
    train_model  = BashOperator(task_id="train_model", bash_command=cd("python3 src/03_train.py"))
    out_train    = BashOperator(task_id="outputs_train", bash_command=cd("python3 src/04_make_outputs.py --split train"))

    ingest_test  = BashOperator(task_id="ingest_test", bash_command=cd("python3 src/01_ingest_join.py --split test"))
    feat_test    = BashOperator(task_id="features_test", bash_command=cd("python3 src/02_clean_features.py --split test"))
    out_test     = BashOperator(task_id="outputs_test", bash_command=cd("python3 src/04_make_outputs.py --split test"))

    ingest_train >> feat_train >> train_model >> out_train
    ingest_test >> feat_test >> out_test
    train_model >> out_test
