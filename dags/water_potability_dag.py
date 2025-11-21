from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'water_potability_pipeline',
    default_args=default_args,
    description='A simple MLOps pipeline for Water Potability',
    schedule_interval=timedelta(days=1),
)

# Task 1: Prepare Data
prepare_data = BashOperator(
    task_id='prepare_data',
    bash_command='python src/data_prep.py data/raw/water_potability.csv data/processed',
    dag=dag,
    cwd='/opt/airflow/dags/repo' # Adjust based on where the repo is mounted in Airflow
)

# Task 2: Train ANN Model
train_ann = BashOperator(
    task_id='train_ann',
    bash_command='python src/train.py data/processed ann',
    dag=dag,
    cwd='/opt/airflow/dags/repo'
)

# Task 3: Train CNN Model
train_cnn = BashOperator(
    task_id='train_cnn',
    bash_command='python src/train.py data/processed cnn',
    dag=dag,
    cwd='/opt/airflow/dags/repo'
)

prepare_data >> [train_ann, train_cnn]
