from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import main
print("[DEBUG] Running main.py from:", main.__file__)


# Add your project path so Airflow can find main.py
sys.path.append("/Users/calebbenton/Downloads/ml_pipeline_project")

# Import your actual ML pipeline function
from main import main as ml_main

# Default args
default_args = {
    'owner': 'calebbenton',
    'depends_on_past': False,
    'email': ['youremail@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
with DAG(
    dag_id='ml_pipeline',
    default_args=default_args,
    description='Run my ML pipeline and update PostgreSQL',
    schedule_interval='@daily',  # or '@hourly' for more frequent runs
    start_date=datetime(2025, 7, 29),
    catchup=False,
    tags=['ml', 'pipeline'],
) as dag:

    def run_ml_pipeline():
        print("[AIRFLOW] Triggering full ML pipeline now...")
        ml_main()
        print("[AIRFLOW] Pipeline finished successfully.")

    run_pipeline_task = PythonOperator(
        task_id='run_ml_pipeline',
        python_callable=run_ml_pipeline
    )
