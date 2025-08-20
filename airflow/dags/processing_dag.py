# Fichero: dags/churn_processing_dag.py

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.variable import Variable

# Importa las funciones específicas para las tareas
from scripts.processing import (
    impute_zeros_task,
    impute_categoricals_task,
    calculate_imputation_values_task,
    apply_imputation_task,
    create_null_flags_task
)

# Carga la configuración desde la Variable de Airflow que creaste
CONFIG = Variable.get("churn_imputation_config", deserialize_json=True)

# Define las rutas de los ficheros. Asegúrate de que la carpeta 'processed' exista.
RAW_DATA_PATH = '/opt/airflow/data/raw/train.csv'
STEP1_NULL_FLAGS_PATH = '/opt/airflow/data/processed/train_step1_null_flags.csv' # <-- NUEVA
STEP2_ZEROS_PATH = '/opt/airflow/data/processed/train_step2_zeros.csv'
IMPUTATION_VALUES_PATH = '/opt/airflow/data/processed/imputation_values.json'
STEP3_MEDIANS_PATH = '/opt/airflow/data/processed/train_step3_medians.csv'
CLEAN_DATA_PATH = '/opt/airflow/data/processed/train_clean.csv'

with DAG(
    dag_id='churn_full_imputation_pipeline',
    start_date=datetime(2025, 8, 19),
    schedule=None,
    catchup=False,
    tags=['churn', 'processing', 'imputation'],
) as dag:
    
    create_null_flags = PythonOperator(
        task_id='create_null_flags',
        python_callable=create_null_flags_task,
        op_kwargs={
            'input_path': RAW_DATA_PATH,
            'output_path': STEP1_NULL_FLAGS_PATH,
            'config': CONFIG
        }
    )
    
    impute_zeros = PythonOperator(
        task_id='impute_with_zero',
        python_callable=impute_zeros_task,
        op_kwargs={
            'input_path': STEP1_NULL_FLAGS_PATH,
            'output_path': STEP2_ZEROS_PATH,
            'config': CONFIG
        }
    )

    calculate_imputation_values = PythonOperator(
        task_id='calculate_imputation_values',
        python_callable=calculate_imputation_values_task,
        op_kwargs={
            'input_path': STEP2_ZEROS_PATH,
            'output_path_json': IMPUTATION_VALUES_PATH,
            'config': CONFIG
        }
    )
    
    apply_imputation = PythonOperator(
        task_id='apply_median_imputation',
        python_callable=apply_imputation_task,
        op_kwargs={
            'input_path_data': STEP2_ZEROS_PATH,
            'output_path_data': STEP3_MEDIANS_PATH,
            'input_path_json': IMPUTATION_VALUES_PATH
        }
    )

    impute_categoricals = PythonOperator(
        task_id='impute_categoricals',
        python_callable=impute_categoricals_task,
        op_kwargs={
            'input_path': STEP3_MEDIANS_PATH,
            'output_path': CLEAN_DATA_PATH,
            'config': CONFIG
        }
    )

    # Flow definition
    create_null_flags >> impute_zeros >> calculate_imputation_values >> apply_imputation >> impute_categoricals