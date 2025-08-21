# Fichero: dags/churn_processing_dag.py

from datetime import datetime
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.models.variable import Variable

# Importa las funciones específicas para las tareas
from scripts.tasks import (
    remove_rows_with_any_null_task,
    clean_negatives_task,
    impute_zeros_task,
    impute_categoricals_task,
    calculate_imputation_values_task,
    apply_imputation_task,
    create_null_flags_task
)

# Carga la configuración desde la Variable de Airflow que creaste
CONFIG = Variable.get("churn_pipeline_config", deserialize_json=True)

# Define las rutas de los ficheros. Asegúrate de que la carpeta 'processed' exista.
RAW_DATA_PATH = '/opt/airflow/data/raw/train.csv'
STEP1_NULL_ROWS_REMOVED_PATH = '/opt/airflow/data/processed/train_step1_null_rows_removed.csv' # <-- NUEVA
STEP2_NEGATIVES_REMOVED_PATH = '/opt/airflow/data/processed/train_step2_negatives_removed.csv' # <-- NUEVA
STEP3_NULL_FLAGS_PATH = '/opt/airflow/data/processed/train_step3_null_flags.csv'
STEP4_ZEROS_PATH = '/opt/airflow/data/processed/train_step4_zeros.csv'
IMPUTATION_VALUES_PATH = '/opt/airflow/data/processed/imputation_values.json'
STEP5_MEDIANS_PATH = '/opt/airflow/data/processed/train_step5_medians.csv'
CLEAN_DATA_PATH = '/opt/airflow/data/processed/train_clean.csv'

with DAG(
    dag_id='churn_full_imputation_pipeline',
    start_date=datetime(2025, 8, 19),
    schedule=None,
    catchup=False,
    tags=['churn', 'processing', 'imputation'],
) as dag:
    remove_null_rows = PythonOperator(
        task_id='remove_null_rows',
        python_callable=remove_rows_with_any_null_task,
        op_kwargs={
            'input_path': RAW_DATA_PATH,
            'output_path': STEP1_NULL_ROWS_REMOVED_PATH,
            'config': CONFIG
        }
    )

    clean_negatives = PythonOperator(
        task_id='clean_negatives',
        python_callable=clean_negatives_task,
        op_kwargs={
            'input_path': STEP1_NULL_ROWS_REMOVED_PATH,
            'output_path': STEP2_NEGATIVES_REMOVED_PATH,
            'config': CONFIG
        }
    )
    create_null_flags = PythonOperator(
        task_id='create_null_flags',
        python_callable=create_null_flags_task,
        op_kwargs={
            'input_path': STEP2_NEGATIVES_REMOVED_PATH,
            'output_path': STEP3_NULL_FLAGS_PATH,
            'config': CONFIG
        }
    )
    
    impute_zeros = PythonOperator(
        task_id='impute_with_zero',
        python_callable=impute_zeros_task,
        op_kwargs={
            'input_path': STEP3_NULL_FLAGS_PATH,
            'output_path': STEP4_ZEROS_PATH,
            'config': CONFIG
        }
    )

    calculate_imputation_values = PythonOperator(
        task_id='calculate_imputation_values',
        python_callable=calculate_imputation_values_task,
        op_kwargs={
            'input_path': STEP4_ZEROS_PATH,
            'output_path_json': IMPUTATION_VALUES_PATH,
            'config': CONFIG
        }
    )
    
    apply_imputation = PythonOperator(
        task_id='apply_median_imputation',
        python_callable=apply_imputation_task,
        op_kwargs={
            'input_path_data': STEP4_ZEROS_PATH,
            'output_path_data': STEP5_MEDIANS_PATH,
            'input_path_json': IMPUTATION_VALUES_PATH
        }
    )

    impute_categoricals = PythonOperator(
        task_id='impute_categoricals',
        python_callable=impute_categoricals_task,
        op_kwargs={
            'input_path': STEP5_MEDIANS_PATH,
            'output_path': CLEAN_DATA_PATH,
            'config': CONFIG
        }
    )

    # Flow definition
    remove_null_rows >> clean_negatives >> create_null_flags >> impute_zeros >> calculate_imputation_values >> apply_imputation >> impute_categoricals