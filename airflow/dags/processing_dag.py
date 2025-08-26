# Fichero: dags/churn_processing_dag.py

from datetime import datetime
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.models.variable import Variable

# Importa las funciones específicas para las tareas
from scripts.tasks import (
    remove_rows_with_any_null_task,
    clean_negatives_task,
    impute_categoricals_task,
    calculate_imputation_values_task,
    apply_imputation_task,
    create_null_flags_task,
    train_test_split_task,
    impute_all_nulls_task
)

# Carga la configuración desde la Variable de Airflow que creaste
CONFIG = Variable.get("churn_pipeline_config", deserialize_json=True)

# Define las rutas de los ficheros. Asegúrate de que la carpeta 'processed' exista.
RAW_DATA_PATH = '/opt/airflow/data/raw/dataset.csv'
CLEAN_DATA_PATH = '/opt/airflow/data/processed'

DATA_STEP_1 = CLEAN_DATA_PATH + '/data_step1_nulls_removed.csv'
DATA_STEP_2 = CLEAN_DATA_PATH + '/data_step2_negatives_cleaned.csv'
DATA_STEP_3 = CLEAN_DATA_PATH + '/data_step3_null_flags.csv'
IMPUTATION_VALUES_JSON = CLEAN_DATA_PATH + '/imputation_values.json'
DATA_STEP_4 = CLEAN_DATA_PATH + '/data_step4_imputed.csv'
DATA_STEP_5 = CLEAN_DATA_PATH + '/data_step5_final_clean.csv'
TRAIN_OUTPUT_PATH = CLEAN_DATA_PATH + '/train_set.csv'
TEST_OUTPUT_PATH = CLEAN_DATA_PATH + '/test_set.csv'

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
            'output_path': DATA_STEP_1,
            'config': CONFIG
        }
    )

    clean_negatives = PythonOperator(
        task_id='clean_negatives',
        python_callable=clean_negatives_task,
        op_kwargs={
            'input_path': DATA_STEP_1,
            'output_path': DATA_STEP_2,
            'config': CONFIG
        }
    )
    create_null_flags = PythonOperator(
        task_id='create_null_flags',
        python_callable=create_null_flags_task,
        op_kwargs={
            'input_path': DATA_STEP_2,
            'output_path': DATA_STEP_3,
            'config': CONFIG
        }
    )
    
    calculate_imputation_values = PythonOperator(
        task_id='calculate_imputation_values',
        python_callable=calculate_imputation_values_task,
        op_kwargs={
            'input_path': DATA_STEP_3,
            'output_path_json': IMPUTATION_VALUES_JSON,
            'config': CONFIG
        }
    )

    impute_all_nulls = PythonOperator(
        task_id='impute_all_nulls',
        python_callable=impute_all_nulls_task,
        op_kwargs={
            'input_path': DATA_STEP_3,
            'output_path': DATA_STEP_4,
            'imputation_values_path': IMPUTATION_VALUES_JSON,
            'config': CONFIG
        }
    )

    train_test_split = PythonOperator(
        task_id='train_test_split',
        python_callable=train_test_split_task,
        op_kwargs={
            'input_path': DATA_STEP_4,
            'train_output_path': TRAIN_OUTPUT_PATH,
            'test_output_path': TEST_OUTPUT_PATH,
            'config': CONFIG
        }
    )

    # Flow definition
    remove_null_rows >> clean_negatives >> create_null_flags >> calculate_imputation_values >> impute_all_nulls >> train_test_split