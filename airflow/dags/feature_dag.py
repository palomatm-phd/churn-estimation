from datetime import datetime
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.models.variable import Variable

# Importa tus funciones de feature engineering
from churn_library.feature_helpers import (
    create_new_features_task,
    read_clean_data_task,
    one_hot_encode_and_align_task,
    binarize_and_align_quantiles_task
)

# Carga la configuración
CONFIG = Variable.get("feature_config", deserialize_json=True)
TEST_SET_PATH = '/opt/airflow/data/processed/test_set.csv'
TRAIN_SET_PATH = '/opt/airflow/data/processed/test_set.csv'
STEP1_TRAIN_REMOVED_COLS = '/opt/airflow/data/processed/step1_TRAIN_removed_cols.csv'
STEP1_TEST_REMOVED_COLS = '/opt/airflow/data/processed/step1_test_removed_cols.csv'
STEP2_TRAIN_FE = '/opt/airflow/data/processed/step2_train_fe.csv'
STEP2_TEST_FE = '/opt/airflow/data/processed/step2_test_fe.csv'
STEP3_TRAIN_ENCODE = '/opt/airflow/data/processed/step3_train_encode.csv'
STEP3_TEST_ENCODE = '/opt/airflow/data/processed/step3_test_encode.csv'
STEP4_TRAIN_BIN = '/opt/airflow/data/processed/step4_train_bin.csv'
STEP4_TEST_BIN = '/opt/airflow/data/processed/step4_test_bin.csv'


with DAG(
    dag_id='feature_engineering_pipeline',
    start_date=datetime(2025, 8, 26),
    schedule=None,
    catchup=False,
    tags=['churn', 'feature_engineering'],
) as dag:
    read_clean_data = PythonOperator(
        task_id='read_clean_data',
        python_callable=read_clean_data_task,
        op_kwargs={
            'input_path_train': TRAIN_SET_PATH,
            'input_path_test': TEST_SET_PATH,
            'output_path_train': STEP1_TRAIN_REMOVED_COLS,
            'output_path_test': STEP1_TEST_REMOVED_COLS,
            'config': CONFIG 
        }
    )

    create_new_features = PythonOperator(
        task_id='create_new_features',
        python_callable=create_new_features_task,
        op_kwargs={
            'input_path_train': STEP1_TRAIN_REMOVED_COLS,
            'input_path_test': STEP1_TEST_REMOVED_COLS,
            'output_path_train': STEP2_TRAIN_FE,
            'output_path_test': STEP2_TEST_FE,
        }
    )

    one_hot_encode = PythonOperator(
        task_id='one_hot_encode_and_align',
        python_callable=one_hot_encode_and_align_task,
        op_kwargs={
            'input_path_train': STEP2_TRAIN_FE,
            'input_path_test': STEP2_TEST_FE,
            'output_path_train': STEP3_TRAIN_ENCODE,
            'output_path_test': STEP3_TEST_ENCODE,
        }
    )

    binarize_task = PythonOperator(
        task_id='binarize_and_align_quantiles',
        python_callable=binarize_and_align_quantiles_task,
        op_kwargs={
            'input_path_train': STEP3_TRAIN_ENCODE,
            'input_path_test': STEP3_TEST_ENCODE,
            'output_path_train': STEP4_TRAIN_BIN,
            'output_path_test': STEP4_TEST_BIN,
            'columns_to_bin': CONFIG['binning']['columns'],
            'q': CONFIG['binning']['quantiles']
        }
    )

    read_clean_data >> create_new_features >> one_hot_encode >> binarize_task