from datetime import datetime
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.models.variable import Variable

from scripts.tasks import predict_task

CONFIG = Variable.get("training_config", deserialize_json=True)

TEST_DATA_PATH = '/opt/airflow/data/processed/step4_test_bin.csv'

PREDICTIONS_PATH = '/opt/airflow/data/predictions/predictions.csv'

MODEL_PATH = '/opt/airflow/models/'

MODEL_TO_DEPLOY = Variable.get("model_to_deploy", default_var="RandomForestClassifier")

with DAG(
    dag_id='prediction_pipeline',
    start_date=datetime(2025, 8, 27),
    schedule=None,
    catchup=False,
    tags=['churn', 'prediction', 'ml'],
) as dag:
    # Tarea para hacer las predicciones
    make_predictions = PythonOperator(
        task_id='make_predictions',
        python_callable=predict_task,
        op_kwargs={
            'model_path': MODEL_PATH, # La ruta ahora es dinámica
            'data_path': TEST_DATA_PATH,
            'output_path': PREDICTIONS_PATH,
            'target_variable': CONFIG['general']['target']
        }
    )

    make_predictions