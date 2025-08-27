from datetime import datetime
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.models.variable import Variable

# Importa las funciones específicas para las tareas
from scripts.tasks import (
    train_model_task,
    evaluate_model_task, # <- Nueva importación
)

# Carga la configuración directamente desde la Variable de Airflow
CONFIG = Variable.get("training_config", deserialize_json=True)

# Define las rutas de los ficheros
TRAIN_DATA_PATH = '/opt/airflow/data/processed/step4_train_bin.csv'
TEST_DATA_PATH = '/opt/airflow/data/processed/step4_test_bin.csv'
MODEL_STORAGE_PATH = '/opt/airflow/models'

with DAG(
    dag_id='training_pipeline',
    start_date=datetime(2025, 8, 27),
    schedule=None,
    catchup=False,
    tags=['churn', 'training', 'ml'],
) as dag:

    for model_config in CONFIG['models_to_train']:
        # Aseguramos que el task_id siempre sea único
        model_name = model_config.get('model_name', 'default_model')
        
        # Tarea 1: Entrenar y guardar el modelo
        train_model = PythonOperator(
            task_id=f'train_{model_name}',
            python_callable=train_model_task,
            op_kwargs={
                'input_path': TRAIN_DATA_PATH,
                'model_config': model_config, 
                'target_variable_name': CONFIG['general']['target'],
            }
        )
        
        # Tarea 2: Evaluar el modelo que se guardó
        evaluate_model = PythonOperator(
            task_id=f'evaluate_{model_name}',
            python_callable=evaluate_model_task,
            op_kwargs={
                'model_path': train_model.output, 
                'test_data_path': TEST_DATA_PATH,
                'target_variable': CONFIG['general']['target'],
            }
        )

        # Definir la dependencia
        train_model >> evaluate_model
