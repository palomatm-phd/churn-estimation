# Usa una imagen base de Python
FROM python:3.9-slim

# Instala MLflow
RUN pip install --no-cache-dir mlflow

# Crea un directorio para los artefactos y establece permisos
RUN mkdir /mlflow_artifacts
RUN chmod -R 777 /mlflow_artifacts

# Opcional: Instala bibliotecas adicionales si las necesitas en el servidor de MLflow
RUN pip install psycopg2-binary