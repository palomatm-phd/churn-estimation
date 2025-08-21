# Empezamos desde la imagen oficial de Airflow
FROM apache/airflow:3.0.4

# Establecemos el directorio de trabajo por defecto
WORKDIR /opt/airflow

# Como root, copiamos todos los ficheros del proyecto y ajustamos permisos
USER root
COPY . .
RUN chown -R airflow:root /opt/airflow

# Cambiamos al usuario airflow para las instalaciones
USER airflow

# 1. Instalamos las dependencias externas desde requirements.txt
# (Asegúrate de que 'pandas', 'scipy', etc. están en este fichero)
RUN pip install --no-cache-dir -r requirements.txt

# 2. El paso mágico que faltaba: instalamos nuestra librería local en modo editable
RUN pip install --no-cache-dir -e .