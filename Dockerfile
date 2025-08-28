FROM apache/airflow:3.0.4

WORKDIR /opt/airflow

USER root
COPY . .
RUN chown -R airflow:root /opt/airflow

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgfortran5 \
    libquadmath0 \
    libatlas3-base

USER airflow

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir -e .