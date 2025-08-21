FROM apache/airflow:3.0.4
USER root
COPY requirements.txt /requirements.txt

RUN chown airflow:root /requirements.txt

USER airflow

RUN pip install --no-cache-dir -r /requirements.txt