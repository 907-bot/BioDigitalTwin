FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libpq-dev curl git && rm -rf /var/lib/apt/lists/*

COPY shared/bio-digital-twin-core /shared/bio-digital-twin-core
RUN pip install -e /shared/bio-digital-twin-core

COPY backend/app /app/app
COPY backend/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY services/patient-svc/app/main.py /app/main.py

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
