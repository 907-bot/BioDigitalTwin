FROM python:3.11-slim
WORKDIR /app

# Install shared library
COPY shared/bio-digital-twin-core /shared/bio-digital-twin-core
RUN pip install -e /shared/bio-digital-twin-core

# Copy service code
COPY services/ontology-svc/app /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
