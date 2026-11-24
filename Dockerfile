FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY data/ ./data/
COPY services/ ./services/
COPY frontend/ ./frontend/

EXPOSE 8000
CMD ["uvicorn", "services.graphql-gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]