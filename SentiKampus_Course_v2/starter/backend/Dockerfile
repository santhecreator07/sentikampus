FROM python:3.10-slim

# Hindari file .pyc dan output yang di-buffer (penting untuk logging container).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/sentiment.pkl \
    METADATA_PATH=/app/models/metadata.json

WORKDIR /app

# Install dependensi terlebih dahulu (layer ini di-cache jika requirements tidak berubah).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin source code, data, scripts, dan direktori model.
COPY app ./app
COPY data ./data
COPY scripts ./scripts
COPY models ./models

# Latih model di dalam container agar artifact tersedia sebelum startup.
RUN python scripts/train_model.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
