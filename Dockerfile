
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario rootless
RUN addgroup --system appgroup && adduser --system --group appuser

# Instalar dependencias 
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar codigo y el modelo ONNX descargado
COPY challenge/ ./challenge/
COPY delay_model.onnx .

# utilizar usuario seguro
RUN chown -R appuser:appgroup /app
USER appuser

# puerto de Cloud Run
EXPOSE 8080

# Comando de arranque
CMD ["sh", "-c", "uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT:-8080}"]