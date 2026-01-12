# 1. Imagen base recomendada en el README (Python 3.10)
FROM python:3.10-slim

# 2. Directorio de trabajo
WORKDIR /app

# 3. Instalar dependencias del sistema (gcc es vital para numpy/pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiar requirements e instalar (Aprovechando caché)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar TODO el código (app.py, modelos, carpetas, etc.)
COPY . .

# 6. Exponer puerto interno
EXPOSE 8000

# 7. Comando de arranque (Producción)
# Usa "app:app" porque tu archivo es app.py y la instancia FastAPI se llama app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
