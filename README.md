# Churn Insight API

API en FastAPI que expone un modelo de machine learning (joblib) para predecir la cancelación de clientes (churn) a partir de características bancarias. Incluye un transformador personalizado (`FeatureGenerator`) y un endpoint REST para predicción.

## Requisitos

- Python 3.10+ recomendado
- Dependencias en `requirements.txt`
- Archivo de modelo entrenado: `churn_model_winner.joblib` (ubicado en la raíz del proyecto)

## Instalación

1) Crear y activar un entorno virtual (opcional pero recomendado).
2) Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución local

Ejecuta el servidor con Uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- Documentación interactiva: http://127.0.0.1:8000/docs
- Redirección raíz (`/`) lleva a `/docs`.

## Contrato de entrada `/predict`

- Método: `POST`
- URL: `/predict`
- Cuerpo JSON (camelCase):

```json
{
  "geography": "France",
  "gender": "Male",
  "age": 42,
  "creditScore": 650,
  "balance": 50000.0,
  "estimatedSalary": 70000.0,
  "tenure": 4,
  "numOfProducts": 2,
  "satisfactionScore": 3,
  "isActiveMember": true,
  "hasCrCard": true,
  "complain": false
}
```

## Respuesta

- `forecast`: `"Va a cancelar"` o `"No va a cancelar"`
- `probability`: probabilidad de churn (0 a 1, redondeada a 2 decimales)

Ejemplo de respuesta:

```json
{
  "forecast": "No va a cancelar",
  "probability": 0.12
}
```

## Lógica principal

- `app.py`: carga el modelo `churn_model_winner.joblib`, define el transformador `FeatureGenerator`, esquema `CustomerRequest`, y el endpoint `/predict`.
- Mapeo de entrada: convierte booleans a enteros y normaliza campos (`Geography`, `Gender`) antes de predecir.

## Pruebas rápidas con `curl`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "geography": "France",
    "gender": "Female",
    "age": 35,
    "creditScore": 720,
    "balance": 12000,
    "estimatedSalary": 55000,
    "tenure": 6,
    "numOfProducts": 1,
    "satisfactionScore": 4,
    "isActiveMember": true,
    "hasCrCard": true,
    "complain": false
  }'
```

## Notas

- Asegúrate de que `churn_model_winner.joblib` esté en la misma carpeta que `app.py`.
- El transformador `FeatureGenerator` añade `HasBalance` y normaliza texto antes de la predicción.
- Usa `uvicorn` en producción detrás de un servidor (e.g., Nginx) si se despliega públicamente.
