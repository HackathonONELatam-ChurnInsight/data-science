# Churn Insight Project

Sistema backend de análisis de retención de clientes. Consta de una **API REST (FastAPI)** para realizar predicciones de churn en tiempo real y batch.

Incluye un modelo de ML (`joblib`) y un transformador personalizado (`FeatureGenerator`).

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

## Ejecución de la API (Backend)

Ejecuta el servidor FastAPI con Uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- Documentación interactiva: http://127.0.0.1:8000/docs
- Redirección raíz (`/`) lleva a `/docs`.



## Contrato de entrada `/predict`

- Método: `POST`
- Endpoint: `/predict`
- Cuerpo JSON (PascalCase):

```json
{
  "Geography": "France",
  "Gender": "Male",
  "Age": 42,
  "CreditScore": 650,
  "Balance": 50000.0,
  "EstimatedSalary": 70000.0,
  "Tenure": 4,
  "NumOfProducts": 2,
  "SatisfactionScore": 3,
  "IsActiveMember": 1,
  "HasCrCard": 1,
  "Complain": 0
}
```

## Respuesta

- `forecast`: `1` (Va a cancelar) o `0` (No va a cancelar)
- `probability`: probabilidad de churn (0 a 1, redondeada a 2 decimales)

Ejemplo de respuesta:

```json
{
  "forecast": 0,
  "probability": 0.12
}
```

## Contrato de entrada `/predict_batch`

- Método: `POST`
- Endoint: `/predict_batch`
- Tipo de contenido: `multipart/form-data`
- Parámetro: `file` (archivo `.csv`)

**Requisitos del CSV:**
Debe contener las columnas requeridas (orden irrelevante, case sensitive):
`Geography`, `Gender`, `Age`, `CreditScore`, `Balance`, `EstimatedSalary`, `Tenure`, `NumOfProducts`, `SatisfactionScore`, `IsActiveMember`, `HasCrCard`, `Complain`.

**Columnas adicionales:**
El endpoint acepta CSV con columnas extra (ej. `ID`, `Nombre`). Estas columnas son **ignoradas por el modelo pero preservadas en la respuesta**, lo que facilita identificar a los clientes procesados.

**Respuesta Batch:**
Retorna un array JSON con los datos originales más dos columnas nuevas:
- `Prediction`: `1` o `0`
- `Probability`: 0.00 - 1.00

## Manejo de Errores

La API utiliza códigos de estado HTTP estándar para comunicar el resultado de las operaciones:

**422 Unprocessable Entity (Error de Validación)**
- **Causa:** El JSON enviado no cumple con el esquema esperado (ej. `Age` enviado como texto en lugar de número, o falta un campo obligatorio).
- **Origen:** Generado automáticamente por la capa de validación de **FastAPI / Pydantic** antes de ejecutar la lógica del endpoint.

**400 Bad Request (Petición Incorrecta)**
- **Causa:** Viollación de reglas de negocio en la entrada.
  - En `/predict_batch`: El archivo subido no es `.csv` o faltan columnas requeridas en la cabecera del archivo.
  - En `/predict`: Error genérico capturado al procesar los datos.
- **Origen:** Validaciones explícitas (`if/raise`) dentro de los endpoints en **`app.py`**.

**500 Internal Server Error (Error Interno)**
- **Causa:** Fallo crítico del servidor, específicamente si el archivo del modelo `churn_model_winner.joblib` no se pudo cargar en memoria al arrancar.
- **Origen:** Verificación de seguridad (`if not model:`) al inicio de cada endpoint en **`app.py`**.

**200 OK (Éxito)**
- **Causa:** La solicitud fue procesada correctamente y se devolvió una predicción.
- **Origen:** Retorno exitoso de la función del endpoint.

## Lógica principal

- `app.py`: carga el modelo `churn_model_winner.joblib`, define el transformador `FeatureGenerator`, esquema `CustomerRequest`, y los endpoint `/predict` y `/predict_batch`.
- Mapeo de entrada: el modelo recibe los datos directamente en PascalCase tras la validacion de Pydantic.

## Pruebas Automatizadas

El proyecto incluye pruebas para la API y la lógica del Dashboard.

```bash
pytest
```

Esto ejecutará:
- `test_app.py`: Pruebas de endpoints de la API.
- `test_dashboard.py`: Pruebas de lógica de negocio del dashboard (cálculos de KPI, filtrado).

### Cobertura de tests (`test_app.py`)
1. **`test_read_root`**: Verifica redirección `/` -> `/docs`.
2. **`test_predict_churn_success_churn`**: Valida respuesta binaria `1` cuando el modelo predice cancelación.
3. **`test_predict_churn_success_no_churn`**: Valida respuesta binaria `0` cuando el modelo predice no cancelación.
4. **`test_predict_endpoint_validation_error`**: Asegura que payloads inválidos retornen 422.
5. **`test_predict_endpoint_model_not_loaded`**: Valida manejo de errores si el modelo no carga.
6. **`test_predict_batch_success`**: Verifica carga masiva CSV, preservación de columnas extra y anexado de predicciones.
7. **`test_predict_batch_missing_columns`**: Valida error 400 si faltan columnas en el CSV.
8. **`test_predict_batch_invalid_file_type`**: Valida rechazo de archivos que no sean `.csv`.

## Pruebas rápidas con `curl`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Geography": "France",
    "Gender": "Female",
    "Age": 35,
    "CreditScore": 720,
    "Balance": 12000,
    "EstimatedSalary": 55000,
    "Tenure": 6,
    "NumOfProducts": 1,
    "SatisfactionScore": 4,
    "IsActiveMember": 1,
    "HasCrCard": 1,
    "Complain": 0
  }'
```

## Notas

- Asegúrate de que `churn_model_winner.joblib` esté en la misma carpeta que `app.py`.
- El transformador `FeatureGenerator` añade `HasBalance` y normaliza texto antes de la predicción.
- Usa `uvicorn` en producción detrás de un servidor (e.g., Nginx) si se despliega públicamente.
