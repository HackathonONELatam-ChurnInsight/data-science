# [Nombre del Proyecto de Hackathon]

**Equipo:** [Nombre del Equipo]
- [Miembro 1]
- [Miembro 2]
- [Miembro 3]

---

## 📂 Estructura del Proyecto

A continuación se detalla la organización de archivos y el propósito de cada componente:

```text
/
├── api/                        # Código fuente de la API (FastAPI)
│   ├── app.py                  # Endpoints y configuración de la app
│   ├── churn_logic.py          # Lógica de predicción y explicabilidad (SHAP)
│   ├── utils.py                # Funciones auxiliares para serialización
│   └── test_app.py             # Tests automatizados (pytest)
├── model/                      # Modelos serializados y metadatos
│   ├── churn_model_winner.joblib  # Modelo productivo
│   └── metadata_modelo.joblib     # Umbrales y configuración extra
├── notebook/                   # Experimentos y entrenamiento
│   └── Reg_logística_Smote.ipynb  # Notebook de entrenamiento final
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación
```

---

## 🏆 Entregable Principal (Notebooks)

Para ver el análisis exploratorio, ingeniería de features y métricas del modelo detalladas, accede al notebook principal:

👉 [**Ir al Notebook de Entrenamiento**](notebook/) *(Enlace pendiente de notebook específico)*

---

## 📊 Resumen Técnico

- **Modelo Elegido**: Regresión Logística con Calibración de Probabilidades y SMOTE para balanceo de clases.
- **Explicabilidad**: Incorporamos valores SHAP en la API para explicar *por qué* se toma cada decisión a nivel de cliente.
- **Métricas Clave (Notebook)**:
    - *Consultar el notebook para los valores exactos, dado que se generan dinámicamente.*

---

## 🚀 API y Despliegue

La API no solo sirve la predicción del modelo `.joblib`, sino que integra una capa de lógica de negocio (`ChurnPredictor`) que:
1. Reconstruye un explicador **SHAP** en tiempo de ejecución.
2. Aplica limpieza automática de datos de texto (ej. "france" -> "France").
3. Carga un umbral de decisión optimizado desde `metadata_modelo.joblib`.

### Requisitos Técnicos

- Python 3.11 (Probado en 3.11.4)
- Dependencias clave: `fastapi`, `scikit-learn`, `shap`, `imbalanced-learn`, `joblib`.
- Consultar `requirements.txt` para la lista completa.

### Instalación y Ejecución

1. **Entorno Virtual**: Crear y activar (recomendado).
   ```bash
   # Crear entorno virtual
   python -m venv venv

   # Activar en Windows
   .\venv\Scripts\activate

   # Activar en macOS/Linux
   source venv/bin/activate
   ```
2. **Dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Levantar el Servidor**:
   Desde la raíz del proyecto, ejecuta:
   ```bash
   uvicorn api.app:app --reload
   ```

### Documentación Interactiva

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- La raíz (`/`) redirecciona automáticamente a la documentación.

### Endpoints Disponibles

#### 1. POST `/predict` (Predicción Individual)

Realiza una predicción de churn (abandono) para un solo cliente.

**Cuerpo de la Petición (JSON):**

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

**Respuesta Exitosa (JSON):**
Incluye predicción, probabilidad y **feature importances** (explicabilidad).

```json
{
  "forecast": 1,
  "probability": 0.85,
  "feature_importances": [
    {
      "feature_name": "Age",
      "feature_value": 45,
      "importance_value": 0.4564,
      "ranking": 1
    },
    {
      "feature_name": "IsActiveMember",
      "feature_value": 0,
      "importance_value": 0.4483,
      "ranking": 2
    }
  ]
}
```

#### 2. POST `/predict_batch` (Carga Masiva)

Sube un archivo `.csv` para obtener predicciones de múltiples clientes a la vez.

**Requisitos del CSV:**
- Debe contener encabezados compatibles (ej: `Geography`, `Age`, `CreditScore`, etc).
- Se preservan las columnas originales del CSV en la respuesta.

**Respuesta:**
- Retorna JSON con la lista de objetos, donde cada objeto incluye los datos originales más `Prediction`, `Probability`, y `FeatureImportances` (ranking de impacto).

- **Tipo de contenido**: `multipart/form-data`
- **Parámetro**: `file` (archivo `.csv`)

**Requisitos del CSV:**
- Debe contener las columnas requeridas (`Geography`, `Gender`, `Age`, etc.).
- Puede contener columnas extra (ej. `ID`, `Nombre`) que el modelo ignorará pero **se devolverán en la respuesta** para mantener la trazabilidad.

**Respuesta Batch (Lista JSON):**
Retorna un array de objetos, uno por fila del CSV, manteniendo el orden original.

```json
[
  {
    "ClienteID": 1001,
    "Nombre": "Maria",
    "Geography": "France",
    ... (resto de columnas originales),
    "Prediction": 1,
    "Probability": 0.85,
    "FeatureImportances": [
       {"feature_name": "Age", "importance_value": 0.45, ...},
       ...
    ]
  },
  ...
]
```

### Manejo de Errores

| Código | Significado | Causa Común |
| :--- | :--- | :--- |
| **422** | Unprocessable Entity | JSON inválido o tipos de datos incorrectos (Validación Pydantic). |
| **400** | Bad Request | Reglas de negocio (ej. archivo subido no es CSV, faltan columnas requeridas). |
| **500** | Internal Server Error | Error crítico del servidor (ej. modelo no cargado). |

### Pruebas

El proyecto incluye pruebas automatizadas con `pytest`. Ejecutar desde la raíz:

```bash
pytest
```
Esto correrá suite de pruebas ubicada en `api/test_app.py`.

#### Cobertura de tests (`test_app.py`)
1. **`test_read_root`**: Verifica redirección `/` -> `/docs`.
2. **`test_predict_churn_success_churn`**: Valida respuesta binaria `1` cuando el modelo predice cancelación.
3. **`test_predict_churn_success_no_churn`**: Valida respuesta binaria `0` cuando el modelo predice no cancelación.
4. **`test_predict_endpoint_validation_error`**: Asegura que payloads inválidos retornen 422.
5. **`test_predict_endpoint_model_not_loaded`**: Valida manejo de errores si el modelo no carga.
6. **`test_predict_batch_success`**: Verifica carga masiva CSV, preservación de columnas extra y anexado de predicciones.
7. **`test_predict_batch_missing_columns`**: Valida error 400 si faltan columnas en el CSV.
8. **`test_predict_batch_invalid_file_type`**: Valida rechazo de archivos que no sean `.csv`.

#### Pruebas rápidas con `curl`

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

