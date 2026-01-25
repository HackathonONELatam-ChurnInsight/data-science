# [Nombre del Proyecto de Hackathon]

**Equipo:** HoldOn Data Labs
- Claudia Ximena Delgado Gutiérrez 
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:cid2024sec@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/claudiax-delgado)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ClaudiaXDG)
- Felipe Octavio Rebolledo
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:felipe.rebolledo.robert@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/felipe-rebolledo-robert/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/FelipeOctavio87)
- Nicolas Ruiz
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:nruizb14@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nicolas-ruiz-953323302)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Noirwolf04)
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

Para ver el análisis exploratorio, ingeniería de features y métricas del modelo detalladas, accede a los notebooks:

👉 [**MVP Churn Insight (Unificado)**](notebook/MVP_ChurnInsight_Notebook_Unificado.ipynb)

---

## 📊 Resumen Técnico

### Modelo y Metodología

- **Algoritmo Base**: Regresión Logística con regularización L2.
- **Manejo del Desbalance**: SMOTE (Synthetic Minority Oversampling Technique) para generar datos sintéticos en la clase minoritaria.
- **Calibración**: Modelo calibrado con método Sigmoid (`CalibratedClassifierCV`) para asegurar probabilidades confiables.
- **Umbral de Decisión Optimizado**: Selección de umbral basada en maximización de F1-Score mediante Precision-Recall curve (umbral calibrado: ~0.22).

### Ingeniería de Características

**Variables Numéricas (5)**:
- `CreditScore`, `Age`, `Balance`, `EstimatedSalary`, `Tenure`
- Tratamiento: Imputación simple + escalado StandardScaler

**Variables Categóricas (2)**:
- `Geography` (Francia, España, Alemania)
- `Gender` (Masculino, Femenino)
- Tratamiento: Imputación + One-Hot Encoding con K-1 dummies para evitar colinealidad

**Variables Binarias (4)**:
- `HasCrCard`, `IsActiveMember`, `Complain`, `NumOfProducts` (convertida a binaria)
- Tratamiento: Conversión a tipo int + imputación

### Análisis de Multicolinealidad

- Cálculo de **Factor de Inflación de Varianza (VIF)** sobre matriz de entrenamiento post-preprocesamiento.
- Resultado: Valores de VIF cercanos a 1, indicando **ausencia de multicolinealidad relevante**.

### Búsqueda de Hiperparámetros

- **Método**: GridSearchCV con validación cruzada estratificada (5-fold, StratifiedKFold).
- **Parámetro sintonizado**: `C` (inversa de la fuerza de regularización) en escala logarítmica.
- **Rango explorado**: `[0.001, 0.01, 0.1, 1, 10, 100, 1000]`.
- **Métrica de Optimización**: ROC-AUC.

### Explicabilidad (SHAP)

- **Explainer**: SHAP Linear Explainer basado en el modelo de Regresión Logística calibrado.
- **Agregación**: SHAP values se agregan por feature original, unificando los dummies one-hot encoded bajo su categoría madre.
- **Aplicación**: Ranking de importancia de features para cada predicción individual en la API.

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

Sube un archivo `.json` para obtener predicciones de múltiples clientes a la vez.

**Tipo de contenido**: `multipart/form-data`
**Parámetro**: `file` (archivo `.json`)

**Requisitos del JSON:**
- Estructura requerida: debe tener `modelVersion` (string) y `customers` (array).
- `customers` debe ser una lista de objetos con las columnas requeridas.
- Cada objeto debe contener: `Geography`, `Gender`, `Age`, `CreditScore`, `Balance`, `EstimatedSalary`, `Tenure`, `NumOfProducts`, `SatisfactionScore`, `IsActiveMember`, `HasCrCard`, `Complain`.
- Puede contener columnas extra (ej. `ClienteID`, `Nombre`) que el modelo ignorará pero **se devolverán en la respuesta** para mantener la trazabilidad.

**Formato de entrada JSON:**

```json
{
  "modelVersion": "v1",
  "customers": [
    {
      "Geography": "France",
      "Gender": "Female",
      "Age": 42,
      "CreditScore": 619,
      "Balance": 0.0,
      "EstimatedSalary": 101348.88,
      "Tenure": 2,
      "NumOfProducts": 1,
      "SatisfactionScore": 3,
      "IsActiveMember": 1,
      "HasCrCard": 1,
      "Complain": 1,
      "ClienteID": 1001,
      "Nombre": "Maria"
    },
    {
      "Geography": "Spain",
      "Gender": "Male",
      "Age": 35,
      "CreditScore": 600,
      "Balance": 1000.0,
      "EstimatedSalary": 50000.0,
      "Tenure": 3,
      "NumOfProducts": 2,
      "SatisfactionScore": 4,
      "IsActiveMember": 0,
      "HasCrCard": 1,
      "Complain": 0,
      "ClienteID": 1002,
      "Nombre": "Juan"
    }
  ]
}
```

**Respuesta Batch (Objeto con propiedad "results"):**
Retorna un objeto JSON con una propiedad `results` que contiene un array de objetos, uno por registro del JSON de entrada.

```json
{
  "results": [
    {
      "ClienteID": 1001,
      "Nombre": "Maria",
      "Geography": "France",
      "Gender": "Female",
      "Age": 42,
      "CreditScore": 619,
      "Balance": 0.0,
      "EstimatedSalary": 101348.88,
      "Tenure": 2,
      "NumOfProducts": 1,
      "SatisfactionScore": 3,
      "IsActiveMember": 1,
      "HasCrCard": 1,
      "Complain": 1,
      "forecast": 1,
      "probability": 0.85,
      "feature_importances": [
         {"feature_name": "Age", "feature_value": 42, "importance_value": 0.45, "ranking": 1},
         {"feature_name": "IsActiveMember", "feature_value": 1, "importance_value": 0.42, "ranking": 2}
      ]
    },
    {
      "ClienteID": 1002,
      "Nombre": "Juan",
      "Geography": "Spain",
      ... (resto de campos),
      "forecast": 0,
      "probability": 0.15,
      "feature_importances": []
    }
  ]
}
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
6. **`test_predict_batch_success`**: Verifica carga masiva JSON, preservación de columnas extra y anexado de predicciones en estructura "results".
7. **`test_predict_batch_missing_columns`**: Valida error 400 si faltan columnas en el CSV.
8. **`test_predict_batch_invalid_file_type`**: Valida rechazo de archivos que no sean `.json`.

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

