import pandas as pd
import joblib
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
import os
import sys

# --- 1. CLASE TRANSFORMADORA ---
class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in ['Geography', 'Gender']:
            if col in X_out.columns:
                X_out[col] = X_out[col].astype(str).str.title()
        return X_out
# Asignamos la clase al módulo __main__ para que joblib la encuentre
sys.modules['__main__'].FeatureGenerator = FeatureGenerator

# --- 2. CARGAR MODELO ---
try:
    # Obtiene la ruta absoluta del directorio donde está este archivo (app.py)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Combina ese directorio con el nombre de archivo (subiendo un nivel a 'model')
    model_path = os.path.join(BASE_DIR, '..', 'model', 'churn_model_winner.joblib')
    
    model = joblib.load(model_path)
    print(f"Modelo cargado exitosamente desde: {model_path}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

# --- 3. CONTRATO DE ENTRADA ---
class CustomerRequest(BaseModel):
    Geography: str
    Gender: str
    Age: int
    CreditScore: int
    Balance: float
    EstimatedSalary: float
    Tenure: int
    NumOfProducts: int
    SatisfactionScore: int
    IsActiveMember: int
    HasCrCard: int
    Complain: int

app = FastAPI(title="Churn Insight API", version="1.0")

# Al entrar a "http://127.0.0.1:8000/", redirecciona a los docs
@app.get("/")
def main():
    return RedirectResponse(url="/docs")

# --- 4. ENDPOINT ---
@app.post("/predict")
def predict_churn(data: CustomerRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
    
    try:
        # A. Obtener datos limpios del request
        # Al definir el modelo con PascalCase, model_dump() generará las claves correctas.
        # Se asume que el types ya vienen correctos (ints para flags).
        input_data = data.model_dump()
        
        # Crear DataFrame
        df = pd.DataFrame([input_data])
        
        # C. Predicción
        # El nuevo modelo devuelve directamente un diccionario con la estructura completa:
        # { "forecast": int, "probability": float, "feature_importances": [...] }
        # predict devuelve una lista (uno por fila), tomamos el primero.
        prediction_result = model.predict(df)[0]
        
        return prediction_result
        
    except Exception as e:
        print(f"Error procesando solicitud: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# --- 5. ENDPOINT BATCH ---
@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Procesa un archivo CSV con múltiples clientes y devuelve las predicciones.
    El CSV debe contener las columnas: 'Geography', 'Gender', 'Age', 'CreditScore', 
    'Balance', 'EstimatedSalary', 'Tenure', 'NumOfProducts', 'SatisfactionScore', 
    'IsActiveMember', 'HasCrCard', 'Complain'.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV.")

    try:
        # Leer el contenido del archivo
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Columnas requeridas
        required_columns = {'Geography', 'Gender', 'Age', 'CreditScore', 'Balance', 
                            'EstimatedSalary', 'Tenure', 'NumOfProducts', 'SatisfactionScore', 
                            'IsActiveMember', 'HasCrCard', 'Complain'}

        # Validar columnas faltantes
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise HTTPException(status_code=400, detail=f"Faltan columnas requeridas en el CSV: {missing}")

        # Realizar predicciones
        # Nota: Asumimos que los datos vienen en el formato correcto (PascalCase, tipos compatibles)
        # El pipeline del modelo se encarga de las transformaciones necesarias (FeatureGenerator)

        # Filtrar solo las columnas que el modelo conoce para evitar errores si hay columnas extra (ID, Nombres, etc)
        # Se mantiene el orden original de las columnas en el CSV
        columns_for_model = [col for col in df.columns if col in required_columns]
        df_clean = df[columns_for_model]
        
        # El modelo devuelve una lista de diccionarios
        results = model.predict(df_clean)

        # Anexar resultados al DataFrame ORIGINAL (para devolver también las columnas extra)
        # Extraemos 'forecast' y 'probability' de cada diccionario
        df['Prediction'] = [res['forecast'] for res in results]
        df['Probability'] = [res['probability'] for res in results]
        df['FeatureImportances'] = [res.get('feature_importances') for res in results]

        # Convertir a lista de diccionarios (JSON)
        return df.to_dict(orient='records')

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error procesando lote: {e}")
        raise HTTPException(status_code=400, detail=f"Error al procesar el archivo CSV: {str(e)}")