import pandas as pd
import joblib
import io
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
import os
import sys
from api.churn_logic import ChurnPredictor # Importamos la nueva lógica
from typing import List, Optional

# Importar función utilitaria requerida por el modelo (Mantener por compatibilidad de joblib)
try:
    from api.utils import to_int_01
except ImportError:
    from utils import to_int_01

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
sys.modules['__main__'].to_int_01 = to_int_01

# --- 2. CARGAR MODELO ---
predictor = None
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, '..', 'model', 'churn_model_winner.joblib')
    
    # Inicializar la lógica compleja
    predictor = ChurnPredictor(model_path)
    print(f"Lógica de predicción cargada exitosamente desde: {model_path}")
except Exception as e:
    print(f"Error al cargar el modelo/predictor: {e}")
    predictor = None

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
    
class CustomerBatchRequest(BaseModel):
    modelVersion: Optional[str] = "v1"
    customers: List[CustomerRequest]

# --- 4. ENDPOINT ---
@app.post("/predict")
def predict_churn(data: CustomerRequest):
    if not predictor:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
    
    try:
        # A. Obtener datos limpios del request
        input_data = data.model_dump()
        
        # A.1 Aplicar FeatureGenerator (Limpieza de datos: Título)
        feat_gen = FeatureGenerator()
        df_wrapper = pd.DataFrame([input_data])
        df_clean = feat_gen.transform(df_wrapper)
        input_data_clean = df_clean.to_dict(orient='records')[0]

        # B. Usar el predictor avanzado con datos limpios
        prediction_result = predictor.predict(input_data_clean)
        
        return prediction_result
        
    except Exception as e:
        print(f"Error procesando solicitud: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# --- 5. ENDPOINT BATCH ---
@app.post("/predict_batch")
def predict_batch(data: CustomerBatchRequest):
    """
    Procesa un JSON con múltiples clientes y devuelve las predicciones detalladas.
    """
    if not predictor:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")

    try:
        # Convertir clientes a DataFrame
        df = pd.DataFrame([c.model_dump() for c in data.customers])

        # Aplicar FeatureGenerator
        feat_gen = FeatureGenerator()
        df = feat_gen.transform(df)

        # Predicciones
        results = []
        for record in df.to_dict(orient="records"):
            pred = predictor.predict(record)
            results.append(pred)

        # Enriquecer resultados
        df['forecast'] = [r['forecast'] for r in results]
        df['probability'] = [r['probability'] for r in results]
        df['feature_importances'] = [r['feature_importances'] for r in results]

        return {
            "modelVersion": data.modelVersion,
            "results": df.to_dict(orient="records")
        }

    except Exception as e:
        print(f"Error procesando lote: {e}")
        raise HTTPException(status_code=400, detail=str(e))
