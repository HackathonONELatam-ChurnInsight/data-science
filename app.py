import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
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
        if 'Balance' in X_out.columns:
            X_out['HasBalance'] = (X_out['Balance'] > 0).astype(int)
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
    
    # Combina ese directorio con el nombre de archivo
    model_path = os.path.join(BASE_DIR, 'churn_model_winner.joblib')
    
    model = joblib.load(model_path)
    print(f"Modelo cargado exitosamente desde: {model_path}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

# --- 3. CONTRATO DE ENTRADA ---
class CustomerRequest(BaseModel):
    geography: str
    gender: str
    age: int
    creditScore: int
    balance: float
    estimatedSalary: float
    tenure: int
    numOfProducts: int
    satisfactionScore: int
    isActiveMember: bool
    hasCrCard: bool
    complain: bool

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
        req = data.model_dump()
        
        # B. Mapeo: JSON (camelCase) -> Modelo ML (PascalCase)
        # Convertimos automáticamente los booleanos (true/false) a enteros (1/0)
        input_data = {
            "Geography": req["geography"],
            "Gender": req["gender"],
            "Age": req["age"],
            "CreditScore": req["creditScore"],
            "Balance": req["balance"],
            "EstimatedSalary": req["estimatedSalary"],
            "Tenure": req["tenure"],
            "NumOfProducts": req["numOfProducts"],
            "SatisfactionScore": req["satisfactionScore"],
            "IsActiveMember": int(req["isActiveMember"]), # true -> 1, false -> 0
            "HasCrCard": int(req["hasCrCard"]),           # true -> 1, false -> 0
            "Complain": int(req["complain"])              # true -> 1, false -> 0
        }
        
        # Crear DataFrame
        df = pd.DataFrame([input_data])
        
        # C. Predicción
        pred_class = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][1])
        
        # D. Respuesta (Manteniendo el formato que espera el Backend Java)
        resultado_texto = "Va a cancelar" if pred_class == 1 else "No va a cancelar"
        
        return {
            "forecast": resultado_texto,
            "probability": round(proba, 2)
        }
        
    except Exception as e:
        print(f"Error procesando solicitud: {e}")
        raise HTTPException(status_code=400, detail=str(e))