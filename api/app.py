import pandas as pd
import joblib
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
import os
import sys
from api.churn_logic import ChurnPredictor # Importamos la nueva lógica

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
async def predict_batch(file: UploadFile = File(...)):
    """
    Procesa un archivo CSV con múltiples clientes y devuelve las predicciones detalladas.
    """
    if not predictor:
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

        # Aplicar FeatureGenerator (Limpieza de datos: Título)
        feat_gen = FeatureGenerator()
        df = feat_gen.transform(df)

        # Realizar predicciones
        results = []
        df_dict = df.to_dict(orient='records')
        
        for record in df_dict:
            # Procesamos uno por uno para obtener el detalle de SHAP
            # (Esto puede ser lento para archivos grandes, pero garantiza el contrato completo)
            pred = predictor.predict(record)
            results.append(pred)

        # Anexar resultados al DataFrame ORIGINAL
        df['Prediction'] = [res['forecast'] for res in results]
        df['Probability'] = [res['probability'] for res in results]
        # Guardamos el JSON de importancias como string o estructura
        df['FeatureImportances'] = [res['feature_importances'] for res in results]

        # Convertir a lista de diccionarios (JSON)
        return df.to_dict(orient='records')

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error procesando lote: {e}")
        raise HTTPException(status_code=400, detail=f"Error al procesar el archivo CSV: {str(e)}")