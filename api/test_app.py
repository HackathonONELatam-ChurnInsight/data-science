from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pytest
from api.app import app
import io
import numpy as np

client = TestClient(app)

# Datos de prueba válidos siguiendo el contrato PascalCase
valid_payload = {
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
    "Complain": 1
}

def test_read_root():
    """Prueba que la raíz redireccione a la documentación"""
    response = client.get("/", follow_redirects=False)
    # FastAPI RedirectResponse usa status 307 por defecto
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"

def test_predict_churn_success_churn():
    """Prueba una predicción exitosa simulando un caso de cancelación (1)"""
    # Mockear el predictor dentro de app.py
    # Usamos patch para reemplazar el objeto 'predictor' en el módulo 'app'
    with patch("api.app.predictor") as mock_predictor:
        # Configurar el comportamiento del mock
        mock_response = {
            "forecast": 1,
            "probability": 0.85,
            "feature_importances": [
                {"feature_name": "Age", "importance_value": 0.45, "ranking": 1}
            ]
        }
        mock_predictor.predict.return_value = mock_response

        response = client.post("/predict", json=valid_payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["forecast"] == 1
        assert data["probability"] == 0.85
        assert "feature_importances" in data

def test_predict_churn_success_no_churn():
    """Prueba una predicción exitosa simulando un caso de NO cancelación (0)"""
    with patch("api.app.predictor") as mock_predictor:
        # Caso negativo
        mock_response = {
            "forecast": 0,
            "probability": 0.05,
            "feature_importances": []
        }
        mock_predictor.predict.return_value = mock_response

        response = client.post("/predict", json=valid_payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["forecast"] == 0
        assert data["probability"] == 0.05

def test_predict_endpoint_validation_error():
    """Prueba que el validador Pydantic rechace datos incompletos"""
    invalid_payload = {
        "Geography": "Spain"
        # Faltan campos obligatorios
    }
    response = client.post("/predict", json=invalid_payload)
    
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data

def test_predict_endpoint_model_not_loaded():
    """Prueba el comportamiento cuando el modelo no está cargado (None)"""
    with patch("api.app.predictor", None):
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 500
        assert response.json()["detail"] == "Modelo no cargado."

def test_predict_batch_success():
    """
    Prueba el endpoint de carga masiva (batch).
    Verifica:
    1. Procesamiento correcto de CSV.
    2. Preservación de columnas extra (ej: ClienteID).
    3. Anexado de Prediction y Probability.
    """
    csv_content = """Geography,Gender,Age,CreditScore,Balance,EstimatedSalary,Tenure,NumOfProducts,SatisfactionScore,IsActiveMember,HasCrCard,Complain,ClienteID,Nombre
France,Female,42,619,0.0,101348.88,2,1,3,1,1,1,1001,Maria
Spain,Male,35,600,1000.0,50000.0,3,2,4,0,1,0,1002,Juan"""
    
    # Crear archivo en simulado memoria
    files = {
        'file': ('test_data.csv', io.BytesIO(csv_content.encode('utf-8')), 'text/csv')
    }

    with patch("api.app.predictor") as mock_predictor:
        # Configurar mocks. El loop llama a predictor.predict una vez por fila.
        # side_effect permite devolver valores distintos en cada llamada
        mock_predictor.predict.side_effect = [
            # Primera fila
            {
                "forecast": 1, 
                "probability": 0.85, 
                "feature_importances": [{"feature_name": "Age", "importance_value": 0.45, "ranking": 1}]
            },
            # Segunda fila
            {
                "forecast": 0, 
                "probability": 0.05,
                "feature_importances": []
            }
        ]

        response = client.post("/predict_batch", files=files)
        
        assert response.status_code == 200
        results = response.json()
        
        assert isinstance(results, list)
        assert len(results) == 2
        
        # Verificar primer registro (Prediction=1, Probability=0.85, ClienteID=1001 preservado)
        row1 = results[0]
        assert row1["Prediction"] == 1
        assert row1["Probability"] == 0.85
        assert row1["ClienteID"] == 1001
        assert row1["Nombre"] == "Maria"
        assert "FeatureImportances" in row1
        
        # Verificar segundo registro (Prediction=0, Probability=0.05)
        row2 = results[1]
        assert row2["Prediction"] == 0
        assert row2["Probability"] == 0.05

def test_predict_batch_missing_columns():
    """Prueba que le falten columnas requeridas al CSV"""
    csv_content = "Geography,Gender\nFrance,Female"
    files = {'file': ('test.csv', io.BytesIO(csv_content.encode()), 'text/csv')}
    
    # No necesitamos mockear el modelo porque fallará antes
    with patch("api.app.predictor", MagicMock()): 
        response = client.post("/predict_batch", files=files)
    
    assert response.status_code == 400
    assert "Faltan columnas requeridas" in response.json()["detail"]

def test_predict_batch_invalid_file_type():
    """Prueba subir un archivo que no sea .csv"""
    files = {'file': ('test.txt', io.BytesIO(b"dummy"), 'text/plain')}
    
    with patch("api.app.predictor", MagicMock()): 
        response = client.post("/predict_batch", files=files)
        
    assert response.status_code == 400
    assert "El archivo debe ser un CSV" in response.json()["detail"]
