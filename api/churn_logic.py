import pandas as pd
import numpy as np
import shap
import joblib
import sys
import os

# --- Definición constantes del negocio ---
NUM_FEATURES = ["Age", "CreditScore", "Balance", "NumOfProducts"]
CAT_FEATURES = ["Geography", "Gender"]
BIN_FEATURES = ["IsActiveMember"]

class ChurnPredictor:
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        self.pipeline = self.model # Alias
        
        # Cargar metadata si existe
        self.threshold = 0.5 # Default
        metadata_path = model_path.replace('churn_model_winner.joblib', 'metadata_modelo.joblib')
        self._load_metadata(metadata_path)

        self.explainer = None
        self.feature_names_out = None
        self._init_shap()

    def _load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo no encontrado en: {path}")
        return joblib.load(path)

    def _load_metadata(self, path):
        """Carga el umbral óptimo desde el archivo de metadatos si existe."""
        if os.path.exists(path):
            try:
                metadata = joblib.load(path)
                # El notebook guarda un dict: {'umbral_optimo': float}
                if isinstance(metadata, dict) and 'umbral_optimo' in metadata:
                    self.threshold = float(metadata['umbral_optimo'])
                    print(f"Metadatos cargados. Umbral calibrado: {self.threshold:.4f}")
            except Exception as e:
                print(f"Advertencia: Error al cargar metadata en {path}. Usando default 0.5. Error: {e}")
        else:
            print(f"Advertencia: Archivo de metadata {path} no encontrado. Usando umbral por defecto 0.5")

    def _init_shap(self):
        """
        Inicializa el explainer de SHAP.
        Al no tener X_train en producción, generamos un 'background' sintético
        basado en la media (ceros, ya que usamos StandardScaler).
        """
        try:
            # 1. Obtener el paso de preprocesamiento y el modelo final
            # Nota: Si el pipeline tiene 'calibrated_model', necesitamos acceder al estimador base
            # Estructura probable: CalibratedClassifierCV -> base_estimator (Pipeline) -> steps
            # O Pipeline -> steps
            
            # Intento de acceder al pipeline interno si es un CalibratedClassifierCV
            if hasattr(self.model, 'estimator'): 
                # Caso: CalibratedClassifierCV con cv='prefit' envuelve al pipeline
                internal_pipeline = self.model.estimator
            else:
                # Caso: Es el pipeline directo
                internal_pipeline = self.model

            self.preprocessor = internal_pipeline.named_steps['preprocess'] 
            # El paso final suele llamarse 'model' o 'classifier', buscamos el último
            self.classifier = internal_pipeline.named_steps['model']

            # 2. Obtener nombres de features de salida
            # Creamos un dummy input para obtener la estructura
            dummy_input = pd.DataFrame([{
                "Age": 30, "CreditScore": 600, "Balance": 0.0, "NumOfProducts": 1,
                "Geography": "France", "Gender": "Female", "IsActiveMember": 1,
                "EstimatedSalary": 50000.0, "Tenure": 5, "HasCrCard": 1, 
                "Complain": 0, "SatisfactionScore": 3
            }])
            
            # Aseguramos que solo usamos las columnas esperadas
            cols = NUM_FEATURES + CAT_FEATURES + BIN_FEATURES
            dummy_transformed = self.preprocessor.transform(dummy_input[cols])
            
            if hasattr(dummy_transformed, 'toarray'):
                dummy_transformed = dummy_transformed.toarray()
            
            self.input_shape = dummy_transformed.shape[1]
            try:
                self.feature_names_out = self.preprocessor.get_feature_names_out()
            except:
                # Fallback si no soporta get_feature_names_out
                self.feature_names_out = [f"feat_{i}" for i in range(self.input_shape)]

            # 3. Crear background data sintético (Ceros = Media en StandardScaler)
            # Usamos 10 muestras de ceros
            background_data = np.zeros((10, self.input_shape))
            background_df = pd.DataFrame(background_data, columns=self.feature_names_out)

            # 4. Inicializar Explainer
            self.explainer = shap.LinearExplainer(self.classifier, background_df)
            print("SHAP Explainer inicializado correctamente.")

        except Exception as e:
            print(f"Advertencia: No se pudo inicializar SHAP. Detalles: {e}")
            self.explainer = None

    def predict(self, input_data: dict):
        """
        Realiza la predicción y devuelve el contrato complejo con SHAP values.
        """
        # 1. Preparar DataFrame
        df = pd.DataFrame([input_data])
        
        # 2. Predicción (Probabilidad y Clase)
        # CalibratedClassifierCV usa predict_proba
        proba_all = self.model.predict_proba(df)[0] # [prob_0, prob_1]
        probability = float(proba_all[1])
        
        # Usamos el umbral cargado de metadata
        forecast = 1 if probability >= self.threshold else 0

        # 3. SHAP Feature Importance
        feature_importances = []
        
        if self.explainer:
            try:
                # Transformar datos
                input_cols = df[NUM_FEATURES + CAT_FEATURES + BIN_FEATURES]
                X_trans = self.preprocessor.transform(input_cols)
                if hasattr(X_trans, 'toarray'):
                    X_trans = X_trans.toarray()
                
                # Calcular SHAP
                shap_values_raw = self.explainer.shap_values(X_trans)[0] # [0] para la primera muestra
                # Nota: shap_values para regresión logística binaria a veces retorna lista, a veces array.
                # LinearExplainer suele retornar array para la clase positiva en binario o lista. 
                # Verificaremos forma.
                if isinstance(shap_values_raw, list):
                     shap_values_raw = shap_values_raw[1] # Clase 1? Depende implementación.
                
                # Mapear a features originales (Agregación)
                aggregated_importances = {}
                feature_original_values = {}
                
                for i, feat_name in enumerate(self.feature_names_out):
                    val = shap_values_raw[i] if len(shap_values_raw.shape)==1 else shap_values_raw[0][i]
                    
                    original_feat = None
                    # Lógica de mapeo del notebook
                    if feat_name in NUM_FEATURES or feat_name in BIN_FEATURES:
                        original_feat = feat_name
                    else:
                        for cat in CAT_FEATURES:
                            if feat_name.startswith(f"{cat}_"):
                                original_feat = cat
                                break
                    
                    if original_feat:
                        aggregated_importances[original_feat] = aggregated_importances.get(original_feat, 0.0) + val
                        if original_feat not in feature_original_values:
                            feature_original_values[original_feat] = input_data.get(original_feat)

                # Construir lista
                for fname, importance in aggregated_importances.items():
                    feature_importances.append({
                        "feature_name": fname,
                        "feature_value": feature_original_values.get(fname),
                        "importance_value": float(importance),
                        "ranking": 0 # Se calcula despues
                    })

            except Exception as e:
                print(f"Error calculando SHAP: {e}")

        # 4. Agregar variables no usadas (Feature Importances 0)
        used_feats = set([f["feature_name"] for f in feature_importances])
        all_input_keys = input_data.keys()
        
        # Features del modelo que no salieron en SHAP (raro pero posible)
        for f in NUM_FEATURES + CAT_FEATURES + BIN_FEATURES:
            if f not in used_feats:
                 feature_importances.append({
                    "feature_name": f,
                    "feature_value": input_data.get(f),
                    "importance_value": 0.0,
                    "ranking": 0
                })
        
        # Features extra (Payload extra)
        for k in all_input_keys:
             if k not in (NUM_FEATURES + CAT_FEATURES + BIN_FEATURES) and k not in used_feats:
                 feature_importances.append({
                    "feature_name": k,
                    "feature_value": input_data.get(k),
                    "importance_value": 0.0,
                    "note": "Variable no utilizada en el entrenamiento",
                    "ranking": 0
                })

        # 5. Ordenar y Ranking
        # Ordenar: Primero las usadas (note is None), luego por valor absoluto desc
        feature_importances.sort(key=lambda x: (
            x.get("note") is not None, # False (0) antes que True (1)
            -abs(x["importance_value"])
        ))

        for idx, item in enumerate(feature_importances):
            item["ranking"] = idx + 1
            item["importance_value"] = round(item["importance_value"], 4)

        return {
            "forecast": forecast,
            "probability": round(probability, 4),
            "feature_importances": feature_importances
        }
