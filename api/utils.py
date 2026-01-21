import pandas as pd

def to_int_01(X):
    """
    Transforma características booleanas o tipo objeto (True/False) a enteros 1/0.
    Esta función es requerida para cargar el modelo churn_model_winner.joblib.
    """
    X = pd.DataFrame(X).copy()
    for col in X.columns:
        # Convertir explícitamente booleanos a int
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
        # Intentar convertir otros tipos a numérico. Los errores se convertirán en NaN.
        else:
            # Usar errors='coerce' para manejar no-numéricos y luego mapear True/False si son objetos
            temp_col = pd.to_numeric(X[col], errors="coerce")
            if temp_col.isna().all() and X[col].dtype == object: # Si todo es NaN y es objeto, intentar mapeo
                 X[col] = X[col].map({"True": 1, "False": 0, True: 1, False: 0})
            else:
                 X[col] = temp_col
    return X.values
