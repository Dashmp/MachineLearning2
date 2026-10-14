from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Cargar datos y entrenar el modelo MP Neurona
cancer_mama = load_breast_cancer()
X = cancer_mama.data
Y = cancer_mama.target

df = pd.DataFrame(X, columns=cancer_mama.feature_names)
X_train, X_test, y_train, y_test = train_test_split(df, Y, stratify=Y, random_state=42)

# Binarizar entrenamiento
X_train_bin = X_train.apply(pd.cut, bins=2, labels=[1, 0])

class MpNeurona:
    def __init__(self):
        self.threshold = None
    
    def model(self, x):
        z = sum(x)
        return (z >= self.threshold)
    
    def predict(self, X):
        Y_preds = []
        for x in X:
            result = self.model(x)
            Y_preds.append(result)
        return np.array(Y_preds)
    
    def fit(self, X, Y):
        accuracy = {}
        for th in range(X.shape[1] + 1):
            self.threshold = th
            Y_pred = self.predict(X)
            accuracy[th] = accuracy_score(Y_pred, Y)
        self.threshold = max(accuracy, key=accuracy.get)

# Instanciar y entrenar
np_neurona = MpNeurona()
np_neurona.fit(X_train_bin.to_numpy(), y_train)

# 2. Configurar FastAPI
app = FastAPI(title="API Neurona MP - Cáncer de Mama", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    features: list[float]

@app.get("/")
def read_root():
    return {
        "mensaje": "API de Neurona McCulloch-Pitts en funcionamiento",
        "threshold_optimo": int(np_neurona.threshold),
        "total_caracteristicas": len(cancer_mama.feature_names)
    }

@app.post("/predecir")
def predecir(data: InputData):
    if len(data.features) != 30:
        return {"error": "Se requieren exactamente 30 características numéricas."}
    
    # Binarizar la entrada usando los rangos exactos de X_train
    input_bin_list = []
    for i, col in enumerate(X_train.columns):
        min_val = X_train[col].min()
        max_val = X_train[col].max()
        val = data.features[i]
        mid = (min_val + max_val) / 2
        
        if val <= mid:
            input_bin_list.append(1)
        else:
            input_bin_list.append(0)
            
    input_bin_array = np.array([input_bin_list])
    
    # Realizar predicción con la neurona MP
    pred = np_neurona.predict(input_bin_array)[0]
    
    resultado_texto = "Benigno (1)" if pred else "Maligno (0)"
    
    return {
        "prediccion_booleana": bool(pred),
        "resultado": resultado_texto,
        "threshold_utilizado": int(np_neurona.threshold)
    }