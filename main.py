#fastapi con modelos pkl y limpieza ya lista
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import os

# cargar el modelo entrenado y las características
model = joblib.load('decision_tree_model.pkl')
model_features = joblib.load('model_features.pkl')
app = FastAPI()
class HouseFeatures(BaseModel):
    superficie: float
    habitaciones: int
    antiguedad: int
    ubicacion: int  # Asegúrate de que la ubicación se codifique como un entero
class PredictionRequest(BaseModel):
    houses: List[HouseFeatures]

# get
@app.get("/")
def read_root():
    return {"message": "API de predicción de precios de casas. Usa el endpoint /predict para hacer predicciones."}
@app.post("/predict")
def predict(request: PredictionRequest):
    # convertir la lista de características a un DataFrame
    input_data = pd.DataFrame([house.dict() for house in request.houses])
    
    # Asegurarse de que las columnas estén en el mismo orden que durante el entrenamiento
    input_data = input_data[model_features]
    
    # hacer predicciones
    predictions = model.predict(input_data)
    
    # mapear predicciones a etiquetas legibles
    labels = ['barato' if pred == 0 else 'caro' for pred in predictions]
    
    return {"predictions": labels}
if __name__ == "__main__":
    uvicorn.run(app, host="", port=int(os.environ.get("PORT", 8000)))  # Puerto dinámico para despliegue

# ejemplos para probar la api
# ejemplo de petición
# {
#   "houses": [
#     {
#       "superficie": 120.0,
#       "habitaciones": 3,
#       "antiguedad": 5,
#       "ubicacion": 1
#     }
