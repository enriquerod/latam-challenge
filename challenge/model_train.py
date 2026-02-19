import pandas as pd
import os
from dotenv import load_dotenv
from google.cloud import storage
from model import DelayModel
from sklearn.metrics import accuracy_score, classification_report

BUCKET_NAME = os.getenv("BUCKET_NAME") 
COMMIT_SHA = os.getenv("COMMIT_SHA")
DATA_PATH = "data/data.csv"
LOCAL_MODEL_PATH = "/tmp/delay_model.onnx"

def main():
    print("Iniciando Pipeline de Entrenamiento...")
    data = pd.read_csv('../data/data.csv')
    model = DelayModel()
    
    print("Generando features...")
    features, target = model.preprocess(data, target_column="delay")
    
    print("Entrenando XGBoost...")
    model.fit(features, target)
    
    print("Conversion a ONNX...")
    model.save_model('./delay_model.onnx')

    # TEST RAPIDO 
    print("Verificando predicciones ONNX...")
    test_model = DelayModel()
    test_model.model_path = "./delay_model.onnx" 
    test_model.load_model()            

    predictions = test_model.predict(features)

    # Convertir target a lista simple
    true_values = target.values.ravel().tolist()

    # Comparación básica
    accuracy = accuracy_score(true_values, predictions)
    report = classification_report(true_values, predictions, digits=3)

    print(f"Accuracy sobre dataset de entrenamiento: {accuracy:.3f}")
    print("Reporte completo:")
    print(report)

if __name__ == "__main__":
    load_dotenv()
    main()