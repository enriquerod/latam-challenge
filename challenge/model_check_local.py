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

    # metadatos
    meta = test_model._onnx_session.get_modelmeta()
    print("=== Metadatos ONNX ===")
    print(f"Producer name: {meta.producer_name}")
    print(f"Description: {meta.description}")
    print(f"Version: {meta.version}")
    print("Propiedades adicionales:")
    keys = meta.custom_metadata_map.keys()
    for key in keys:
        value = meta.custom_metadata_map[key]
        print(f"  {key}: {value}")

    # Predicciones
    predictions = test_model.predict(features)

    true_values = target.values.ravel().tolist()
    accuracy = accuracy_score(true_values, predictions)
    report = classification_report(true_values, predictions, digits=3)

    print(f"Accuracy sobre dataset: {accuracy:.3f}")
    print("Reporte completo:")
    print(report)

if __name__ == "__main__":
    load_dotenv()
    main()