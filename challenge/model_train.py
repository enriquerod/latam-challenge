import pandas as pd
import os
from google.cloud import storage
from model import DelayModel

def main():
    BUCKET_NAME = os.getenv("BUCKET_NAME") 
    COMMIT_SHA = os.getenv("COMMIT_SHA")
    DATA_PATH = "data/data.csv" # Ruta relativa 
    LOCAL_MODEL_PATH = "delay_model.onnx"

    print(f"Iniciando Pipeline de Entrenamiento para el commit {COMMIT_SHA}...")
    
    # Verificar si el archivo de datos existe
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"No se encontró el dataset en {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    model = DelayModel()
    
    print("Generando features y entrenando...")
    features, target = model.preprocess(data, target_column="delay")
    model.fit(features, target)
    
    print("Guardando modelo localmente en ONNX...")
    model.save_model(LOCAL_MODEL_PATH)

    # Subida a Google Cloud Storage
    print(f"Subiendo a bucket {BUCKET_NAME}...")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    # Guardamos con el SHA para tener versionamiento
    blob = bucket.blob(f"models/model_{COMMIT_SHA}.onnx")
    blob.upload_from_filename(LOCAL_MODEL_PATH)
    print("Proceso completado")

if __name__ == "__main__":
    main()