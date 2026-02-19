import argparse
import pandas as pd
import os
from google.cloud import storage
from model import DelayModel

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de Entrenamiento de Atrasos")
    
    # Argumentos principales usados en el flujo actual
    parser.add_argument('--data_path', type=str, required=True, help="Ruta local al dataset")
    parser.add_argument('--bucket_name', type=str, required=True, help="Nombre del bucket en GCS")
    parser.add_argument('--commit_sha', type=str, required=True, help="SHA del commit para versionamiento")
    parser.add_argument('--model_path', type=str, required=True, help="Ruta local para guardar el modelo ONNX")
    
    # Argumentos extra mapeados desde GitHub Actions (listos para usarse si DelayModel los requiere)
    parser.add_argument('--project_id', type=str, required=False)
    parser.add_argument('--model_secret_key', type=str, required=False, default="")
    parser.add_argument('--delay_threshold_minutes', type=int, required=False, default=15)
    parser.add_argument('--learning_rate', type=float, required=False, default=0.01)
    parser.add_argument('--random_state', type=int, required=False, default=1)
    parser.add_argument('--top_features', type=str, required=False, default="")

    return parser.parse_args()

def main():
    # 1. Capturamos los argumentos de línea de comandos en lugar de os.getenv
    args = parse_args()
    
    BUCKET_NAME = args.bucket_name
    COMMIT_SHA = args.commit_sha
    DATA_PATH = args.data_path
    LOCAL_MODEL_PATH = args.model_path

    # Deshacemos el truco del pipe (|) por si necesitas filtrar las columnas en el futuro
    top_features_list = args.top_features.split('|') if args.top_features else []

    print(f"Iniciando Pipeline de Entrenamiento para el commit {COMMIT_SHA}...")
    
    # 2. Verificar si el archivo de datos existe
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"No se encontró el dataset en {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    model = DelayModel()
    
    print("Generando features y entrenando...")
    # Si quisieras usar los features pasados por GitHub Actions, podrías inyectar top_features_list aquí
    features, target = model.preprocess(data, target_column="delay")
    model.fit(features, target)
    
    print("Guardando modelo localmente en ONNX...")
    model.save_model(LOCAL_MODEL_PATH)

    # 3. Subida a Google Cloud Storage
    print(f"Subiendo a bucket {BUCKET_NAME}...")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    # Guardamos con el SHA para tener versionamiento
    blob = bucket.blob(f"models/model_{COMMIT_SHA}.onnx")
    blob.upload_from_filename(LOCAL_MODEL_PATH)
    
    print("Proceso completado con éxito.")

if __name__ == "__main__":
    main()