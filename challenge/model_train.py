import argparse
import pandas as pd
import os
from google.cloud import storage
from challenge.model import DelayModel # Asegúrate de usar la ruta correcta según tu estructura

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de Entrenamiento de Atrasos")
    
    parser.add_argument('--data_path', type=str, required=True, help="Ruta local al dataset")
    parser.add_argument('--bucket_name', type=str, required=True, help="Nombre del bucket en GCS")
    parser.add_argument('--commit_sha', type=str, required=True, help="SHA del commit para versionamiento")
    parser.add_argument('--model_path', type=str, required=True, help="Ruta local para guardar el modelo ONNX")
    
    parser.add_argument('--project_id', type=str, required=False)
    parser.add_argument('--delay_threshold_minutes', type=int, required=False, default=15)
    parser.add_argument('--learning_rate', type=float, required=False, default=0.01)
    parser.add_argument('--random_state', type=int, required=False, default=1)
    parser.add_argument('--top_features', type=str, required=False, default="")

    return parser.parse_args()

def main():
    args = parse_args()
    
    BUCKET_NAME = args.bucket_name
    COMMIT_SHA = args.commit_sha
    DATA_PATH = args.data_path
    LOCAL_MODEL_PATH = args.model_path

    # 1. Procesar la lista de features desde los argumentos (usando el separador '|')
    # Limpiamos espacios en blanco por si acaso
    top_features_list = [f.strip() for f in args.top_features.split('|') if f.strip()]

    print(f"Iniciando Pipeline de Entrenamiento para el commit {COMMIT_SHA}...")
    print(f"Features seleccionadas ({len(top_features_list)}): {top_features_list}")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"No se encontró el dataset en {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    # 2. INSTANCIAR EL MODELO PASANDO LAS TOP_FEATURES
    # Esto asegura que el preprocesamiento y el guardado ONNX usen el tamaño correcto (10)
    model = DelayModel(
        top_features=top_features_list,
        delay_threshold=args.delay_threshold_minutes,
        random_state=args.random_state
    )
    
    print("Generando features y entrenando...")
    # Ahora preprocess devolverá solo las columnas en top_features_list
    features, target = model.preprocess(data, target_column="delay")
    
    print(f"Dimensiones de los features: {features.shape}")
    model.fit(features, target)
    
    # 3. GUARDAR EL MODELO
    # El archivo ONNX ahora tendrá un input_shape coincidente con len(top_features_list)
    print("Guardando modelo localmente en ONNX...")
    model.save_model(LOCAL_MODEL_PATH)

    # Subida a Google Cloud Storage
    print(f"Subiendo a bucket {BUCKET_NAME}...")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    gcs_blob_path = f"models/{COMMIT_SHA}/delay_model.onnx"
    
    blob = bucket.blob(gcs_blob_path)
    blob.upload_from_filename(LOCAL_MODEL_PATH)
    
    print(f"Modelo guardado en GCS: gs://{BUCKET_NAME}/{gcs_blob_path}")

if __name__ == "__main__":
    main()