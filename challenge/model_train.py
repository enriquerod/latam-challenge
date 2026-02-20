import argparse
import pandas as pd
import os
import sys
from google.cloud import storage


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

try:
    from challenge.model import DelayModel
except ModuleNotFoundError:
    from model import DelayModel

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de Entrenamiento de Atrasos")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--bucket_name', type=str, required=True)
    parser.add_argument('--commit_sha', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--project_id', type=str, required=False)
    parser.add_argument('--delay_threshold_minutes', type=int, required=False, default=15)
    parser.add_argument('--learning_rate', type=float, required=False, default=0.01)
    parser.add_argument('--random_state', type=int, required=False, default=1)
    parser.add_argument('--top_features', type=str, required=False, default="")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Procesar features (Separador '|')
    top_features_list = [f.strip() for f in args.top_features.split('|') if f.strip()]

    print(f"Iniciando Entrenamiento. Features: {len(top_features_list)}")
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"No hay data en {args.data_path}")

    data = pd.read_csv(args.data_path)

    # Instanciar con las top_features para que el ONNX sea de tamaño 10
    model = DelayModel(
        top_features=top_features_list,
        delay_threshold=args.delay_threshold_minutes,
        random_state=args.random_state
    )
    
    # Preprocess + Fit
    features, target = model.preprocess(data, target_column="delay")
    print(f"Dataset final: {features.shape}")
    model.fit(features, target)
    
    # Guardar local
    model.save_model(args.model_path)

    # Subida a GCS
    client = storage.Client()
    bucket = client.bucket(args.bucket_name)
    gcs_blob_path = f"models/{args.commit_sha}/delay_model.onnx"
    blob = bucket.blob(gcs_blob_path)
    blob.upload_from_filename(args.model_path)
    
    print(f"Éxito. Modelo en GCS: gs://{args.bucket_name}/{gcs_blob_path}")

if __name__ == "__main__":
    main()