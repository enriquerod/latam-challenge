import xgboost
print(xgboost.__version__)

# import pandas as pd
# import os
# from dotenv import load_dotenv
# from google.cloud import storage
# from model import DelayModel

# BUCKET_NAME = os.getenv("BUCKET_NAME", "tu-bucket-gcp") # REEMPLAZAR EN PIPELINE
# COMMIT_SHA = os.getenv("COMMIT_SHA", "latest")
# DATA_PATH = "data/data.csv"
# LOCAL_MODEL_PATH = "/tmp/delay_model.onnx"

# def main():
#     print("🚀 Iniciando Pipeline de Entrenamiento...")
#     data = pd.read_csv('../data/data.csv')
#     model = DelayModel()
    
#     print("⚙️ Generando features...")
#     features, target = model.preprocess(data, target_column="delay")
    
#     print("🧠 Entrenando XGBoost...")
#     model.fit(features, target)
    
#     print("📦 Serializando a ONNX...")
#     model.save_model('./')

# if __name__ == "__main__":
#     load_dotenv()
#     main()