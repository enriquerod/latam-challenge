import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from typing import Tuple, Union, List

from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


class DelayModel:

    def __init__(
        self, 
        top_features: List[str] = None, 
        delay_threshold: int = None, 
        random_state: int = None, 
        model_version: str = None
    ):
        # Cargamos el .env para las pruebas locales 
        load_dotenv()
        
        self._model = None
        self._onnx_session = None

        # ==========================================
        # Args y env vars
        # ==========================================
        if top_features is not None:
            self.top_features = top_features
        else:
            env_tf = os.getenv("TOP_FEATURES", "")
            self.top_features = [f.strip() for f in env_tf.split(",") if f.strip()]

        if delay_threshold is not None:
            self.delay_threshold = delay_threshold
        else:
            self.delay_threshold = int(os.getenv("DELAY_THRESHOLD_MINUTES", 15))

        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = int(os.getenv("RANDOM_STATE", 1))

        if model_version is not None:
            self.model_version = model_version
        else:
            self.model_version = os.getenv("MODEL_VERSION", "1.0")

        # El path se define al guardar o cargar el modelo
        self.model_path = os.getenv("MODEL_PATH")


    # ==========================
    # Preprocesamiento
    # ==========================
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:

        features = pd.concat([
            pd.get_dummies(data["OPERA"], prefix="OPERA"),
            pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
            pd.get_dummies(data["MES"], prefix="MES"),
        ], axis=1)

        # asegurar todas las top_features
        if self.top_features:
            for col in self.top_features:
                if col not in features.columns:
                    features[col] = 0
            features = features[self.top_features]

        if target_column == "delay":
            data["Fecha-O"] = pd.to_datetime(data["Fecha-O"])
            data["Fecha-I"] = pd.to_datetime(data["Fecha-I"])

            data["min_diff"] = (
                (data["Fecha-O"] - data["Fecha-I"]).dt.total_seconds() / 60
            )

            data["delay"] = np.where(
                data["min_diff"] > self.delay_threshold, 1, 0
            )

            target = data[[target_column]]
            return features, target

        return features

    # ==========================
    # Entrenamiento
    # ==========================
    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        self._model = LogisticRegression(
            class_weight="balanced",
            random_state=self.random_state,
            max_iter=1000,
            solver="lbfgs"
        )

        self._model.fit(features, target.values.ravel())

    # ==========================
    # Predicción
    # ==========================
    def predict(self, features: pd.DataFrame) -> List[int]:
        # Si hay una sesion ONNX
        if self._onnx_session is not None:
            input_name = self._onnx_session.get_inputs()[0].name
            output_name = self._onnx_session.get_outputs()[0].name
            preds = self._onnx_session.run(
                [output_name],
                {input_name: features.astype(np.float32).values}
            )
            return preds[0].tolist()

        # Si hay un modelo sklearn
        if self._model is not None:
            return self._model.predict(features).tolist()

        # Intentar cargar ONNX
        if self.model_path:
            try:
                self.load_model(self.model_path)  
                return self.predict(features)  # recursivo para cargar la sesion
            except FileNotFoundError:
                pass
                
        raise RuntimeError("No hay ningun modelo cargado para realizar predicciones")

    # ==========================
    # Guardar modelo ONNX
    # ==========================
    def save_model(self, filepath: str = None) -> None:
        if self._model is None:
            return

        path = filepath or self.model_path

        num_features = len(self.top_features) if self.top_features else self._model.n_features_in_

        initial_type = [
            ("float_input", FloatTensorType([None, num_features]))
        ]

        onnx_model = convert_sklearn(
            self._model,
            initial_types=initial_type
        )

        onnx_model.doc_string = "Delay Prediction Model - LATAM Airlines"

        meta_v = onnx_model.metadata_props.add()
        meta_v.key = "version"
        meta_v.value = str(self.model_version)

        meta_thr = onnx_model.metadata_props.add()
        meta_thr.key = "delay_threshold_minutes"
        meta_thr.value = str(self.delay_threshold)

        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())

    # ==========================
    # Cargar modelo ONNX
    # ==========================
    def load_model(self, filepath: str = None) -> None:
        import onnxruntime as onnx_rt

        path = filepath or self.model_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo no encontrado: {path}")

        self._onnx_session = onnx_rt.InferenceSession(path)