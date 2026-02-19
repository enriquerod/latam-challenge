import pandas as pd
import numpy as np
import xgboost as xgb
import os
from typing import Tuple, Union, List

class DelayModel:

    def __init__(self):
        self._model = None
        self._onnx_session = None

        # ENV VARS
        self.top_features = [f.strip() for f in os.getenv("TOP_FEATURES", "").split(",") if f.strip()]
        self.delay_threshold = int(os.getenv("DELAY_THRESHOLD_MINUTES"))
        self.model_path = os.getenv("MODEL_PATH")
        self.learning_rate = float(os.getenv("LEARNING_RATE"))
        self.random_state = int(os.getenv("RANDOM_STATE"))
        self.model_version = os.getenv("MODEL_VERSION")

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
        n_y0 = len(target[target.iloc[:, 0] == 0])
        n_y1 = len(target[target.iloc[:, 0] == 1])
        scale = n_y0 / n_y1 if n_y1 > 0 else 1.0

        self._model = xgb.XGBClassifier(
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale,
            eval_metric="logloss"
        )

        self._model.fit(features, target)

    # ==========================
    # Predicción
    # ==========================
    def predict(self, features: pd.DataFrame) -> List[int]:

        if self._onnx_session is not None:
            input_name = self._onnx_session.get_inputs()[0].name
            output_name = self._onnx_session.get_outputs()[0].name
            input_data = features.astype(np.float32).values

            preds = self._onnx_session.run(
                [output_name],
                {input_name: input_data}
            )
            return preds[0].tolist()

        if self._model is not None:
            return self._model.predict(features).tolist()

        # TODO: despues cargar desde  gcp bucket
        if self._onnx_session is not None:
            return self.predict(features)
        
        raise RuntimeError("No hay ningún modelo cargado para realizar predicciones")

    # ==========================
    # MODELO PARA ARTEFACTO
    # ==========================
    def save_model(self, filepath: str = None) -> None:
        if self._model is None:
            return

        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        path = filepath or self.model_path

        initial_type = [
            ("float_input", FloatTensorType([None, len(self.top_features)]))
        ]

        onnx_model = onnxmltools.convert_xgboost(
            self._model,
            initial_types=initial_type
        )

        # onnx_model.model_version = 1
        onnx_model.doc_string = "Delay Prediction Model - LATAM Airlines"

        meta_v = onnx_model.metadata_props.add()
        meta_v.key = "version"
        meta_v.value = str(self.model_version)

        meta_thr = onnx_model.metadata_props.add()
        meta_thr.key = "delay_threshold_minutes"
        meta_thr.value = str(self.delay_threshold)
        
         # TODO: despues subir a gcp bucket
        onnxmltools.utils.save_model(onnx_model, path)

     # TODO: despues cargar desde  gcp bucket
    def load_model(self, filepath: str = None) -> None:
        import onnxruntime as onnx_rt

        path = filepath or self.model_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo no encontrado: {path}")

        self._onnx_session = onnx_rt.InferenceSession(path)