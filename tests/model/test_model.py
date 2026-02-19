import unittest
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from challenge.model import DelayModel

class TestModel(unittest.TestCase):

    FEATURES_COLS = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    TARGET_COL = [
        "delay"
    ]


    def setUp(self) -> None:
        super().setUp()
        self.model = DelayModel()
        self.data = pd.read_csv(filepath_or_buffer="./data/data.csv")

    def test_model_preprocess_for_training(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

        assert isinstance(target, pd.DataFrame)
        assert target.shape[1] == len(self.TARGET_COL)
        assert set(target.columns) == set(self.TARGET_COL)


    def test_model_preprocess_for_serving(
        self
    ):
        features = self.model.preprocess(
            data=self.data
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)


    def test_model_fit(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        _, features_validation, _, target_validation = train_test_split(features, target, test_size = 0.33, random_state = 42)

        self.model.fit(
            features=features,
            target=target
        )

        predicted_target = self.model._model.predict(
            features_validation
        )

        report = classification_report(target_validation, predicted_target, output_dict=True)
        
        assert report["0"]["recall"] < 0.60
        assert report["0"]["f1-score"] < 0.70
        assert report["1"]["recall"] > 0.60
        assert report["1"]["f1-score"] > 0.30


    def test_model_predict(
        self
    ):
        features = self.model.preprocess(
            data=self.data
        )

        predicted_targets = self.model.predict(
            features=features
        )

        assert isinstance(predicted_targets, list)
        assert len(predicted_targets) == features.shape[0]
        assert all(isinstance(predicted_target, int) for predicted_target in predicted_targets)


    # ==========================================
    # NUEVOS TESTS PARA ALCANZAR 100% COVERAGE
    # ==========================================
    
    def test_model_custom_init(self):
        """Cubre las ramas 'if arg is not None' en el constructor __init__"""
        custom_model = DelayModel(
            top_features=["MES_7"], 
            delay_threshold=20, 
            random_state=42, 
            model_version="2.0"
        )
        self.assertEqual(custom_model.delay_threshold, 20)
        self.assertEqual(custom_model.model_version, "2.0")
        self.assertEqual(custom_model.random_state, 42)
        self.assertEqual(custom_model.top_features, ["MES_7"])

    def test_model_preprocess_missing_columns(self):
        """Cubre la rama if col not in features.columns en preprocess"""
        # Forzamos una feature que no existe en la data original
        custom_model = DelayModel(top_features=["COLUMNA_FALSA"])
        features = custom_model.preprocess(self.data)
        self.assertIn("COLUMNA_FALSA", features.columns)
        self.assertEqual(features["COLUMNA_FALSA"].sum(), 0)

    def test_model_save_and_load_onnx(self):
        """Cubre save_model, load_model y la predicción nativa con ONNX"""
        import os
        
        # 1. Entrenamos para poder guardar
        features, target = self.model.preprocess(data=self.data, target_column="delay")
        self.model.fit(features, target)
        
        # 2. Guardamos (cubre save_model)
        test_path = "test_coverage.onnx"
        self.model.save_model(test_path)
        self.assertTrue(os.path.exists(test_path))
        
        # 3. Cargamos y predecimos (cubre load_model y el if onnx_session en predict)
        new_model = DelayModel()
        new_model.load_model(test_path)
        preds = new_model.predict(features)
        
        self.assertIsInstance(preds, list)
        
        # Limpieza del archivo residual
        if os.path.exists(test_path):
            os.remove(test_path)

    def test_model_predict_sklearn(self):
        """Cubre la rama donde se usa _model de sklearn en memoria antes de exportar a ONNX"""
        # 1. Preprocesamos y entrenamos el modelo
        features, target = self.model.preprocess(data=self.data, target_column="delay")
        self.model.fit(features, target)
        
        # 2. Nos aseguramos explícitamente de que no haya una sesión ONNX cargada
        self.model._onnx_session = None 
        
        # 3. Al predecir ahora, el código entrará obligatoriamente al 'if self._model is not None:'
        preds = self.model.predict(features)
        
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), len(features))

    def test_model_predict_fallback(self):
        """Cubre el except FileNotFoundError y el RuntimeError final en predict()"""
        empty_model = DelayModel()
        features = empty_model.preprocess(self.data)
        
        # Caso A: Forzamos model_path a None (ignora el .env)
        # Esto hace que pase de largo las validaciones y llegue al RuntimeError final
        empty_model.model_path = None
        with self.assertRaises(RuntimeError):
            empty_model.predict(features)
            
        # Caso B: Forzamos un path que no existe
        # Esto entra al bloque "try load_model", salta al "except FileNotFoundError", 
        # y luego llega al RuntimeError final
        empty_model.model_path = "falso_modelo_inexistente.onnx"
        with self.assertRaises(RuntimeError):
            empty_model.predict(features)

    def test_model_save_no_model(self):
        """Cubre el return temprano en save_model cuando no hay modelo entrenado"""
        empty_model = DelayModel()
        empty_model.save_model("dummy.onnx") # No debe arrojar error
        
    def test_model_load_not_found(self):
        """Cubre el FileNotFoundError en load_model explícito"""
        empty_model = DelayModel()
        with self.assertRaises(FileNotFoundError):
            empty_model.load_model("archivo_que_no_existe.onnx")