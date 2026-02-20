# LATAM Airlines - Flight Delay Prediction (MLOps Challenge)

Este repositorio contiene la solución al desafío técnico para el rol de Machine Learning Engineer. El objetivo principal es operacionalizar un modelo de predicción de atrasos de vuelos utilizando prácticas de MLOps, garantizando trazabilidad, escalabilidad y despliegues automatizados.

## Arquitectura MLOps y Flujo CI/CD/CT

El proyecto implementa un orquestador maestro en GitHub Actions que coordina tres flujos principales:

1. **Continuous Integration (CI):** - Ejecuta validaciones de código y pruebas unitarias/integración (`make model-test`, `make api-test`).
   - Verifica el *Code Coverage* de las piezas críticas del modelo y la API.

2. **Continuous Training (CT):** - Ante cambios en la rama `develop` o `main`, lanza un Custom Job en Google Cloud Vertex AI.
   - El script de entrenamiento (`model_train.py`) se ejecuta en una máquina `n1-highmem-2`, serializa el modelo en formato ONNX y lo guarda en Google Cloud Storage (GCS).
   - **Enfoque GitOps:** Una vez que Vertex AI termina exitosamente, el pipeline realiza un commit automático actualizando el archivo `model_config.json` con el SHA del commit que disparó el entrenamiento. Esto garantiza una trazabilidad perfecta: *Commit = Versión del Modelo = Despliegue*.

3. **Continuous Deployment (CD):** - Lee la versión del modelo desde `model_config.json`.
   - Empaqueta la API de FastAPI en un contenedor Docker.
   - Despliega la aplicación en Google Cloud Run, la cual descarga dinámicamente el modelo `.onnx` correspondiente desde GCS durante el arranque.

## Tecnologías y Herramientas

* **Machine Learning:** Scikit-Learn, Pandas, XGBoost/LogisticRegression.
* **Serialización:** ONNX (Open Neural Network Exchange) para una inferencia más rápida y agnóstica al framework.
* **API:** FastAPI, Uvicorn, Pydantic (Validación estricta de esquemas).
* **Orquestación y CI/CD:** GitHub Actions (Workflows modulares).
* **Cloud Platform (GCP):** * Vertex AI (Entrenamiento serverless).
  * Cloud Storage (Registro de artefactos/modelos).
  * Artifact Registry (Imágenes Docker).
  * Cloud Run (Inferencia serverless).

## Estructura del Proyecto

```text
├── .github/workflows/   # Orquestador y pipelines CI/CT/CD
├── challenge/           # Código fuente principal
│   ├── model.py         # Clase DelayModel y lógica del modelo
│   ├── model_train.py   # Script ejecutable para entrenamiento en Vertex AI
│   └── api.py           # Endpoints de FastAPI
├── data/                # Datasets (ignorados en git, excepto muestras para tests)
├── tests/               # Pruebas unitarias y de estrés (API y Modelo)
├── model_config.json    # Estado GitOps: Indica la versión ONNX actual en producción
├── Dockerfile           # Receta de la imagen para Cloud Run
└── Makefile             # Comandos automatizados (build, test, install)