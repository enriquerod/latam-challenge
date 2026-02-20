# Flight Delay Prediction 
By: Luis Enrique Rodriguez

#### <span style="color:red;"> QUEDO POR TERMINAR </span>
##### Me falto contruir mas pruebas unitarias para aumentar el coverage del api.py y del script model_train.py

---
---
## Parte I & II

### Selección del Modelo: Regresión Logística vs. XGBoost
Tras analizar los resultados del Data Scientist en el notebook de exploración, se evaluaron dos candidatos principales: XGBoost y Regresión Logística (ambos con balanceo de clases `class_weight='balanced'`). 

**Decisión:** Se optó por la **Regresión Logística**.
**Justificación:**
1. **Rendimiento Similar en la Métrica Clave:** Dado el desbalanceo del dataset, la métrica crítica es el *Recall* de la clase 1 (predecir correctamente los atrasos). Ambos modelos mostraron un Recall y F1-Score muy similares para la clase minoritaria.
2. **Eficiencia Computacional:** A igualdad de rendimiento predictivo, se prefiere el modelo más simple. La Regresión Logística requiere significativamente menos poder de cómputo y memoria para inferir en tiempo real.
3. **Explicabilidad:** Es matemáticamente transparente, lo que permite al equipo de negocio entender exactamente cómo cada *feature* impacta la probabilidad de atraso.

Para optimizar la latencia y evitar el sobreajuste, el modelo se entrenó exclusivamente con las 10 características más predictivas identificadas.

### Serialización ci ONNX
En lugar de exportar el modelo usando el estándar `pickle` o `joblib` de Scikit-Learn, se implementó una transformación a **ONNX (Open Neural Network Exchange)**. 
1. **Seguridad:** Los archivos Pickle pueden ejecutar código arbitrario; ONNX es un formato de solo datos estructurales, mitigando riesgos de seguridad en la API.
2. **Desacoplamiento e Imagen Ligera:** El contenedor de producción en Cloud Run **no necesita instalar dependencias pesadas** como `scikit-learn` o `pandas`, solo `onnxruntime`. Esto reduce drásticamente el tamaño de la imagen Docker y acelera los tiempos de *Cold Start*.

### Buenas Prácticas: Parametrización y Agnosticismo de Entorno
Para garantizar buenas pracitcas, **toda la configuración y dependencias de la infraestructura están parametrizadas**. 
Tanto el script de entrenamiento en la Parte 1 (`model_train.py`), la API en la Parte 2 (`api.py`), y el despliegue en la Parte 3 son completamente agnósticos al ambiente. Valores críticos como el `PROJECT_ID`, `BUCKET_NAME`, `MODEL_PATH`, configuraciones de red y umbrales no están *hardcodeados*, sino que se inyectan dinámicamente mediante **variables de entorno**. 
Esto asegura que exactamente el mismo código base y la misma imagen Docker puedan ser promovidos desde desarrollo local hasta producción sin requerir alteraciones internas, eliminando discrepancias entre entornos.

---

## Arquitectura Cloud y Organización en GCP



Para el ecosistema en Google Cloud Platform, se aplicó un diseño de arquitectura *Enterprise-grade* centrado en la seguridad y la escalabilidad:

* **Jerarquía de Recursos (Folders):** Se creó un *Folder* dedicado a nivel organización para la unidad de negocio / dominio del proyecto.
* **Aislamiento de Ambientes (Projects):** Debajo de este folder, se dividieron los ambientes en **Proyectos de GCP completamente aislados** (`dev`, `test`, `prod`). 
* **Ventajas de este enfoque:** - Separación estricta de facturación (Billing) y cuotas por ambiente.
  - Aislamiento de seguridad mediante políticas de IAM (Principio de Menor Privilegio), asegurando que los roles de desarrollo no tengan acceso destructivo a producción.
  - Prevención de incidentes en cascada: un error o saturación de carga en las pruebas de estrés de `test` no afecta los recursos del sistema en vivo en `prod`.


## Tecnologías y Herramientas

* **Machine Learning:** Scikit-Learn, Pandas, XGBoost/LogisticRegression.
* **Serialización:** ONNX (Open Neural Network Exchange) para una inferencia más rápida y agnóstica al framework.
* **API:** FastAPI, Uvicorn, Pydantic (Validación estricta de esquemas).
* **Orquestación y CI/CD:** GitHub Actions (Workflows modulares).
* **Cloud Platform (GCP):** * Vertex AI (Entrenamiento serverless).
  * Cloud Storage (Registro de artefactos/modelos).
  * Artifact Registry (Imágenes Docker).
  * Cloud Run (Inferencia serverless).
---

## Flujo de Trabajo y Promoción entre Ambientes



Para maximizar el desarrollo, todo el flujo está automatizado. Los desarrolladores solo se preocupan por escribir código y el pipeline orquesta el resto. La promoción del modelo y la API hacia *upper environments* (Test y Prod) está estrictamente ligada a la estrategia de ramas (GitFlow):

1. **`feature/*` (Desarrollo Local):** Los desarrolladores crean ramas *feature* para experimentar. Al abrir un Pull Request, el **CI** valida el código (`make model-test`, `make api-test`).
2. **`develop` -> Ambiente `DEV`:** Al fusionar en `develop`, el orquestador dispara el **CT**. Vertex AI entrena un nuevo modelo con datos frescos. El despliegue a Cloud Run en el proyecto `dev` ocurre automáticamente, permitiendo experimentación rápida.
3. **`release/*` -> Ambiente `TEST`:** Cuando se prepara una versión candidata, se crea una rama de `release`. El pipeline despliega la API en el proyecto `test`. **Aquí no se re-entrena el modelo**; el pipeline toma el artefacto `.onnx` exacto generado en `dev` (gracias al SHA guardado) para garantizar que se pruebe exactamente lo mismo que se validó. Aquí corren las pruebas de estrés (`make stress-test`).
4. **`main` -> Ambiente `PROD`:** Tras aprobar el PR de release, el código llega a `main`. El pipeline orquesta el despliegue final al proyecto `prod` en Cloud Run, exponiendo la API al tráfico real con cero *downtime*.

---

## Parte III: Arquitectura MLOps y Flujo CI/CD/CT

El orquestador maestro en GitHub Actions coordina las tres piezas clave:

1. **Continuous Integration (CI):** - Verifica el comportamiento y *Code Coverage* de las funciones y lógica de la clase `DelayModel`.
   - Verifica el comportamiento y *Code Coverage* de los endpoints de la API FastAPI.

2. **Continuous Training (CT):** - El script de entrenamiento se ejecuta en Vertex AI (`n1-highmem-2`), serializa el modelo a ONNX y lo guarda de forma segura en Cloud Storage.
   - **Enfoque GitOps:** Una vez que Vertex AI termina, el pipeline realiza un commit automático actualizando el archivo `model_config.json` con el SHA del commit que disparó el entrenamiento.
   - **Trazabilidad Absoluta:** Este *SHA corto* es la columna vertebral de la trazabilidad. Versiona el artefacto ONNX en el bucket, la imagen Docker en Artifact Registry, y nombra la revisión (`--revision-suffix`) del endpoint en Cloud Run. *Commit = Versión del Modelo = Despliegue*.

3. **Continuous Deployment (CD):** - Para **DEV**, toma el SHA generado en caliente por el CT. Para **TEST y PROD**, lee la versión estabilizada del modelo desde `model_config.json`.
   - Empaqueta la API en un contenedor Docker.
   - Despliega la aplicación en Cloud Run, inyectando las variables de entorno correspondientes al proyecto (dev/test/prod) y descargando dinámicamente el modelo `.onnx`.


<span style="color:red; font-size:24px; font-weight:bold;">
  MEJORAS
</span>

- **Terminar de hacer el unit test para cumplir coverage**
- **PRacticas de seguridad en los servicios de GCP**
- **Implementar pipelines de seguridad para escaneo de codigo y de imagenes docker**
- **Infraestructura como Código (IaC) con Terraform**




### Estructura del Proyecto

```text
├── .github/workflows/   # Orquestador y pipelines CI/CT/CD
├── challenge/           # 
│   ├── model.py         # Clase DelayModel y lógica de preprocesamiento/inferencia
│   ├── model_train.py   # Script ejecutable para entrenamiento en Vertex AI
│   └── api.py           # Endpoints de FastAPI con validación Pydantic
├── data/                # Dataset
├── docs/                # Documentación del desafío (challenge.md)
├── tests/               # Pruebas unitarias
├── model_config.json    # Estado del model version: Indica la versión ONNX actual
├── Dockerfile           # Imagen para Cloud Run
└── Makefile             # Comandos automatizados (build, test, stress-test)
```



