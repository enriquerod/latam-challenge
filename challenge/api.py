import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List
from contextlib import asynccontextmanager
from challenge.model import DelayModel


# Instanciamos el modelo de manera global
model = DelayModel()

# Logica de startup para carga del modelo ONNX
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Usa la ruta absoluta o asegúrate de que el archivo exista
        model.load_model("./delay_model.onnx") 
        print("Modelo ONNX cargado exitosamente")
    except Exception as e:
        print(f"ERROR CRÍTICO: No se pudo cargar el modelo: {e}")
    yield
    
    # Logica de shutdown
    print("Apagando la API y liberando recursos")

# Lifespan al instanciar FastAPI
app = FastAPI(title="SCL Delay Prediction API", lifespan=lifespan)


# Por defecto, Pydantic/FastAPI devuelven HTTP 422 cuando la validación falla.
# Los tests del challenge exigen estrictamente un HTTP 400. Esto sobrescribe el comportamiento.
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": "Bad Request: Error en la validacion de datos", "errors": exc.errors()}
    )


# Validacion Pydantic

class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator('TIPOVUELO')
    def validate_tipo_vuelo(cls, v):
        if v not in ['I', 'N']:
            raise ValueError("TIPOVUELO debe ser 'I' (Internacional) o 'N' (Nacional)")
        return v

    @validator('MES')
    def validate_mes(cls, v):
        if v < 1 or v > 12:
            raise ValueError("MES debe estar entre 1 y 12")
        return v

class FlightList(BaseModel):
    flights: List[Flight]


# ENDPOINTS

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(payload: FlightList) -> dict:
    # lista Pydantic a un DataFrame de Pandas
    df = pd.DataFrame([flight.dict() for flight in payload.flights])
    
    # Preprocesamos las features
    features = model.preprocess(df)
    
    # Prediccion con ONNX
    predictions = model.predict(features)
    
    return {"predict": predictions}