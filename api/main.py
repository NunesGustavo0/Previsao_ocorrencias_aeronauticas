from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, ConfigDict, Field
from starlette.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parents[1]
TRACKING_URI = BASE_DIR / "pre_processamento" / "mlruns"
MODEL_NAME = "previsor_classificacao_ocorrencia"
MODEL_URI = f"models:/{MODEL_NAME}@production"

FEATURE_COLUMNS = [
    "PMD",
    "Numero_de_Assentos",
    "Ano_da_Ocorrencia",
    "Mes_da_Ocorrencia",
    "Dia_Semana_da_Ocorrencia",
    "PMD_ausente",
    "PMD_zero",
    "Numero_de_Assentos_ausente",
    "Numero_de_Assentos_zero",
    "ICAO_ausente",
    "Tipo_de_Aerodromo_ausente",
    "UF_indeterminada",
    "Fase_da_Operacao",
    "Regiao",
    "Operacao",
    "ICAO",
    "Tipo_de_Ocorrencia",
    "Tipo_de_Aerodromo",
    "UF",
    "Categoria_da_Aeronave",
    "Tipo_ICAO",
    "CLS",
    "Nome_do_Fabricante",
    "Municipio",
    "PMD_faixa",
]


class OcorrenciaFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "PMD": 2250.0,
                "Numero_de_Assentos": 4.0,
                "Ano_da_Ocorrencia": 2024,
                "Mes_da_Ocorrencia": 5,
                "Dia_Semana_da_Ocorrencia": 2,
                "PMD_ausente": 0,
                "PMD_zero": 0,
                "Numero_de_Assentos_ausente": 0,
                "Numero_de_Assentos_zero": 0,
                "ICAO_ausente": 0,
                "Tipo_de_Aerodromo_ausente": 0,
                "UF_indeterminada": 0,
                "Fase_da_Operacao": "POUSO",
                "Regiao": "SUDESTE",
                "Operacao": "PRIVADA",
                "ICAO": "SBSP",
                "Tipo_de_Ocorrencia": "FALHA DO MOTOR EM VOO",
                "Tipo_de_Aerodromo": "PUBLICO",
                "UF": "SP",
                "Categoria_da_Aeronave": "AVIAO",
                "Tipo_ICAO": "PA34",
                "CLS": "L1P",
                "Nome_do_Fabricante": "PIPER AIRCRAFT",
                "Municipio": "SAO PAULO",
                "PMD_faixa": "leve",
            }
        },
    )

    PMD: float | None = Field(None, ge=0)
    Numero_de_Assentos: float | None = Field(None, ge=0)
    Ano_da_Ocorrencia: int | None = Field(None, ge=1900, le=2100)
    Mes_da_Ocorrencia: int | None = Field(None, ge=1, le=12)
    Dia_Semana_da_Ocorrencia: int | None = Field(None, ge=0, le=6)
    PMD_ausente: int = Field(ge=0, le=1)
    PMD_zero: int = Field(ge=0, le=1)
    Numero_de_Assentos_ausente: int = Field(ge=0, le=1)
    Numero_de_Assentos_zero: int = Field(ge=0, le=1)
    ICAO_ausente: int = Field(ge=0, le=1)
    Tipo_de_Aerodromo_ausente: int = Field(ge=0, le=1)
    UF_indeterminada: int = Field(ge=0, le=1)
    Fase_da_Operacao: str | None = None
    Regiao: str | None = None
    Operacao: str
    ICAO: str | None = None
    Tipo_de_Ocorrencia: str | None = None
    Tipo_de_Aerodromo: str | None = None
    UF: str
    Categoria_da_Aeronave: str
    Tipo_ICAO: str | None = None
    CLS: str | None = None
    Nome_do_Fabricante: str | None = None
    Municipio: str | None = None
    PMD_faixa: str | None = None


class PredicaoResponse(BaseModel):
    predicao: str
    modelo: str


class SaudeResponse(BaseModel):
    ok: bool
    modelo: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.modelo = None
    app.state.modelo_erro = None
    app.state.modelo_uri = MODEL_URI

    try:
        mlflow.set_tracking_uri(TRACKING_URI.as_uri())
        app.state.modelo = mlflow.pyfunc.load_model(MODEL_URI)
    except Exception as exc:  # pragma: no cover - message is asserted via endpoint
        app.state.modelo_erro = str(exc)

    yield


app = FastAPI(
    title="API de Classificacao de Ocorrencias Aeronauticas",
    description="Endpoint de inferencia para o modelo registrado no MLflow Registry.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    swagger_ui_parameters={"docExpansion": "full"},
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "api" / "static"), name="static")


def _modelo_carregado() -> Any:
    if app.state.modelo is None:
        detalhe = app.state.modelo_erro or "Modelo nao carregado."
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao carregar o modelo {MODEL_URI}: {detalhe}",
        )
    return app.state.modelo


def _montar_dataframe(features: OcorrenciaFeatures) -> pd.DataFrame:
    dados = pd.DataFrame([features.model_dump()], columns=FEATURE_COLUMNS)

    int32_columns = [
        "Ano_da_Ocorrencia",
        "Mes_da_Ocorrencia",
        "Dia_Semana_da_Ocorrencia",
    ]
    int64_columns = [
        "PMD_ausente",
        "PMD_zero",
        "Numero_de_Assentos_ausente",
        "Numero_de_Assentos_zero",
        "ICAO_ausente",
        "Tipo_de_Aerodromo_ausente",
        "UF_indeterminada",
    ]
    float_columns = ["PMD", "Numero_de_Assentos"]

    for column in int32_columns:
        dados[column] = dados[column].astype("int32")
    for column in int64_columns:
        dados[column] = dados[column].astype("int64")
    for column in float_columns:
        dados[column] = dados[column].astype("float64")

    # O modelo registrado inferiu esta coluna como double porque o exemplo de
    # entrada tinha apenas valores ausentes apos a limpeza das categoricas.
    dados["Tipo_de_Aerodromo"] = pd.to_numeric(
        dados["Tipo_de_Aerodromo"], errors="coerce"
    ).replace({None: np.nan})

    return dados


@app.get("/saude", response_model=SaudeResponse)
def saude() -> SaudeResponse:
    _modelo_carregado()
    return SaudeResponse(ok=True, modelo=MODEL_URI)


@app.get("/docs", include_in_schema=False)
def docs():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_ui_parameters={"docExpansion": "full"},
    )


@app.post("/predict", response_model=PredicaoResponse)
def predict(features: OcorrenciaFeatures) -> PredicaoResponse:
    modelo = _modelo_carregado()
    dados = _montar_dataframe(features)
    try:
        predicao = modelo.predict(dados)[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao executar a predicao com o modelo {MODEL_URI}: {exc}",
        ) from exc
    return PredicaoResponse(predicao=str(predicao), modelo=MODEL_URI)
