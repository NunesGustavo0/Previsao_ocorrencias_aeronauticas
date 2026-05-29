# Realizando imports
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = Path(__file__).resolve().parent


def limpar_categoricas(df, colunas):
    for coluna in colunas:
        df[coluna] = df[coluna].map(
            lambda valor: np.nan
            if pd.isna(valor) or str(valor).strip() in {"", "-"}
            else str(valor).strip()
        )
    return df


def preparar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv)

    df["Data_da_Ocorrencia"] = pd.to_datetime(df["Data_da_Ocorrencia"], errors="coerce")
    df["Ano_da_Ocorrencia"] = df["Data_da_Ocorrencia"].dt.year
    df["Mes_da_Ocorrencia"] = df["Data_da_Ocorrencia"].dt.month
    df["Dia_Semana_da_Ocorrencia"] = df["Data_da_Ocorrencia"].dt.dayofweek

    for coluna in ["PMD", "Numero_de_Assentos"]:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
        df[f"{coluna}_ausente"] = df[coluna].isna().astype(int)
        df[f"{coluna}_zero"] = (df[coluna] == 0).astype(int)
        df[coluna] = df[coluna].clip(lower=0, upper=df[coluna].quantile(0.99))

    df["PMD_faixa"] = pd.cut(
        df["PMD"],
        bins=[-np.inf, 750, 2250, 5700, 27000, np.inf],
        labels=["ultraleve", "leve", "medio", "grande", "pesado"],
    )

    log_numeric_features = ["PMD", "Numero_de_Assentos"]
    plain_numeric_features = [
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
    ]
    categorical_features = [
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

    df = limpar_categoricas(df, categorical_features)
    df["ICAO_ausente"] = df["ICAO"].isna().astype(int)
    df["Tipo_de_Aerodromo_ausente"] = df["Tipo_de_Aerodromo"].isna().astype(int)
    df["UF_indeterminada"] = (df["UF"] == "Indeterminado").astype(int)

    features = log_numeric_features + plain_numeric_features + categorical_features
    X = df[features]
    y = df["Classificacao_da_Ocorrencia"]

    return X, y, log_numeric_features, plain_numeric_features, categorical_features


def criar_preprocessor(log_numeric_features, plain_numeric_features, categorical_features):
    log_num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p)),
            ("scaler", StandardScaler()),
        ]
    )

    plain_num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Desconhecido")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("log_num", log_num_transformer, log_numeric_features),
            ("plain_num", plain_num_transformer, plain_numeric_features),
            ("cat", cat_transformer, categorical_features),
        ]
    )


def criar_matriz_confusao(y_true, y_pred, nome_modelo):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        ax=ax,
        cmap="Blues",
        values_format="d",
    )
    ax.set_title(f"Matriz de confusão - {nome_modelo}")
    fig.tight_layout()

    caminho = BASE_DIR / f"confusion_matrix_{nome_modelo}.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)

    return caminho


X, y, log_numeric_features, plain_numeric_features, categorical_features = preparar_dados(
    BASE_DIR / "ocorrencia_pre_tratado.csv"
)

preprocessor = criar_preprocessor(
    log_numeric_features,
    plain_numeric_features,
    categorical_features,
)

sample_weights = compute_sample_weight(class_weight="balanced", y=y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

modelos = {
    "LogisticRegression": LogisticRegression(max_iter=2000, C=0.5),
    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        min_samples_leaf=10,
        subsample=0.85,
        random_state=42,
    ),
}

mlflow.set_tracking_uri(str(BASE_DIR / "mlruns"))
mlflow.set_experiment("desafio01_classificacao_ocorrencia")

resultados = []

for nome_modelo, modelo in modelos.items():
    with mlflow.start_run(run_name=nome_modelo):
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", modelo),
            ]
        )

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="f1_macro",
            params={"classifier__sample_weight": sample_weights},
            return_train_score=False,
        )
        y_pred_cv = cross_val_predict(
            pipeline,
            X,
            y,
            cv=cv,
            params={"classifier__sample_weight": sample_weights},
        )

        f1_macro_media = scores["test_score"].mean()
        f1_macro_std = scores["test_score"].std()
        report = classification_report(y, y_pred_cv, output_dict=True, zero_division=0)

        mlflow.log_param("modelo", nome_modelo)
        mlflow.log_param("features", ", ".join(X.columns))
        mlflow.log_param("cv_n_splits", cv.n_splits)
        mlflow.log_param("cv_shuffle", cv.shuffle)
        mlflow.log_param("cv_random_state", cv.random_state)
        mlflow.log_params(
            {f"classifier__{param}": str(valor) for param, valor in modelo.get_params().items()}
        )

        mlflow.log_metric("f1_macro_media_cv", f1_macro_media)
        mlflow.log_metric("f1_macro_std_cv", f1_macro_std)
        mlflow.log_metric("f1_acidente", report["Acidente"]["f1-score"])
        mlflow.log_metric("precision_acidente", report["Acidente"]["precision"])
        mlflow.log_metric("recall_acidente", report["Acidente"]["recall"])
        mlflow.log_metric("f1_incidente_grave", report["Incidente Grave"]["f1-score"])
        mlflow.log_metric("precision_incidente_grave", report["Incidente Grave"]["precision"])
        mlflow.log_metric("recall_incidente_grave", report["Incidente Grave"]["recall"])

        matriz_confusao = criar_matriz_confusao(y, y_pred_cv, nome_modelo)
        mlflow.log_artifact(matriz_confusao, artifact_path="graficos")

        pipeline.fit(X, y, classifier__sample_weight=sample_weights)
        input_example = X.head(5)
        signature = infer_signature(input_example, pipeline.predict(input_example))

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="modelo_completo",
            input_example=input_example,
            signature=signature,
        )

        resultados.append(
            {
                "modelo": nome_modelo,
                "run_id": mlflow.active_run().info.run_id,
                "f1_macro_media_cv": f1_macro_media,
            }
        )

        print(
            f"{nome_modelo} - F1-macro CV: "
            f"{f1_macro_media:.4f} (+/- {f1_macro_std:.4f})"
        )

melhor_resultado = max(resultados, key=lambda resultado: resultado["f1_macro_media_cv"])
nome_modelo_registrado = "previsor_classificacao_ocorrencia"
modelo_uri = f"runs:/{melhor_resultado['run_id']}/modelo_completo"

model_version = mlflow.register_model(
    model_uri=modelo_uri,
    name=nome_modelo_registrado,
)

client = MlflowClient()
client.set_registered_model_alias(
    name=nome_modelo_registrado,
    alias="production",
    version=model_version.version,
)

print(
    f"Modelo vencedor registrado: {melhor_resultado['modelo']} "
    f"-> {nome_modelo_registrado} v{model_version.version} @production"
)
