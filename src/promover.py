"""Promove para @production o melhor run ja registrado no MLflow local."""

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


ROOT_DIR = Path(__file__).resolve().parents[1]
TRACKING_URI = (ROOT_DIR / "pre_processamento" / "mlruns").as_uri()
EXPERIMENT_NAME = "desafio01_classificacao_ocorrencia"
MODEL_NAME = "previsor_classificacao_ocorrencia"
METRIC_NAME = "f1_macro_media_cv"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experimento nao encontrado: {EXPERIMENT_NAME}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{METRIC_NAME} > 0",
        order_by=[f"metrics.{METRIC_NAME} DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"Nenhum run com a metrica {METRIC_NAME} foi encontrado.")

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/modelo_completo"
    model_version = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="production",
        version=model_version.version,
    )

    print(
        "Modelo promovido: "
        f"{best_run.data.tags.get('mlflow.runName', best_run.info.run_id)} "
        f"({METRIC_NAME}={best_run.data.metrics[METRIC_NAME]:.4f}) "
        f"-> {MODEL_NAME} v{model_version.version} @production"
    )


if __name__ == "__main__":
    main()
