"""Kubeflow pipeline: ingest -> feature engineering -> train/evaluate."""

from kfp import compiler, dsl

IMAGE = "docker.io/l232525/btc-forecast-api:latest"


@dsl.container_component
def ingest_data(days: int, raw_data: dsl.Output[dsl.Dataset]):
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "src.etl_fetch"],
        args=["--output", raw_data.path, "--days", days],
    )


@dsl.container_component
def build_features(
    raw_data: dsl.Input[dsl.Dataset], features: dsl.Output[dsl.Dataset]
):
    code = (
        "import sys; from src.train_forecast import preprocess; "
        "preprocess(sys.argv[1], sys.argv[2])"
    )
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-c"],
        args=[code, raw_data.path, features.path],
    )


@dsl.container_component
def train_model(
    features: dsl.Input[dsl.Dataset],
    max_mape: float,
    model: dsl.Output[dsl.Model],
):
    code = (
        "import sys; from src.train_forecast import train_and_evaluate; "
        "train_and_evaluate(sys.argv[1], sys.argv[2], float(sys.argv[3]))"
    )
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-c"],
        args=[code, features.path, model.path, max_mape],
    )


@dsl.pipeline(
    name="btc-forecast-pipeline",
    description="Ingest BTC-USD data, build features, train and evaluate",
)
def btc_pipeline(days: int = 730, max_mape: float = 5.0):
    ingest = ingest_data(days=days)
    feats = build_features(raw_data=ingest.outputs["raw_data"])
    train_model(features=feats.outputs["features"], max_mape=max_mape)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=btc_pipeline,
        package_path="kubeflow/btc_kfp_pipeline.yaml",
    )
    print("compiled -> kubeflow/btc_kfp_pipeline.yaml")
