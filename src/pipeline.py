from pathlib import Path

import mlflow
import pandas as pd
from prefect import flow, task
from train import main as train_model

MAIN_DATA_PATH = Path("data/house_prices.csv")
INCOMING_DIR = Path("data/new")
PROCESSED_DIR = INCOMING_DIR / "processed"
REGISTERED_MODEL_NAME = "house_price_model"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


@task
def find_new_files() -> list[Path]:
    if not INCOMING_DIR.exists():
        return []
    return sorted(
        f for f in INCOMING_DIR.glob("*.csv")
        if f.is_file()
    )


@task
def merge_new_data(new_files: list[Path]):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_csv(MAIN_DATA_PATH) if MAIN_DATA_PATH.exists() else pd.DataFrame()

    for f in new_files:
        incoming_df = pd.read_csv(f)
        main_df = pd.concat([main_df, incoming_df], ignore_index=True)
        f.rename(PROCESSED_DIR / f.name)
        print(f"Merged and archived: {f.name}")

    main_df.to_csv(MAIN_DATA_PATH, index=False)
    print(f"Updated {MAIN_DATA_PATH} -- now {len(main_df)} rows.")


@task(retries=2, retry_delay_seconds=10)
def run_training() -> dict:
    return train_model()


@task
def promote_if_better(training_result: dict):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    new_version = training_result["model_version"]
    new_mae = training_result["mae"]

    try:
        current_prod = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "production")
        current_run = mlflow.get_run(current_prod.run_id)
        current_mae = current_run.data.metrics.get("mae")
    except Exception:
        current_prod = None
        current_mae = None

    if current_prod is None:
        print(f"No current production model -- promoting version {new_version} (MAE={new_mae:.4f})")
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", new_version)
        return

    if new_mae < current_mae:
        print(
            f"New version {new_version} (MAE={new_mae:.4f}) beats "
            f"current production version {current_prod.version} (MAE={current_mae:.4f}) -- promoting."
        )
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", new_version)
    else:
        print(
            f"New version {new_version} (MAE={new_mae:.4f}) did not beat "
            f"current production version {current_prod.version} (MAE={current_mae:.4f}) -- keeping current."
        )


@flow(name="house-price-mlops-pipeline")
def house_price_pipeline():
    new_files = find_new_files()

    if not new_files:
        print("No new files in data/new/ -- skipping retrain.")
        return

    print(f"Found {len(new_files)} new file(s) -- merging and retraining.")
    merge_new_data(new_files)
    result = run_training()
    promote_if_better(result)


if __name__ == "__main__":
    house_price_pipeline()