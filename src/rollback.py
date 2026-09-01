import argparse
import mlflow

REGISTERED_MODEL_NAME = "house_price_model"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


def rollback(version: str):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    # Confirm the version actually exists before pointing production at it
    target = client.get_model_version(REGISTERED_MODEL_NAME, version)

    try:
        current = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "production")
        print(f"Current production: Version {current.version}")
    except Exception:
        print("No production alias currently set.")

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", version)
    print(f"Rolled back: production now points to Version {target.version} (run_id={target.run_id})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roll back the production alias to a specific model version.")
    parser.add_argument("version", help="Model version number to roll back to, e.g. 1")
    args = parser.parse_args()
    rollback(args.version)