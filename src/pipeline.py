from prefect import flow, task
from train import main as train_model


@task(retries=2, retry_delay_seconds=10)
def run_training():
    train_model()


@flow(name="house-price-mlops-pipeline")
def house_price_pipeline():
    run_training()


if __name__ == "__main__":
    house_price_pipeline()
