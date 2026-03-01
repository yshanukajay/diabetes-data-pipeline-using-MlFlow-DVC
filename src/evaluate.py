import pandas as pd
import pickle 
from sklearn.metrics import accuracy_score
import yaml
import os
import logging
import mlflow
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)

os.environ["MLFLOW_TRACKING_URL"]="https://dagshub.com/YohanJay23/diabetes-data-pipeline-using-MlFlow-DVC.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="YohanJay23"
os.environ["MLFLOW_TRACKING_PASSWORD"]="12aa8bae310f0795c7c23dcd43e1bb74fec38ea1"

params=yaml.safe_load(open("params.yaml"))['evaluate']

def evaluate_model(model_path: str, data_path: str) -> None:

    """
    Evaluate the trained model on the test data and log the results.

    Args:
        model_path (str): The path to the trained model.
        data_path (str): The path to the preprocessed test data.
    """

    logging.info("Starting model evaluation...")

    try:
        logging.info(f"Loading data from {data_path}")
        df=pd.read_csv(data_path)
        X=df.drop("Outcome",axis=1)
        y=df["Outcome"]

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URL"])

        logging.info(f"Loading model from {model_path}")
        model=pickle.load(open(model_path, "rb"))

        predictions=model.predict(X)
        accuracy=accuracy_score(y, predictions)

        mlflow.log_metric("accuracy", accuracy)
        logging.info(f"Model evaluation completed successfully. Accuracy: {accuracy}")

    except Exception as e:
        logging.error(f"Error loading data from {data_path}: {e}")
        raise e
    
if __name__ == "__main__":
    evaluate_model(params["model_dir"], params["data_dir"])
        