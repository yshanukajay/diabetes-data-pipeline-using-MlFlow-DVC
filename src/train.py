import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
from mlflow.models.signature import infer_signature
import logging
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)

os.environ["MLFLOW_TRACKING_URL"]="https://dagshub.com/YohanJay23/diabetes-data-pipeline-using-MlFlow-DVC.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="YohanJay23"
os.environ["MLFLOW_TRACKING_PASSWORD"]="12aa8bae310f0795c7c23dcd43e1bb74fec38ea1"

def hyperparameter_tuning(x_train: pd.DataFrame, y_train: pd.Series, param_grid: dict) -> GridSearchCV:

    """
    Perform hyperparameter tuning using GridSearchCV to find the best parameters for the RandomForestClassifier.

    Args:
        x_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training labels.
    """

    logging.info("Starting hyperparameter tuning using GridSearchCV...")

    try:
        rf = RandomForestClassifier()
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(x_train, y_train)
        #best_params = grid_search.best_params_
        logging.info("Successfully completed hyperparameter tuning.")
        return grid_search
    except Exception as e:
        logging.error(f"Error during hyperparameter tuning: {e}")
        raise e
    

params=yaml.safe_load(open("params.yaml"))['train']


def train_model(data_path: str, model_path: str, random_state: int, n_estimators: int, max_depth: int) -> None:

    """
    Train a RandomForestClassifier model on the provided data and save the trained model.

    Args:
        data_path (str): The path to the preprocessed data.
        model_path (str): The path where the trained model will be saved.
        random_state (int): The random state for reproducibility.
        n_estimators (int): The number of trees in the random forest.
        max_depth (int): The maximum depth of the trees in the random forest.
    """
    
    try:
        logger.info(f"Loading preprocessed data from {data_path}")
        data = pd.read_csv(data_path)
        X=data.drop("Outcome", axis=1)
        y=data["Outcome"]
        logger.info("Data loaded successfully")

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URL"])
        
        logger.info("Starting MLflow run...")
        with mlflow.start_run():
            logger.info("Splitting data into training and testing sets...")
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=params["random_state"])
            signature = infer_signature(x_train, y_train)

            params_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
             }

            grid_search = hyperparameter_tuning(x_train, y_train, params_grid)

            best_model = grid_search.best_estimator_
            logger.info("Best model found")

            y_pred = best_model.predict(x_test)

            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model accuracy: {accuracy}")

            #mlflow.log_params(grid_search.best_params_)
            #mlflow.log_metric("accuracy", accuracy)
      
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise e
    
    logger.info("Logging parameters and metrics to MLflow...")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
    mlflow.log_param("best_min_samples_split", grid_search.best_params_['min_samples_split'])
    mlflow.log_param("best_min_samples_leaf", grid_search.best_params_['min_samples_leaf'])
    logger.info("Parameters and metrics logged to MLflow successfully")

    cm= confusion_matrix(y_test, y_pred)
    cr= classification_report(y_test, y_pred)

    mlflow.log_text(str(cm), "confusion_matrix.txt")
    mlflow.log_text(str(cr), "classification_report.txt")
    logger.info("Confusion matrix and classification report logged to MLflow as text files successfully")

    tracking_uri_type=urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_uri_type!="file":
        mlflow.sklearn.log_model(best_model, "model", registered_model_name="BestRandomForestModel")
    else:
        mlflow.sklearn.log_model(best_model, "model", signature=signature)

    logger.info("Parameters and metrics logged to MLflow successfully")

    logger.info(f"Saving the best model to {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    logger.info(f"Model saved successfully to {model_path}")

if __name__=="__main__":
    train_model(
        params['data_dir'],
        params['model_dir'],
        params['random_state'], 
        params['n_estimators'],
        params['max_depth']
    )
