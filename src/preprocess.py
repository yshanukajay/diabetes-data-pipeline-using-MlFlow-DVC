import pandas as pd
import yaml
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

##load params
params = yaml.safe_load(open('params.yaml'))['preprocess']

logger.info("Starting data preprocessing.....")
def preprocess_data(input_path: str, output_path: str) -> None:
    """
    Preprocess the data by handling missing values and saving the cleaned data.

    Args:
        input_path (str): The path to the raw input data.
        output_path (str): The path where the processed data will be saved.
    """
    try:
        logger.info(f"Loading data from {input_path}")
        data = pd.read_csv(input_path)
        logger.info("Data loaded successfully")
    except Exception as e:
        logger.error(f"Error loading data from {input_path}: {e}")
        raise e
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, header=None, index=False)
    logger.info(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data(params["input_dir"], params["output_dir"])