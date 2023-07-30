import os
import pandas as pd
import pickle
import joblib
import sys
from src.logger import logging
from src.exception import CustomException
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer

def check_if_model_is_available():
    artifacts_path = os.path.join("artifacts", 'model.joblib')

    if os.path.exists(artifacts_path):
        return True
    else:
        logging.info("Model is not available, training the model first...")
        # The pipeline is not ready, so run the data ingestion, transformation, and model training code
        logging.info("Pipeline is not ready, running data ingestion, transformation, and model training code...")

        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info("Pipeline is ready, executing predict pipeline...")

        return True

if __name__=="__main__":
    if not check_if_model_is_available():
        exit(1)
