import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from sklearn.preprocessing import FunctionTransformer
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from src.utils import S3_load_data

# Define the columns to drop and rename in the dataset
drop_columns = ['address', 'url', 'name', 'listed_in(city)', 'phone', 'dish_liked', 'reviews_list', 'menu_item', 'listed_in(type)']
rename_columns = {'approx_cost(for two people)': 'cost'}

# Create a dataclass to hold the configuration for Data Ingestion
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

# Create the DataIngestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # The initiate_data_ingestion() method loads the dataset from S3, performs data cleaning and splitting,
    # and saves the train and test datasets to artifacts/train.csv and artifacts/test.csv respectively.
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')
        try:
            # Load the dataset from S3 bucket called 'zomato-data-set'
            df = S3_load_data(bucket_name_='zomato-data-set', object='zomato.csv')
            logging.info('Dataset read as a pandas DataFrame')

            # Save the raw data to artifacts/raw.csv
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            # Create a pipeline to perform data cleaning and transformation
            edit_pipeline = Pipeline(
                steps=[
                    ("drop_columns", FunctionTransformer(lambda x: x.drop(drop_columns, axis=1))),
                    ("rename_columns", FunctionTransformer(lambda x: x.rename(columns=rename_columns))),
                    ("drop_na", FunctionTransformer(lambda x: x.dropna(how='any'))),
                    ("drop_duplicates", FunctionTransformer(lambda x: x.drop_duplicates())),
                    ('replace_into_bool', FunctionTransformer(lambda x: x.replace(('Yes', 'No'), (True, False))))
                ]
            )

            logging.info("Irrelevant columns removed and some columns renamed")

            # Apply the transformation pipeline to the dataset
            df = edit_pipeline.fit_transform(df)

            logging.info('Train-test split')
            # Split the dataset into train and test sets (70% train, 30% test)
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            # Save the train and test datasets to artifacts/train.csv and artifacts/test.csv respectively
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info('Ingestion of data is completed')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.info('Exception occurred at the Data Ingestion stage')
            raise CustomException(e, sys)

# Define the main() function
if __name__ == "__main__":
    obj = DataIngestion()
