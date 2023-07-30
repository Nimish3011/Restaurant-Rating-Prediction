# This code defines a data transformation pipeline for preprocessing and cleaning the data before using it for machine learning.

# Import required libraries
import sys
import os
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from dataclasses import dataclass
from src.utils import save_object

# Define a dataclass to hold configuration information for data transformation
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

# DataTransformation class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation has started")

            # Define which columns to ordinal encode and which items to scale
            categorical_cols = ['online_order', 'book_table', 'location', 'rest_type', 'cuisines']
            numerical_cols = ['votes', 'cost']

            logging.info("Pipeline Initiated")

            # Numerical Pipeline
            '''
            1) Handle Missing values
            2) Scaling
            '''
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),  # Median since there are outliers
                ]
            )

            # Categorical Pipeline
            '''
            1) Handle Missing values
            2) Label Encoding
            3) Scaling
            '''
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder()),
                ]
            )

            # ColumnTransformer to apply different pipelines to different columns
            preprocessor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical_cols),
                ('categorical_pipeline', categorical_pipeline, categorical_cols)
            ], remainder='passthrough')

            return preprocessor

            # This line will never be executed as the function will return above
            logging.info('Data Pipeline has been completed.')

        except Exception as e:
            logging.info("Error occurred in Data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test dataset complete")

            logging.info(f"Read Train Dataset \n {train_df.head().to_string()}")

            logging.info(f"Read Test Dataset \n {test_df.head().to_string()}")

            logging.info('Concatenating Train and Test CSV')

            df = pd.concat([train_df, test_df], axis=0)

            logging.info(f'DataFrame Head: \n {df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            # Rate handling
            df['rate'] = df['rate'].apply(lambda x: float(x.split('/')[0]) if (len(x) > 3) else 0)

            # Cost handling
            df['cost'] = df['cost'].str.replace(',', '').astype(float)

            logging.info(f'DataFrame Head: \n {df.head().to_string()}')

            # Splitting features and target variable
            features = df.drop(['rate'], axis=1)
            target = df['rate']

            logging.info(f'Features DataFrame before transformation: \n {features.head().to_string()}')

            # Get the data transformation object
            preprocessor_obj = self.get_data_transformation_object()

            # Apply the transformation on the features dataset
            features = preprocessor_obj.fit_transform(features)

            num_columns = ['votes', 'cost']
            cat_columns = ['online_order', 'book_table', 'location', 'rest_type', 'cuisines']

            all_columns = num_columns + cat_columns

            # Convert the transformed features array back to a DataFrame
            features = pd.DataFrame(features, columns=all_columns)

            logging.info(f'DataFrame Head: \n {df.head().to_string()}')

            logging.info("Applying preprocessing object on training and testing datasets.")

            # Save the preprocessor object to a file using a utility function
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info('All sorts of transformation have been done.')

            # Return the preprocessed features, target, and the file path of the preprocessor object
            return features, target, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.info('Error occurred in initiate Data Transformation')
            raise CustomException(e, sys)

# The code is now commented. This script defines a class "DataTransformation" that performs data preprocessing for
# a machine learning model. The class contains two methods, "get_data_transformation_object" and
# "initiate_data_transformation", which perform the actual data preprocessing steps. The script reads data from
# CSV files, applies various data transformation steps such as handling missing values, scaling, and label encoding
# for categorical columns. The preprocessed data is then returned along with the file path to save the preprocessor
# object for future use.
