import os
import sys
from src.utils import save_object,load_object
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
import pandas as pd
import numpy as np
import joblib # import the lib to load / Save the model

# Define the num_columns and cat_columns variables
num_columns = ['votes', 'cost']
cat_columns = ['online_order', 'book_table', 'location', 'rest_type', 'cuisines',]            
all_columns = num_columns+cat_columns

# Create the PredictPipeline class
class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):
        """
        This method loads the preprocessor and the model, scales the data, and predicts the rating for the given data.

        Args:
            features: The data to be predicted.

        Returns:
            The predicted rating.
        """

        try:
            # Load the preprocessor and the model.
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.joblib')  # Save the model as .joblib

            preprocessor = load_object(preprocessor_path)
            model = joblib.load(model_path)  # Use joblib.load to load the model

            # Scale the data.
            scaled_data = preprocessor.transform(features)

            # Predict the rating.
            df = pd.DataFrame(scaled_data, columns=all_columns)
            pred = model.predict(df)
            return pred

        except Exception as e:
            logging.info('Error occurred in Prediction')
            raise CustomException(e, sys)

@dataclass
class ModelTrainerConfig():
    model_trainer_path = os.path.join('artifacts', 'model.joblib')

class CustomData:
    def __init__(self,
                votes:float,
                cost:float,
                online_order:bool,
                book_table:bool,
                location:str,
                rest_type:str,
                cuisines:str
                ):

        self.votes = votes
        self.cost = cost
        self.online_order = online_order
        self.book_table = book_table
        self.location = location
        self.rest_type = rest_type
        self.cuisines = cuisines
   

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict ={
                'votes':[self.votes],
                'cost':[self.cost],
                'online_order':[self.online_order],
                'book_table':[self.book_table],
                'location':[self.location],
                'rest_type':[self.rest_type],
                'cuisines':[self.cuisines]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            # print(df)
            return df

        
        except Exception as e:
            logging.info('Error occured in get data as dataframe')
            raise CustomException(e,sys)



"""
if __name__=='__main__':
    predict_obj = PredictPipeline()
    data  = CustomData(357,3500,False,True,"Richmond Road","Fine Dining","North Indian, Mughlai")
    df = data.get_data_as_dataframe()
    print(df)
    print(predict_obj.predict(df))
"""

