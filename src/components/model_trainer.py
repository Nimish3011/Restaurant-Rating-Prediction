import os
import sys
import joblib
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model
from src.utils import save_object
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor


@dataclass
class ModelTrainerConfig():
    model_trainer_path = os.path.join('artifacts','model.joblib')  # for pkl file use .pkl and for jolib file use .joblib

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,features,target):
        """
        This method defines the dependent and independent features, trains a model using different algorithms,
        selects the best model based on the R2 score, and saves the model in .joblib format.

        Args:
            features: The independent features.
            target: The dependent feature.

        Returns:
            The best model.
        """

        try:
            logging.info('Defining Dependent and Independent features')
            X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.3,random_state=42)

            # Create a dictionary of models and their corresponding algorithms.
            models = {
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet(),
                'RandomForestRegressor': RandomForestRegressor(),
                'RandomForestRegressor_Tuned_1' : RandomForestRegressor(n_estimators=500,random_state=329,min_samples_leaf=.0001),
                'ExtraTreeRegressor': ExtraTreesRegressor(),
                'ExtraTreeRegressor_Tuned' : ExtraTreesRegressor(n_estimators = 100),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'DecisionTreeRegressor_Tuned' : DecisionTreeRegressor(min_samples_leaf=.0001)
            }

            # Evaluate the models using different evaluation metrics.
            model_report = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            # Get the best model score from the dictionary.
            best_model_score = max(sorted(model_report.values()))

            # Get the name of the best model.
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Get the best model.
            best_model = models[best_model_name]

            # Save the best model in .joblib format.
            joblib.dump(best_model, self.model_trainer_config.model_trainer_path)

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)

