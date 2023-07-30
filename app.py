import os
import pandas as pd
import pickle
import joblib
from flask import Flask, render_template, request, jsonify
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer
from src.pipelines.training_pipeline import check_if_model_is_available

application = Flask(__name__)

app = application

# Load the train.csv file
unique_combinations = pd.read_csv("static/assets/unique_combinations.csv") 

# Load the model
# artifacts_path = os.path.join("artifacts", 'model.pkl')
# model = pickle.load(artifacts_path)

# Get unique values from the train_df DataFrame
unique_locations = unique_combinations["location"].unique()
unique_rest_types = unique_combinations["rest_type"].unique()
unique_cuisines = unique_combinations["cuisines"].unique()

@app.route("/")
def index():
    return render_template("index.html",
                           unique_locations=unique_locations,
                           unique_rest_types=unique_rest_types,
                           unique_cuisines=unique_cuisines)

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        # Get the form data
        data = CustomData(
            location=request.form["location"],
            rest_type=request.form["rest_type"],
            cuisines=request.form["cuisines"],
            cost=float(request.form["cost"]),
            votes=int(request.form["votes"]),
            online_order=request.form["online_order"] == "Yes",
            book_table=request.form["book_table"] == "Yes"
        )

        # Creating DataFrame of CustomData
        new_data = data.get_data_as_dataframe()
        # Initializing Predict pipeline
        predict_pipeline = PredictPipeline()
        predicted = predict_pipeline.predict(new_data)
        # Assigning predicted value in the result variable.
        results = round(predicted[0], 2)

        return jsonify({"final_result": f"Predict Rating is: {results}"})


if __name__ == "__main__":
    # Trigger the training pipeline before starting the Flask app
    check_if_model_is_available()
    app.run(host='0.0.0.0', debug=True)