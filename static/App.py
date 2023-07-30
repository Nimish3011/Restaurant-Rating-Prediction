import streamlit as st
import pandas as pd
import joblib

def app():
    st.title("Zomato Restaurant Rating Prediction")

    # Load the train.csv file
    train_df = pd.read_csv("artifacts/train.csv")

    # Get the unique values from the location column
    unique_locations = train_df["location"].unique()

    # Create a selectbox for the location field
    location = st.selectbox("What is the location of this restaurant?", unique_locations)
    st.write("You selected:", location)

    # Get the unique values from the rest_type column
    unique_rest_types = train_df["rest_type"].unique()

    # Create a selectbox for the rest_type field
    rest_type = st.selectbox("What is the type of restaurant?", unique_rest_types)
    st.write("You selected:", rest_type)

    # Get the unique values from the cuisines column
    unique_cuisines = train_df["cuisines"].unique()

    # Create a selectbox for the cuisines field
    cuisines = st.selectbox("What cuisines does this restaurant serve?", unique_cuisines)
    st.write("You selected:", cuisines)

    # Create a text input for the cost of two field
    cost = st.text_input("What is the estimated cost for two people?")
    st.write("You entered:", cost)

    # Create a text input for the votes field
    votes = st.text_input("How many votes has this restaurant received?")
    st.write("You entered:", votes)

    # Create a selectbox for online_order
    online_order = st.selectbox("Does the restaurant accept online orders?", ("Yes", "No"))
    st.write("You selected:", online_order)

    # Create a selectbox for book_table
    book_table = st.selectbox("Does the restaurant have an option to book a table?", ("Yes", "No"))
    st.write("You selected:", book_table)

    if st.button("Submit"):
        # Load the model
        model = joblib.load("artifacts/model.joblib")

        # Convert 'cost' and 'votes' to numerical data types
        cost = float(cost)
        votes = int(votes)

        # Convert 'online_order' and 'book_table' to boolean values
        online_order = online_order == "Yes"
        book_table = book_table == "Yes"

        features = {
            "location": [location],
            "rest_type": [rest_type],
            "cuisines": [cuisines],
            "cost": [cost],
            "votes": [votes],
            "online_order": [online_order],
            "book_table": [book_table]
        }

        # Predict the rating
        rating = model.predict(features)[0]

        # Display the predicted rating
        st.write("Predicted rating:", rating)

if __name__ == "__main__":
    app()
