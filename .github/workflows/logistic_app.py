# logistic_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('logistics.pkl', 'rb'))  # Save your model as 'logistic_model.pkl'

st.title("🚑 Titanic Survival Prediction")
st.markdown("Predict whether a passenger would survive based on their details.")

# User input form
def user_input():
    Pclass = st.selectbox("Passenger Class", [1, 2, 3])
    Sex = st.selectbox("Sex", ['male', 'female'])
    Age = st.slider("Age", 0, 100, 25)
    SibSp = st.slider("Number of siblings/spouses aboard", 0, 8, 0)
    Parch = st.slider("Number of parents/children aboard", 0, 6, 0)
    Fare = st.number_input("Fare", 0.0, 600.0, 32.2)

    # Encode Sex
    Sex = 1 if Sex == 'male' else 0

    data = {
        'Pclass': Pclass,
        'Sex': Sex,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare
    }

    return pd.DataFrame([data])

input_df = user_input()

if st.button("Predict"):
    prediction = model.predict(input_df)
    pred_proba = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.success(f"Prediction: Survived (Probability: {pred_proba[0][1]:.2f})")
    else:
        st.error(f"Prediction: Did Not Survive (Probability: {pred_proba[0][0]:.2f})")