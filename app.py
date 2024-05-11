import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Attempt to load the model
try:
    model = joblib.load('random_forest_model.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")


# Load the model and preprocessor
model = joblib.load('random_forest_model.joblib')

# Define a function for preprocessing and predicting
def preprocess_and_predict(inputs):
    # Unpack inputs
    job_title, ex_level_demand, description, technical_tool, applicants_num, spent = inputs

    # Prepare the input data for prediction
    input_df = pd.DataFrame({
        'Job Title': [job_title],
        'Description': [description],
        'Technical_Tool': [technical_tool],
        'Client_Country': ['United States'],  # Assume a fixed country for example
        'Applicants_Num': [float(applicants_num)],
        'EX_level_demand': [int(ex_level_demand)],
        'Spent($)': [float(spent)]
    })

    # Predict using the pipeline
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit user interface
def prediction_page():
    st.title("Predict Hourly Rate")

    # Input fields
    job_title = st.selectbox('Job Title', ['Software Developer', 'Data Scientist', 'Project Manager'])
    ex_level_demand = st.selectbox('Experience Level Demand', ['1', '2', '3'])
    description = st.selectbox('Project Description', ['Energy and Utilities', 'Automotive', 'Small Business'])
    technical_tool = st.selectbox('Technical Tool Used', ['Python', 'Excel', 'Tableau'])
    applicants_num = st.selectbox('Number of Applicants', ['2.5', '12.5', '17.5', '35', '75'])
    spent = st.number_input('Budget Spent', min_value=0.0, format="%.2f")

    # Prediction button
    if st.button('Predict Hourly Rate'):
        prediction = preprocess_and_predict([job_title, ex_level_demand, description, technical_tool, applicants_num, spent])
        st.write(f"The predicted hourly rate is ${prediction:.2f}")

def main():
    prediction_page()

if __name__ == '__main__':
    main()
