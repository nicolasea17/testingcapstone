import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('random_forest_model.joblib')

st.title('Hourly Rate Prediction')

# Dropdowns based on unique values in the dataset
data = pd.read_csv('combined_dataset1-1300.csv')
job_title_options = data['Job Title'].unique()
description_options = data['Description'].unique()
technical_tool_options = data['Technical_Tool'].unique()
client_country_options = data['Client_Country'].unique()

job_title = st.selectbox('Job Title', job_title_options)
description = st.selectbox('Project Description', description_options)
technical_tool = st.selectbox('Technical Tool Used', technical_tool_options)
client_country = st.selectbox('Client Country', client_country_options)
applicants_num = st.number_input('Number of Applicants', min_value=0, value=5)
ex_level_demand = st.selectbox('Experience Level Demand', ['Entry Level', 'Intermediate', 'Expert'], index=1)
spent = st.number_input('Budget Spent ($)', min_value=0.0, value=1000.0, format='%.2f')

if st.button('Predict Hourly Rate'):
    # Create DataFrame for prediction
    input_data = pd.DataFrame({
        'Job Title': [job_title],
        'Description': [description],
        'Technical_Tool': [technical_tool],
        'Client_Country': [client_country],
        'Applicants_Num': [applicants_num],
        'EX_level_demand': [1 if ex_level_demand == 'Entry Level' else 2 if ex_level_demand == 'Intermediate' else 3],
        'Spent($)': [spent]
    })

    # Make prediction
    prediction = model.predict(input_data)
    st.write(f'The predicted hourly rate is ${prediction[0]:.2f}')
