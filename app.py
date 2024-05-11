import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title('Hourly Rate Prediction')

# Check and display current directory and files
st.write('Current directory:', os.getcwd())
st.write('Directory contents:', os.listdir())

# Function to load the trained model
def load_model():
    try:
        model = joblib.load('random_forest_model.joblib')
        st.write("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Function to load data
def load_data():
    try:
        data = pd.read_csv('combined_dataset1-1300.csv')
        st.write("Data loaded successfully.")
        return data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# Load model and data
model = load_model()
data = load_data()

# Proceed only if data and model are loaded successfully
if data is not None and model is not None:
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
        try:
            input_data = pd.DataFrame({
                'Job Title': [job_title],
                'Description': [description],
                'Technical_Tool': [technical_tool],
                'Client_Country': [client_country],
                'Applicants_Num': [applicants_num],
                'EX_level_demand': [1 if ex_level_demand == 'Entry Level' else 2 if ex_level_demand == 'Intermediate' else 3],
                'Spent($)': [spent]
            })
            prediction = model.predict(input_data)
            st.write(f'The predicted hourly rate is ${prediction[0]:.2f}')
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.error("Required data or model is not available.")
