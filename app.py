import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model():
    try:
        model = joblib.load('random_forest_model.joblib')
        st.write("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()


st.title('Hourly Rate Prediction')

# Load the dataset
@st.cache(allow_output_mutation=True)
def get_data():
    logging.info("Loading data...")
    try:
        data = pd.read_csv('combined_dataset1-1300.csv')
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error("Data file not found.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        return None

data = get_data()

if data is not None:
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
        if model is not None:
    try:
        # Example input based on your described features
        input_data = pd.DataFrame({
            'Job Title': ['Application Development'],  # example category
            'Description': ['Energy and Utilities'],  # example category
            'Technical_Tool': ['Balsamiq'],  # example category
            'Client_Country': ['Canada'],  # example category
            'Applicants_Num': [5],  # example number
            'EX_level_demand': [2],  # 1 for 'Entry Level', 2 for 'Intermediate', 3 for 'Expert'
            'Spent($)': [1000.0]  # example number
        })
        prediction = model.predict(input_data)
        st.write(f'The predicted hourly rate is ${prediction[0]:.2f}')
    except Exception as e:
        st.error(f"Prediction failed: {e}")

