import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load the trained model
def load_model():
    logging.info("Attempting to load the model...")
    try:
        model = joblib.load('random_forest_model.joblib')
        logging.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        logging.error("Model file not found.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}")
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
            try:
                prediction = model.predict(input_data)
                st.write(f'The predicted hourly rate is ${prediction[0]:.2f}')
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                logging.error(f"Prediction error: {e}")
        else:
            st.error("Model is not loaded. Please check the logs.")
else:
    st.error("Data is not loaded. Please check the logs.")
