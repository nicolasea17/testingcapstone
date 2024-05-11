import streamlit as st
import pandas as pd
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import os

# List files in the current directory
st.write('Current directory contents:', os.listdir())

# Check if the model file exists
if os.path.exists('random_forest_model.joblib'):
    st.write('Model file found.')
else:
    st.write('Model file not found.')

# Check if the data file exists
if os.path.exists('combined_dataset1-1300.csv'):
    st.write('Data file found.')
else:
    st.write('Data file not found.')


# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model = joblib.load('random_forest_model.joblib')
        logging.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        logging.error("Model file not found.")
        st.error("Model file not found.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}")
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

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
        st.error("Data file not found.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        st.error(f"Failed to load data: {e}")
        return None

data = get_data()

st.title('Hourly Rate Prediction')

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
            st.error(f"Prediction failed: {e}")
            logging.error(f"Prediction error: {e}")
else:
    st.error("Required data or model is not available.")
