import streamlit as st
import pandas as pd
import joblib
import logging
import sklearn
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model and label encoders safely
def load_resources():
    try:
        model = joblib.load('random_forest_hourly_rate_model.joblib')
        encoders = joblib.load('label_encoders.joblib')
        return model, encoders
    except Exception as e:
        st.error(f"Failed to load resources. Error: {e}")
        return None, None

# Define the unique entries for categorical inputs
job_type_options = ['Application Development', 'Web Scraping', 'Website Development']
ex_level_demand_options = ['Intermediate', 'Entry Level', 'Expert']
industry_options = ['Energy and Utilities', 'Automotive', 'Small Business/Local Business', 
                    'Non-Profit and NGOs', 'Real Estate', 'Retail (Non-E-commerce)', 'E-Commerce',
                    'Telecommunications', 'Manufacturing and Industrial', 'Finance and Banking', 
                    'Media and Entertainment', 'Insurance', 'Healthcare', 'Construction and Engineering',
                    'Personal Blogs/Portfolios', 'Hospitality and Travel', 'Government', 'Education', 
                    'Professional Services', 'Technology and SaaS', 'Technology', 'Retail', 'Finance',
                    'Entertainment', 'Manufacturing', 'Transportation', 'Legal', 'Marketing', 'Hospitality']
technical_tool_options = ['Balsamiq', 'WebHarvy', 'Figma', 'Ruby on Rails', 'Next.js',
                          'OutWit Hub', 'Proto.io', 'DataMiner', 'WordPress', 'Flutter',
                          'Scrapy', 'Laravel', 'Sketch', 'Octoparse', 'UXPin', 'TypeScript',
                          'Visual Studio Code', 'Bootstrap', 'Affinity Designer', 'Selenium',
                          'Django', 'Axure', 'Adobe XD', 'Node.js', 'Express.js', 'Marvel',
                          'Import.io', 'Beautiful Soup', 'Mozenda', 'InVision', 'ParseHub',
                          'Squarespace', 'Angular', 'Vue.js', 'Drupal', 'React.js', 'Python',
                          'Excel', 'MongoDB', 'HTML', 'Git', 'PHP', 'SQL', 'JavaScript',
                          'React', 'AWS', 'C#', 'TensorFlow', 'CSS', 'Pandas', 'Ruby',
                          'Java', 'Firebase']
competitive_level_options = ['Uncompetitive', 'Competitive', 'Extremely Competitive']
client_country_options = ['Canada', 'Indonesia', 'Germany', 'Australia', 'United Arab Emirates',
                          'Switzerland', 'Israel', 'China', 'Netherlands', 'Spain', 'South Korea', 
                          'Japan', 'India', 'Thailand', 'Brazil', 'Russia', 'Austria', 'Mexico', 
                          'Norway', 'Belgium', 'Italy', 'France', 'Argentina', 'United States',
                          'Sweden', 'South Africa', 'United Kingdom', 'Turkey', 'Saudi Arabia', 
                          'Poland', 'Egypt', 'New Zealand', 'Philippines', 'Iceland', 'Albania', 
                          'Pakistan', 'Greece', 'Kuwait', 'Ireland', 'Hong Kong', 'Uruguay', 
                          'Czech Republic', 'Nigeria', 'Dominican Republic', 'Puerto Rico', 
                          'Lebanon', 'Malaysia', 'Kenya', 'Romania', 'Estonia', 'Singapore', 
                          'Costa Rica']

model, label_encoders = load_resources()

# Streamlit form for user inputs
st.title('Hourly Rate Prediction App')
if model and label_encoders:
    with st.form("prediction_form"):
        st.header("Enter the details:")
        job_type = st.selectbox('Select Job Type', label_encoders['Job Type'].classes_)
        ex_level_demand = st.selectbox('Select Experience Level Demand', label_encoders['EX_level_demand'].classes_)
        industry = st.selectbox('Select Industry', label_encoders['Industry'].classes_)
        technical_tool = st.selectbox('Select Technical Tool', label_encoders['Technical_Tool'].classes_)
        competitive_level = st.selectbox('Select Competitive Level', label_encoders['Competitive Level'].classes_)
        client_country = st.selectbox('Select Client Country', label_encoders['Client_Country'].classes_)
        submitted = st.form_submit_button("Predict Hourly Rate")

    if submitted:
        # Encode the inputs using the loaded label encoders
        input_data = pd.DataFrame([[
            label_encoders['Job Type'].transform([job_type])[0],
            label_encoders['EX_level_demand'].transform([ex_level_demand])[0],
            label_encoders['Industry'].transform([industry])[0],
            label_encoders['Technical_Tool'].transform([technical_tool])[0],
            label_encoders['Competitive Level'].transform([competitive_level])[0],
            label_encoders['Client_Country'].transform([client_country])[0]
        ]], columns=['Job Type', 'EX_level_demand', 'Industry', 'Technical_Tool', 'Competitive Level', 'Client_Country'])

        try:
            # Predict
            prediction = model.predict(input_data)
            st.write(f"The predicted hourly rate is ${prediction[0]:.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction. Error: {e}")
