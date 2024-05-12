import streamlit as st
import pandas as pd
import joblib

# Function to load the model safely
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load the model. Make sure the model path is correct and the environment is compatible. Error: {e}")
        return None

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

# Load the model
model = load_model('random_forest_hourly_rate_model.joblib')

# Streamlit form for user inputs
st.title('Hourly Rate Prediction App')
if model:
    with st.form("prediction_form"):
        st.header("Enter the details:")
        job_type = st.selectbox('Select Job Type', options=job_type_options)
        ex_level_demand = st.selectbox('Select Experience Level Demand', options=ex_level_demand_options)
        industry = st.selectbox('Select Industry', options=industry_options)
        technical_tool = st.selectbox('Select Technical Tool', options=technical_tool_options)
        competitive_level = st.selectbox('Select Competitive Level', options=competitive_level_options)
        client_country = st.selectbox('Select Client Country', options=client_country_options)
        submitted = st.form_submit_button("Predict Hourly Rate")

    if submitted and model:
        # Prepare the input data
        input_data = pd.DataFrame([[job_type, ex_level_demand, industry, technical_tool, competitive_level, client_country]],
                                  columns=['Job Type', 'EX_level_demand', 'Industry', 'Technical_Tool', 'Competitive Level', 'Client_Country'])
        
        try:
            # Predict
            prediction = model.predict(input_data)
            st.write(f"The predicted hourly rate is ${prediction[0]:.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction. Error: {e}")
