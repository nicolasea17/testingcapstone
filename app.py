import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging

# Load data
@st.cache_resource
def load_data():
    data = pd.read_csv('combined_dataset1-1300.csv')
    # Additional preprocessing can be placed here
    return data

data = load_data()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define GDP data (copy from your training script)
# GDP data and log transformation
gdp_data = {
    'United States': 26949643,
    'China': 17700899,
    'Germany': 4429838,
    'Japan': 4230862,
    'India': 3732224,
    'United Kingdom': 3332059,
    'France': 3049016,
    'Italy': 2186082,
    'Brazil': 2126809,
    'Canada': 2117805,
    'Russia': 1862470,
    'Mexico': 1811468,
    'South Korea': 1709232,
    'Australia': 1687713,
    'Spain': 1582054,
    'Indonesia': 1417387,
    'Turkey': 1154600,
    'Netherlands': 1092748,
    'Saudi Arabia': 1069437,
    'Switzerland': 905684,
    'Poland': 842172,
    'Taiwan': 751930,  # Estimate based on placeholder
    'Belgium': 627511,
    'Argentina': 621833,
    'Sweden': 597110,
    'Ireland': 589569,
    'Norway': 546768,
    'Austria': 526182,
    'Israel': 521688,
    'Thailand': 512193,
    'United Arab Emirates': 509179,
    'Singapore': 497347,
    'Bangladesh': 446349,
    'Philippines': 435675,
    'Vietnam': 433356,
    'Malaysia': 430895,
    'Denmark': 420800,
    'Egypt': 398397,
    'Nigeria': 390002,
    'Hong Kong': 385546,  # Estimate based on placeholder
    'South Africa': 380906,
    'Iran': 366438,
    'Colombia': 363835,
    'Romania': 350414,
    'Chile': 344400,
    'Pakistan': 340636,
    'Czech Republic': 335243,
    'Finland': 305689,
    'Iraq': 297695,
    'Portugal': 276432,
    'Peru': 264636,
    'Kazakhstan': 259292,
    'New Zealand': 249415,
    'Greece': 242385,
    'Qatar': 235500,
    'Algeria': 224107,
    'Hungary': 203829,
    'Ukraine': 173413,  # Estimate based on placeholder
    'Kuwait': 159687,
    'Ethiopia': 155804,
    'Morocco': 147343,
    'Slovakia': 133044,
    'Cuba': 107350,  # Estimate for 2020, no data for 2023
    'Dominican Republic': 120629,
    'Ecuador': 118686,
    'Puerto Rico': 117515,
    'Kenya': 112749,
    'Oman': 108282,
    'Bulgaria': 103099,
    'Guatemala': 102765,
    'Angola': 93796,
    'Venezuela': 92210,  # Estimate, no recent data
    'Uzbekistan': 90392,
    'Luxembourg': 89095,
    'Costa Rica': 85590,
    'Tanzania': 84033,
    'Panama': 82348,
    'Turkmenistan': 81822,  # Estimate, no recent data
    'Croatia': 80185,
    'Ivory Coast': 79430,
    'Lithuania': 79427,
    'Azerbaijan': 77392,
    'Ghana': 76628,
    'Uruguay': 76244,
    'Serbia': 75015,
    'Myanmar': 74861,
    'Sri Lanka': 74404,  # Estimate, no recent data
    'Belarus': 68864,
    'Slovenia': 68394,
    'DR Congo': 67512,
    'Uganda': 52390,
    'Tunisia': 51271,
    'Jordan': 50022,
    'Cameroon': 49262,
    'Bolivia': 46796,
    'Latvia': 46668,
    'Bahrain': 44994,
    'Paraguay': 44142,
    'Estonia': 41799,
    'Nepal': 41339,
    'Libya': 40194,
    'Macau': 38480,  # Estimate based on placeholder
    'Lebanon': 37945,  # Estimate, no recent data
    'El Salvador': 35339,
    'Honduras': 33992,
    'Zimbabwe': 32424,
    'Cyprus': 32032,  # Estimate based on placeholder
    'Papua New Guinea': 31692,
    'Senegal': 31141,
    'Cambodia': 30943,
    'Iceland': 30570,
    'Georgia': 30023,  # Estimate based on placeholder
    'Zambia': 29536,
    'Trinidad and Tobago': 27887,
    'Bosnia and Herzegovina': 26945,
    'Haiti': 25986,
    'Sudan': 25569,
    'Armenia': 24540,
    'Guinea': 23205,
    'Albania': 23032,
    'Mozambique': 21936,
    'Mali': 21309,
    'Yemen': 21045,
    'Burkina Faso': 20785,
    'Botswana': 20756,
    'Malta': 20311,
    'Benin': 19940,
    'Syria': 19719,  # Estimate, no recent data
    'Gabon': 19319,
    'Palestine': 18109,  # Estimate based on placeholder
    'Mongolia': 18782,
    'Jamaica': 18761,
    'Nicaragua': 17353,
    'Niger': 17073,
    'North Korea': 16750,  # Estimate, no recent data
    'Guyana': 16329,
    'Moldova': 16000,  # Estimate based on placeholder
    'North Macedonia': 15801,
    'Madagascar': 15763,
    'Brunei': 15153,
    'Afghanistan': 14939,  # Estimate, no recent data
    'Mauritius': 14819,
    'Congo': 14407,
    'Laos': 14244,
    'Rwanda': 13927,
    'Bahamas': 13876,
    'Malawi': 13176,
    'Kyrgyzstan': 12681,
    'Namibia': 12647,
    'Chad': 12596,
    'Tajikistan': 11816,
    'Somalia': 11515,
    'Kosovo': 10469,
    'Mauritania': 10357,
    'New Caledonia': 10071,  # Estimate, no recent data
    'Equatorial Guinea': 10041,
    'Togo': 9111,
    'Monaco': 8596,  # Estimate, no recent data
    'Bermuda': 7551,  # Estimate for 2022, no data for 2023
    'Montenegro': 7058,
    'Maldives': 6977,
    'Liechtenstein': 6608,  # Estimate, no recent data
    'South Sudan': 6267,
    'Barbados': 6220,
    'French Polynesia': 6055,  # Estimate, no recent data
    'Cayman Islands': 5809,  # Estimate, no recent data
    'Fiji': 5511,
    'Eswatini': 4648,
    'Liberia': 4347,
    'Djibouti': 3873,
    'Aruba': 3827,  # Estimate, no recent data
    'Andorra': 3692,
    'Suriname': 3539,
    'Sierra Leone': 3519,
    'Greenland': 3273,  # Estimate, no recent data
    'Belize': 3218,
    'Burundi': 3190,
    'Central African Republic': 2760,
    'Curaçao': 2700,  # Estimate, no recent data
    'Bhutan': 2686,
    'Cape Verde': 2598,
    'Saint Lucia': 2469,
    'Gambia': 2388,
    'Lesotho': 2373,
    'Eritrea': 2255,  # Estimate, no recent data
    'Seychelles': 2085,
    'Zanzibar': 2080,  # Estimate, no recent data
    'East Timor': 2023,
    'San Marino': 1998,
    'Guinea-Bissau': 1991,
    'Antigua and Barbuda': 1949,
    'Solomon Islands': 1690,
    'Sint Maarten': 1572,  # Estimate for 2022, no data for 2023
    'British Virgin Islands': 1539,  # Estimate, no recent data
    'Comoros': 1364,
    'Grenada': 1306,
    'Vanuatu': 1166,
    'Turks and Caicos Islands': 1139,  # Estimate for 2022, no data for 2023
    'Saint Kitts and Nevis': 1069,
    'Saint Vincent and the Grenadines': 1039,
    'Samoa': 939,
    'Dominica': 697,
    'São Tomé and Príncipe': 674,
    'Tonga': 547,
    'Micronesia': 458,
    'Cook Islands': 328,  # Estimate, no recent data
    'Anguilla': 303,  # Estimate, no recent data
    'Marshall Islands': 277,
    'Palau': 267,
    'Kiribati': 246,
    'Nauru': 150,
    'Montserrat': 72,  # Estimate, no recent data
    'Tuvalu': 63
}

# Load models and preprocessors
@st.cache_resource
def load_models():
    logging.info("Loading models...")
    try:
        model = joblib.load('random_forest_model.joblib')
        kmeans = joblib.load('kmeans_model.joblib')
        preprocessor = joblib.load('preprocessor.joblib')
        logging.info("All models loaded successfully.")
        return model, kmeans, preprocessor
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        return None, None, None

model, kmeans, preprocessor = load_models()

def preprocess_and_predict(job_title, ex_level_demand, description, technical_tool, applicants_num, client_country, spent):
    try:
        # Mapping and transformation logic
        applicants_map = {'Less than 5': 2.5, '10 to 15': 12.5, '15 to 20': 17.5, '20 to 50': 35, '50+': 75}
        ex_level_map = {'Entry Level': 1, 'Intermediate': 2, 'Expert': 3}

        input_data = pd.DataFrame({
            'Job Title': [job_title],
            'Description': [description],
            'Technical_Tool': [technical_tool],
            'Client_Country': [client_country],
            'Applicants_Num': [applicants_map[applicants_num]],
            'EX_level_demand': [ex_level_map[ex_level_demand]],
            'Spent($)': [spent]
        })

        # Adding GDP and GDP_cluster columns
        gdp = np.log(gdp_data.get(client_country, np.nan))
        gdp_cluster = kmeans.predict([[gdp]])[0]
        input_data['GDP'] = gdp
        input_data['GDP_cluster'] = gdp_cluster

        # Process the data through the pipeline
        processed_data = preprocessor.transform(input_data)
        prediction = model.predict(processed_data)
        return prediction
    except Exception as e:
        logging.error(f"Error during preprocessing or prediction: {e}")
        return None

def prediction_page():
    st.title("Customer Tailored Hourly Rate Prediction")
    job_title_options = data['Job Title'].unique().tolist()
    description_options = data['Description'].unique().tolist()
    technical_tool_options = data['Technical_Tool'].unique().tolist()
    applicants_num_options = ['Less than 5', '10 to 15', '15 to 20', '20 to 50', '50+']
    client_country_options = list(gdp_data.keys())

    job_title = st.selectbox('Job Title', job_title_options)
    ex_level_demand = st.selectbox('Experience Level Demand', ['Entry Level', 'Intermediate', 'Expert'])
    description = st.selectbox('Project Description', description_options)
    technical_tool = st.selectbox('Technical Tool Used', technical_tool_options)
    applicants_num = st.selectbox('Number of Applicants', applicants_num_options)
    client_country = st.selectbox('Client Country', client_country_options)
    spent = st.number_input('Budget Spent', format="%.2f")

    if st.button('Predict Hourly Rate'):
        prediction = preprocess_and_predict(job_title, ex_level_demand, description, technical_tool, applicants_num, client_country, spent)
        if prediction is not None:
            st.write(f"The predicted hourly rate is ${prediction[0]:.2f}")
        else:
            st.error("Prediction failed. Please check the logs.")
            
def login_page():
    st.title("Welcome to Incoding's Page")
    col1, col2 = st.columns(2)

    with col1:
        st.image('https://github.com/nicolasea17/Capstone_Project/blob/main/Incoding%20Picture.png?raw=true', width=250)

    with col2:
        st.image('https://github.com/nicolasea17/Capstone_Project/blob/main/OSB%20Picture.png?raw=true', width=250)

    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Sign In'):
        if username == 'admin' and password == '1234':
            st.session_state.login_successful = True
            st.rerun()
        else:
            st.error('Invalid credentials')

def main():
    if 'login_successful' not in st.session_state:
        st.session_state.login_successful = False

    if st.session_state.login_successful:
        prediction_page()
    else:
        login_page()

if __name__ == '__main__':
    main()
