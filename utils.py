import pandas as pd 
import streamlit as st
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def display_quick_analyzer(df): 

    with st.expander("Click to show results: "):
        #data shape
        st.write("Data Shape: ")
        st.write(df.shape) 

        #data columns 
        st.write("Data Columns: ") 
        st.write(df.columns)

        #descriptive statistics 
        st.write("Descriptive Statistics: ")
        st.write(df.describe())

def check_for_nulls(df): 
    null_columns = df.columns[df.isnull().any()].tolist()
    with st.expander("Click to show results: "):
        if not null_columns: 
            st.write("There are no nulls in the dataset")
        else:
            st.write(f"Columns with null values: {', '.join(null_columns)}.")

def generate_boxplot(df,column):
    with st.expander("Click to show results: "):
        plt.figure(figsize=(6,6))
        sns.boxplot(data=df, x=column)
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.ylabel('Values')
        st.pyplot()

def split_data(df, test_size):
    X = df.drop(columns=['Gross Cigarette Tax Revenue ($)'])
    y = df['Gross Cigarette Tax Revenue ($)']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state = 42, shuffle = True)

    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test, scale_feature):
    if scale_feature=="Yes": 
        ss = StandardScaler() 
        X_train_ss = ss.fit_transform(X_train) 
        X_test_ss = ss.transform(X_test)
        st.write("Features have been scaled successfully using StandardScaler!")

    else: 
        X_train_ss = X_train 
        X_test_ss = X_test 
        st.write("Features have not been scaled!")

    with st.expander("Click to show some of the results of X_train: "):
        st.write(X_train_ss[:5])

    return X_train_ss , X_test_ss 

def scale_features_for_model_training(X_train, X_test, scale_feature):
    if scale_feature=="Yes": 
        ss = StandardScaler() 
        X_train_ss = ss.fit_transform(X_train) 
        X_test_ss = ss.transform(X_test)

    else: 
        X_train_ss = X_train 
        X_test_ss = X_test 

    return ss, X_train_ss , X_test_ss 


def train_and_evaluate_model(model_name, X_train, X_test, y_train, y_test):
    if model_name == 'Linear Regression':
        model = LinearRegression() 
    elif model_name == 'Lasso Regression': 
        model = Lasso()
    elif model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators = 23, max_depth = 29, max_features = 'log2')
    elif model_name == 'Linear Support Vector Machine':
        model = SVR(kernel='linear')
    elif model_name == 'Kernel Support Vector Machine':
        model = SVR(kernel='rbf') 
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Model: {model_name}")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"R^2: {r2:.2f}")

def train_linear_regression(X_train, y_train):
    model = LinearRegression() 
    model.fit(X_train, y_train)
    return model 

def train_lasso_regression(X_train, y_train):
    model = Lasso() 
    model.fit(X_train, y_train)
    return model 

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators = 23, max_depth = 29, max_features = 'log2')
    model.fit(X_train, y_train)
    return model 

def train_linear_svm(X_train, y_train):
    model = LinearSVR() 
    model.fit(X_train, y_train)
    return model 

def train_kernel_svm(X_train, y_train):
    model = SVR(kernel = 'rbf') 
    model.fit(X_train, y_train)
    return model 

def evaluate_model(model,X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2

def display_feature_importance(model, columns): 
    if isinstance(model, RandomForestRegressor):
        feature_importance = pd.DataFrame({
            'Feature': columns,
            'Importance': model.feature_importances_
        })
    feature_importance.sort_values(by="Importance", ascending = False, inplace = True)
    st.write("Feature Importances: ") 
    st.write(feature_importance)

def predict(model, input_values):
    return model.predict(input_values)