import streamlit as st 
import os
import pandas as pd 

from utils import display_quick_analyzer, check_for_nulls, generate_boxplot, split_data, scale_features, train_and_evaluate_model
from utils import scale_features_for_model_training, train_linear_svm, train_lasso_regression, train_random_forest
from utils import train_kernel_svm, train_linear_svm, train_linear_regression, evaluate_model, display_feature_importance, predict

st.title('Gross Tax Revenue Predictor')
st.write("This application predicts the gross tax revenue from six features")
st.set_option('deprecation.showPyplotGlobalUse', False)

uploaded_file = st.file_uploader("Please Upload the required csv file", type = 'csv')

if uploaded_file is not None: 
    df = pd.read_csv(uploaded_file)
    df.drop(["LocationAbbr", "LocationDesc"], axis = 1, inplace = True)


    if st.button("Quick Analyzer"): 
        display_quick_analyzer(df)

    if st.button("Check for Nulls"):
        check_for_nulls(df)

    if uploaded_file is not None: 
        boxplot_column = st.selectbox("Select column for boxplot", df.columns[2:])

    if st.button("Generate Boxplot"):
        generate_boxplot(df, boxplot_column)

    test_size = st.slider("Select test size (%)", min_value = 10, max_value=30, step = 5) 

    if st.button("Split Data"):
        with st.expander("Click to show results: "):
            X_train, X_test, y_train, y_test = split_data(df, test_size/100)
            st.write("Test size", test_size, "%")
            st.write("Training data shape:", X_train.shape)
            st.write("Testing data shape:", X_test.shape)

    scale_feature = st.radio("Would you like to scale features using StandardScaler?", ("Yes","No"))

    if st.button('Scale Features'):
        X_train, X_test, y_train, y_test = split_data(df, test_size/100)
        X_train_ss, X_test_ss = scale_features(X_train, X_test, scale_feature)

    model_name = st.selectbox("Select model to train", ("Linear Regression", 'Lasso Regression', 'Random Forest', 'Linear Support Vector Machine', 'Kernel Support Vector Machine'))

    if st.button("Train and Evaluate Model"):
        X_train, X_test, y_train, y_test = split_data(df, test_size/100)
        _,X_train_ss, X_test_ss = scale_features_for_model_training(X_train, X_test, scale_feature)
        train_and_evaluate_model(model_name, X_train_ss, X_test_ss, y_train, y_test)

    if st.button('Select Best Model'):
        me_scores = {} 
        X_train, X_test, y_train, y_test = split_data(df, test_size/100)
        _,X_train_ss, X_test_ss = scale_features_for_model_training(X_train, X_test, scale_feature)
        models = {"Linear Regression": train_linear_regression,
                    "Lasso Regression": train_lasso_regression,
                    "Random Forest": train_random_forest,
                    "Linear Support Vector Machine": train_linear_svm,
                    "Kernel Support Vector Machine": train_kernel_svm}
        
        for model_name, train_model_func in models.items():
                model = train_model_func(X_train, y_train)
                mse, _, _ = evaluate_model(model, X_test, y_test)
                me_scores[model_name] = mse
            
        best_model = min(me_scores, key=me_scores.get)
        st.write(f"The best model based on MSE is: {best_model}")

    if st.button("Click for Feature Importance"):
        X_train, X_test, y_train, y_test = split_data(df, test_size/100)
        _,X_train_ss, X_test_ss = scale_features_for_model_training(X_train, X_test, scale_feature)
        best_model = train_random_forest(X_train_ss, y_train)
        display_feature_importance(best_model, df.columns[:-1])

    st.header("Enter Feature Values for Prediction")
    input_values = {}
    for feature in df.columns[:-1]: 
        input_values[feature] = st.number_input(f"Enter value for {feature}", step = 0.1)
        
    #Botton to predict Y 
    if st.button("Predict Gross Cigarette Tax Revenue"):
        X_train, X_test, y_train, y_test = split_data(df, test_size/100)
        ss, X_train_ss, X_test_ss = scale_features_for_model_training(X_train, X_test, scale_feature)
        best_model = train_random_forest(X_train_ss, y_train)
        scaled_input_values = ss.transform([list(input_values.values())])
        prediction =  predict(best_model, scaled_input_values)
        answer = int(prediction)
        formatted_number = '{:,}'.format(answer)
        st.write(f"The predicted value of Gross Cigarette Tax Revenue is: {formatted_number} $.")