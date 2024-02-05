# app.py
import streamlit as st
from logistic_regression import predict_logistic_regression
from naive_bayes import predict_naive_bayes
from random_forest import predict_random_forest
from svm import predict_svm
from cnn import predict_cnn

st.set_page_config(page_title="HGT VS MGT")


st.title("Text Classification Application")
st.markdown("Welcome to the Text Classification Application, where you can predict whether a text is Human Generated or Machine Generated. You can select the model of your choice from the sidebar.")

# User input
text_input = st.text_area("Enter your text here:", "")

# Model selection
st.sidebar.header("HGT VS MGT Application")
model_option = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Naive Bayes","Random Forest","SVM","CNN"])

# Initialize prediction variable
prediction = None

# Prediction
if st.button("Predict"):
    if model_option == "Logistic Regression":
        prediction = predict_logistic_regression(text_input)

    elif model_option == "Naive Bayes":
         prediction = predict_naive_bayes(text_input)

    elif model_option == "Random Forest":
         prediction = predict_random_forest(text_input)

    elif model_option == "SVM":
         prediction = predict_svm(text_input)

    elif model_option == "CNN":
         prediction = predict_cnn(text_input)            

    else:
        st.error("Invalid model selection")

# Check if prediction is not None before using it
if prediction is not None:
    prediction_result = "Human Generated text" if prediction[0] == 0 else "Machine Generated Text"
    st.success(f"Prediction: {prediction_result}")
