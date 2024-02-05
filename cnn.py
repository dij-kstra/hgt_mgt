# cnn.py
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.feature_extraction.text import TfidfVectorizer


def load_cnn_model():
    # Load the random forest model
    model = tf.keras.models.load_model("models/cnn_model.h5")
    return model


def predict_cnn(text):
    # Load the TfidfVectorizer
    vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

    # Transform the input text
    text_tfidf = vectorizer.transform([text]).toarray()

    # Predict using the logistic regression model
    model = load_cnn_model()
    prediction = model.predict(text_tfidf)

    return prediction
