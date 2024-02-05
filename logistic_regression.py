# logistic_regression.py
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def load_logistic_regression_model():
    # Load the logistic regression model
    model = joblib.load('models/logistic_regression_model.joblib')
    return model

def predict_logistic_regression(text):
    # Load the TfidfVectorizer
    vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

    # Transform the input text
    text_tfidf = vectorizer.transform([text])

    # Predict using the logistic regression model
    model = load_logistic_regression_model()
    prediction = model.predict(text_tfidf)

    return prediction
