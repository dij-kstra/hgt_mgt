# svm.py
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

def load_svm_model():
    # Load the svm model
    model = joblib.load('models/svm_model.joblib')
    return model

def predict_svm(text):
    # Load the TfidfVectorizer
    vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

    # Transform the input text
    text_tfidf = vectorizer.transform([text])

    # Predict using the logistic regression model
    model = load_svm_model()
    prediction = model.predict(text_tfidf)

    return prediction
