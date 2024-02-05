# naive_bayes.py
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def load_naive_bayes_model():
    # Load the naive bayes model
    model = joblib.load('models/naive_bayes_model.joblib')
    return model

def predict_naive_bayes(text):
    # Load the TfidfVectorizer
    vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

    # Transform the input text
    text_tfidf = vectorizer.transform([text])

    # Predict using the logistic regression model
    model = load_naive_bayes_model()
    prediction = model.predict(text_tfidf)

    return prediction
