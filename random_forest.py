# random_forest.py
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def load_random_forest_model():
    # Load the random forest model
    with open('models/random_forest_model.pkl', 'rb') as file:
           model = pickle.load(file)
    return model


def predict_random_forest(text):
    # Load the TfidfVectorizer
    vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

    # Transform the input text
    text_tfidf = vectorizer.transform([text])

    # Predict using the logistic regression model
    model = load_random_forest_model()
    prediction = model.predict(text_tfidf)

    return prediction
