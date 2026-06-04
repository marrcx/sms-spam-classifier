import pandas as pd
import joblib
from text_cleaning import clean_text


model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
encoder = joblib.load("models/encoder.pkl")


def predict_spam(message):
    cleaned = clean_text(message)
    X = vectorizer.transform([cleaned])

    prediction = model.predict(X)
    probability = model.predict_proba(X)

    label = encoder.inverse_transform(prediction)[0]
    confidence = probability.max()

    return label, confidence

msg = input("Nachricht eingeben: ")

label, confidence = predict_spam(msg)

print("Ergebnis:", label)
print("Sicherheit:", round(confidence * 100, 2), "%")