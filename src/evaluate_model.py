import pandas as pd
import joblib
from text_cleaning import clean_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
encoder = joblib.load("models/encoder.pkl")


df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)
df["cleaned"] = df["message"].apply(clean_text)

X = vectorizer.transform(df["cleaned"])
y_true = encoder.transform(df["label"])

y_pred = model.predict(X)

print("Accuracy:", accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=encoder.classes_
))

print("\nConfusion Matrix:")
print("Reihen = echte Labels, Spalten = vorhergesagte Labels")
print(encoder.classes_)
print(confusion_matrix(y_true, y_pred))