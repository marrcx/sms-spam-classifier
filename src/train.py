## Ablauf:
##Daten laden , CHECK
##→ Text bereinigen CHECK (STOPWORDS WERDEN NICHT ENTFERNT)
##→ in Zahlen umwandeln CHECK
##→ Modell trainieren CHECK
##→ Praktisch versuchen irgendwo einzubauen
##→ train.py immer weiter verbessern

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from text_cleaning import clean_text
import joblib

df = pd.read_csv("data/spam.csv", encoding="latin-1", usecols=[0, 1])## Zeilen namen geben
df.columns = ["label", "message"]

## Text bereinigen
df["message"] = df["message"].apply(clean_text)

## Buchstaben in Zahlen umwandeln (Text mit vectorizer und Labels mit dem encoder)
texts = df["message"]
labels = df["label"]

encoder = LabelEncoder()
Y = encoder.fit_transform(labels)

## Trainiert und testet nicht auf selbe daten
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

## ngram range erkennt nicht nur die einzelnen worte wie "free" und "gift" sondern auch "free gift", min_df ignoriert wörter die nur einmal vorkomen, max_df ignoriert wörter die fast überall vorkommen
## TfidfVectorizer ist oft besser als CountVectorizer, weil wichtige Wörter stärker bewertet werden
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

## Modell trainieren , max_iter=1000 gibt genug zeit zum lernen
## class_weight="balanced" hilft, wenn es mehr ham als spam Nachrichten gibt
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

## testen
y_pred = model.predict(X_test)

print("Labels:", encoder.classes_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
))

print("\nConfusion Matrix:")
print("Reihen = echte Labels, Spalten = vorhergesagte Labels")
print(encoder.classes_)
print(confusion_matrix(y_test, y_pred))


## Jetzt wird das trainierte (oberer code) gespeichert mithilfe von joblib
joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(encoder, "models/encoder.pkl")

print("Modell wurde gespeichert.")

