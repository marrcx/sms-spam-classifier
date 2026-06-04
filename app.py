import streamlit as st
import joblib
from src.text_cleaning import clean_text

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
encoder = joblib.load("models/encoder.pkl")

language = st.selectbox(
    "Sprache / Language",
    ["Deutsch", "English"]
)

if language == "Deutsch":
    title = "📩 SMS Spam Erkennung"
    description = "Gib eine Nachricht ein und das Modell erkennt, ob sie Spam oder Ham ist."
    input_label = "Nachricht eingeben:"
    button_text = "Analysieren"
    result_text = "Ergebnis"
    confidence_text = "Sicherheit"
    empty_warning = "Bitte gib zuerst eine Nachricht ein."
    spam_label = "Spam-Nachricht"
    ham_label = "Normale Nachricht"
else:
    title = "📩 SMS Spam Classifier"
    description = "Enter a message and the model predicts whether it is spam or ham."
    input_label = "Enter message:"
    button_text = "Analyze"
    result_text = "Result"
    confidence_text = "Confidence"
    empty_warning = "Please enter a message first."
    spam_label = "Spam Message"
    ham_label = "Normal Message"

st.title(title)
st.write(description)

message = st.text_area(input_label, height=120)

if st.button(button_text):
    if message.strip() == "":
        st.warning(empty_warning)
    else:
        cleaned = clean_text(message)
        X = vectorizer.transform([cleaned])

        prediction = model.predict(X)
        probability = model.predict_proba(X)

        label = encoder.inverse_transform(prediction)[0]
        confidence = probability.max() * 100

        st.subheader(result_text)

        if label == "spam":
            st.error(f"🚨 {spam_label}")
        else:
            st.success(f"✅ {ham_label}")

        st.write(f"**{confidence_text}:** {confidence:.2f}%")
        st.progress(confidence / 100)