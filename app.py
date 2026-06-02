import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

model_cat = pickle.load(open("category_model.pkl", "rb"))
model_pri = pickle.load(open("priority_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.split()
    text = [w for w in text if w not in stop_words]
    return " ".join(text)

st.title("Ticket Classifier System")

text = st.text_area("Enter ticket text")

if st.button("Predict"):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    cat = model_cat.predict(vec)[0]
    pri = model_pri.predict(vec)[0]

    st.success("Category: " + cat)
    st.success("Priority: " + pri)