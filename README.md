# 📩 Customer Support Ticket Classification System (NLP)

An end-to-end Machine Learning project that automatically classifies customer support tickets and assigns priority levels based on text input.

This system helps simulate real-world customer support automation by improving response speed, ticket routing, and workload prioritization.

---

## 🚀 Project Overview

Support teams handle large volumes of customer messages daily.  
This project uses Natural Language Processing (NLP) to automatically:

- Classify support tickets into categories
- Assign priority levels based on urgency
- Improve efficiency in handling customer requests

---

## 🔑 Key Features

✔ Text cleaning and preprocessing (NLTK)  
✔ Ticket category classification (Billing, Technical, Refund, etc.)  
✔ Priority prediction (Critical / High / Medium / Low)  
✔ Machine Learning pipeline using Scikit-learn  
✔ TF-IDF feature extraction  
✔ Interactive Streamlit dashboard (optional UI)

---

## 🧠 Tech Stack

- Python  
- Pandas / NumPy  
- Scikit-learn  
- NLTK  
- Streamlit  
- Logistic Regression (ML Model)

---

## 📊 Machine Learning Approach

- Text preprocessing (lowercasing, cleaning, stopword removal)
- Feature extraction using TF-IDF Vectorizer
- Supervised classification models:
  - Ticket Category Model
  - Priority Classification Model

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
streamlit run app.py
