import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# LOAD CSV (THIS IS YOUR FIX — correct filename)
df = pd.read_csv("customer_support_tickets.csv")

print(df.head())

# CLEAN TEXT
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.split()
    text = [w for w in text if w not in stop_words]
    return " ".join(text)

df["text"] = (df["Ticket Subject"].fillna("") + " " + df["Ticket Description"].fillna("")).apply(clean_text)

# LABELS
y_category = df["Ticket Type"]
y_priority = df["Ticket Priority"]

# SPLIT
X_train, X_test, y_cat_train, y_cat_test = train_test_split(
    df["text"], y_category, test_size=0.2, random_state=42
)

_, _, y_pri_train, y_pri_test = train_test_split(
    df["text"], y_priority, test_size=0.2, random_state=42
)

# VECTORIZE
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# CATEGORY MODEL
category_model = LogisticRegression(max_iter=1000)
category_model.fit(X_train_vec, y_cat_train)

# PRIORITY MODEL
priority_model = LogisticRegression(max_iter=1000)
priority_model.fit(X_train_vec, y_pri_train)

# EVALUATION
cat_pred = category_model.predict(X_test_vec)
pri_pred = priority_model.predict(X_test_vec)

print("\nCATEGORY MODEL\n")
print(accuracy_score(y_cat_test, cat_pred))
print(classification_report(y_cat_test, cat_pred))

print("\nPRIORITY MODEL\n")
print(accuracy_score(y_pri_test, pri_pred))
print(classification_report(y_pri_test, pri_pred))

# SAVE MODELS
pickle.dump(category_model, open("category_model.pkl", "wb"))
pickle.dump(priority_model, open("priority_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("DONE - MODELS SAVED")