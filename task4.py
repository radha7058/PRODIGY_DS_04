import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("twitter_training.csv", header=None)

# Rename columns
df.columns = ['id', 'topic', 'sentiment', 'text']

# Keep only needed columns
df = df[['sentiment', 'text']]

# Remove missing values
df.dropna(inplace=True)

# Convert sentiment to binary
df = df[df['sentiment'].isin(['Positive', 'Negative'])]
df['sentiment'] = df['sentiment'].map({'Positive': 1, 'Negative': 0})

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

df['text'] = df['text'].apply(clean_text)

# Features & Target
X = df['text']
y = df['sentiment']

# Convert text to numbers (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 📊 Visualization
sentiment_counts = df['sentiment'].value_counts()

plt.bar(['Negative', 'Positive'], sentiment_counts)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# 🔥 Custom Test
msg = ["This product is amazing and I love it"]
msg_clean = [clean_text(m) for m in msg]
msg_vec = vectorizer.transform(msg_clean)

prediction = model.predict(msg_vec)
print("\nCustom Prediction:", "Positive" if prediction[0]==1 else "Negative")