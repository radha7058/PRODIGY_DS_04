import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load dataset
df = pd.read_csv("twitter_training.csv")

# Rename columns (important)
df.columns = ["id", "topic", "sentiment", "text"]

# Remove null values
df = df.dropna()

# Function for sentiment
def get_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment
df["Predicted_Sentiment"] = df["text"].apply(get_sentiment)

# Count sentiments
sentiment_counts = df["Predicted_Sentiment"].value_counts()

print(sentiment_counts)

# Plot
plt.figure(figsize=(6,4))
sentiment_counts.plot(kind="bar")

plt.title("Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.savefig("sentiment_chart.png")
plt.show()