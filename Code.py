!pip install nest_asyncio

pip install asyncpraw

import asyncpraw
import pandas as pd
import nest_asyncio
import asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure Reddit API credentials
reddit = asyncpraw.Reddit(
    client_id="S7U_hwLdA6G1oAG9LupOtQ",
    client_secret="hyv2gAc1PETc0y-GZ3-vSidUmT7pnA",
    user_agent="WebMining by /u/Web_Mining_660"
)

# Define an asynchronous function to fetch subreddit data
async def fetch_reddit_data(subreddit_name, keyword, limit=100):
    try:
        subreddit = await reddit.subreddit(subreddit_name)
        posts_data = []

        # Check if subreddit exists and is accessible
        if not subreddit:
            print(f"Failed to access subreddit: {subreddit_name}")
            return pd.DataFrame(columns=["title", "selftext", "upvotes", "comments"])

        # Fetch posts based on search term
        async for post in subreddit.search(keyword, limit=limit):
            post_info = {
                "title": post.title,
                "selftext": post.selftext,
                "upvotes": post.score,
                "comments": []
            }

            # Replace more comments if needed
            if post.comments:
                await post.comments.replace_more(limit=0)

                # Extract comments
                async for comment in post.comments.list():
                    post_info["comments"].append(comment.body)

            posts_data.append(post_info)

        # If no data is fetched, return an empty DataFrame
        if not posts_data:
            print(f"No posts found for {subreddit_name} with keyword {keyword}")
            return pd.DataFrame(columns=["title", "selftext", "upvotes", "comments"])

        return pd.DataFrame(posts_data)

    except Exception as e:
        print(f"An error occurred while fetching data from {subreddit_name}: {e}")
        return pd.DataFrame(columns=["title", "selftext", "upvotes", "comments"])  # Return empty DataFrame on error

# Asynchronous wrapper to run the function
async def main():
    try:
        # Fetch data asynchronously
        pcos_data = await fetch_reddit_data("PCOS", "PCOS")
        thyroid_data = await fetch_reddit_data("thyroidhealth", "thyroid")

        # Save data to JSON if fetched
        if not pcos_data.empty:
            pcos_data.to_json("pcos_data.json", orient="records")
            print("PCOS data saved successfully.")
        else:
            print("PCOS data fetch failed.")

        if not thyroid_data.empty:
            thyroid_data.to_json("thyroid_data.json", orient="records")
            print("Thyroid data saved successfully.")
        else:
            print("Thyroid data fetch failed.")
    finally:
        # Ensure the session is properly closed after the task
        await reddit.close()

# Run the main function
await main()

import requests
from bs4 import BeautifulSoup
import json
import pandas as pd

# Function to fetch and parse data from a health website
def fetch_health_data(url, condition):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Example: Find all paragraphs (you may need to adjust this based on the website structure)
        paragraphs = soup.find_all('p')

        # Extract text from the paragraphs and create a summary
        content = " ".join([para.text for para in paragraphs if para.text])

        # Format the extracted content into a dictionary for easy storage
        health_info = {
            "condition": condition,
            "source": url,
            "content": content[:1000]  # Limit to first 1000 characters for brevity
        }

        return health_info
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# Scrape data for PCOS and Thyroid health from different websites
pcos_urls = [
    "https://www.mayoclinic.org/diseases-conditions/polycystic-ovary-syndrome/symptoms-causes/syc-20350497",
    "https://www.healthline.com/health/pcos"
]

thyroid_urls = [
    "https://www.mayoclinic.org/diseases-conditions/hypothyroidism/symptoms-causes/syc-20350284",
    "https://www.webmd.com/women/guide/understanding-thyroid-problems"
]

# Fetch data for PCOS and Thyroid
pcos_data_scraped = []
for url in pcos_urls:
    data = fetch_health_data(url, "PCOS")
    if data:
        pcos_data_scraped.append(data)

thyroid_data_scraped = []
for url in thyroid_urls:
    data = fetch_health_data(url, "Thyroid")
    if data:
        thyroid_data_scraped.append(data)

# Load the existing JSON data into DataFrames
try:
    pcos_df = pd.read_json("pcos_data.json")
    thyroid_df = pd.read_json("thyroid_data.json")
except ValueError:
    # If the JSON files are empty or don't exist, create empty DataFrames
    pcos_df = pd.DataFrame(columns=["title", "selftext", "upvotes", "comments", "sentiment", "emotion", "remedy"])
    thyroid_df = pd.DataFrame(columns=["title", "selftext", "upvotes", "comments", "sentiment", "emotion", "remedy"])

# Convert the scraped data into DataFrames
pcos_scraped_df = pd.DataFrame(pcos_data_scraped)
thyroid_scraped_df = pd.DataFrame(thyroid_data_scraped)

# Append the scraped data to the existing DataFrames
pcos_df = pd.concat([pcos_df, pcos_scraped_df], ignore_index=True)
thyroid_df = pd.concat([thyroid_df, thyroid_scraped_df], ignore_index=True)

# Save the updated DataFrames back to JSON
pcos_df.to_json("cure_pcos_data.json", orient="records")
thyroid_df.to_json("cure_thyroid_data.json", orient="records")

print("PCOS and Thyroid data updated successfully.")

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to perform sentiment analysis using VADER
def analyze_sentiment_vader(text):
    if not isinstance(text, str):
        text = ""  # Convert non-string values to empty strings
    sentiment_score = sia.polarity_scores(text)
    if sentiment_score['compound'] > 0.05:
        return 'positive'
    elif sentiment_score['compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis to the PCOS and Thyroid Data
def add_sentiment_to_df_vader(df):
    df['sentiment'] = df['selftext'].apply(analyze_sentiment_vader)
    return df

# Example: Apply sentiment analysis to the PCOS and Thyroid Data
pcos_df = add_sentiment_to_df_vader(pcos_df)
thyroid_df = add_sentiment_to_df_vader(thyroid_df)

# Save the updated data with sentiment analysis
pcos_df.to_json("pcos_data_with_sentiment.json", orient="records")
thyroid_df.to_json("thyroid_data_with_sentiment.json", orient="records")

print("Sentiment analysis complete with VADER and data saved.")

!pip install transformers torch

from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and check data
pcos_df = pd.read_json("cure_pcos_data.json")
thyroid_df = pd.read_json("cure_thyroid_data.json")

# Ensure 'selftext' column exists and fill missing values
pcos_df['selftext'] = pcos_df.get('selftext', '').fillna('')
thyroid_df['selftext'] = thyroid_df.get('selftext', '').fillna('')

# Initialize emotion detection pipeline
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", truncation=True, max_length=512)

# Define the emotion detection function
def detect_emotion(text):
    if not isinstance(text, str) or text.strip() == "":
        return None  # Return None for invalid inputs
    result = emotion_analyzer(text)
    return result[0]['label'] if result else None

# Apply emotion detection
pcos_df['emotion'] = pcos_df['selftext'].apply(detect_emotion)
thyroid_df['emotion'] = thyroid_df['selftext'].apply(detect_emotion)

# Filter rows with valid emotions
pcos_df_filtered = pcos_df.dropna(subset=['emotion'])
thyroid_df_filtered = thyroid_df.dropna(subset=['emotion'])

# Save the updated dataframes to JSON files
pcos_df_filtered.to_json("pcos_data_with_emotions.json", orient="records")
thyroid_df_filtered.to_json("thyroid_data_with_emotions.json", orient="records")

# Display basic statistics for numerical columns (upvotes, comments count)
print(pcos_df[['upvotes']].describe())
print(thyroid_df[['upvotes']].describe())

# Check if 'selftext' column is present
print(pcos_df.columns)  # Ensure 'selftext' column exists in pcos_df
print(thyroid_df.columns)  # Ensure 'selftext' column exists in thyroid_df

# Make sure there are no null values in 'selftext' before applying emotion detection
pcos_df['selftext'] = pcos_df['selftext'].fillna('')
thyroid_df['selftext'] = thyroid_df['selftext'].fillna('')

# Apply the emotion detection function to 'selftext'
pcos_df['emotion'] = pcos_df['selftext'].apply(detect_emotion)
thyroid_df['emotion'] = thyroid_df['selftext'].apply(detect_emotion)

# Check if the 'emotion' column is added
print(pcos_df[['selftext', 'emotion']].head())  # Check the first few rows of the DataFrame
print(thyroid_df[['selftext', 'emotion']].head())

# Load the updated data with sentiment analysis
pcos_df = pd.read_json("pcos_data_with_sentiment.json")
thyroid_df = pd.read_json("thyroid_data_with_sentiment.json")

# Check if the 'emotion' column exists in the DataFrames
if 'emotion' not in pcos_df.columns:
    print("Warning: 'emotion' column not found in pcos_df. Adding an empty column.")
    pcos_df['emotion'] = ''  # Add an empty 'emotion' column if it doesn't exist

if 'emotion' not in thyroid_df.columns:
    print("Warning: 'emotion' column not found in thyroid_df. Adding an empty column.")
    thyroid_df['emotion'] = ''  # Add an empty 'emotion' column if it doesn't exist

# Summary statistics for sentiment and emotion
print(pcos_df['sentiment'].value_counts())  # Sentiment summary
print(thyroid_df['sentiment'].value_counts())  # Sentiment summary

# Check emotion distribution
print(pcos_df['emotion'].value_counts())  # Emotion summary
print(thyroid_df['emotion'].value_counts())  # Emotion summary

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns

# Load the updated data (before summary statistics)
pcos_df = pd.read_json("pcos_data_with_sentiment.json")
thyroid_df = pd.read_json("thyroid_data_with_sentiment.json")

# Combine the dataframes for analysis
data = pd.concat([pcos_df, thyroid_df], ignore_index=True)

# Ensure 'selftext' column exists and fill missing values
data['selftext'] = data.get('selftext', '').fillna('')
# Text Length Analysis
data['text_length'] = data['selftext'].apply(len)

# Print basic statistics for text length
print("Text Length Statistics:")
print(data['text_length'].describe())

# Visualize text length distribution
plt.figure(figsize=(10, 5))
sns.histplot(data['text_length'], bins=50, kde=True)
plt.title('Distribution of Text Length')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# Word Frequency Analysis
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')  # Download punkt tokenizer if not already downloaded

def analyze_word_frequency(text):
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    fdist = FreqDist(tokens)
    return fdist.most_common(10)  # Get top 10 most frequent words

# Apply word frequency analysis to the 'selftext' column
data['word_frequency'] = data['selftext'].apply(analyze_word_frequency)

# Print word frequency for a few samples
print("\nWord Frequency for a few samples:")
for index in range(5):  # Print for the first 5 rows
    print(f"Row {index}: {data['word_frequency'][index]}")

# Visualize word frequency for the entire dataset (top 20 words)
all_words = []
for row in data['word_frequency']:
    all_words.extend([word for word, freq in row])

fdist_all = FreqDist(all_words)
top_20_words = fdist_all.most_common(20)

plt.figure(figsize=(12, 6))
sns.barplot(x=[word for word, freq in top_20_words], y=[freq for word, freq in top_20_words])
plt.title('Top 20 Most Frequent Words')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Plot histograms for upvotes and comments
plt.figure(figsize=(10, 5))
sns.histplot(pcos_df['upvotes'], kde=True, bins=30, color='blue', label="PCOS Upvotes")
sns.histplot(thyroid_df['upvotes'], kde=True, bins=30, color='green', label="Thyroid Upvotes")
plt.legend()
plt.title("Distribution of Upvotes for PCOS and Thyroid Data")
plt.xlabel('Upvotes')
plt.ylabel('Frequency')
plt.show()

# Plot sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=pcos_df, palette="Set2")
plt.title("Sentiment Distribution for PCOS Data")
plt.show()

sns.countplot(x='sentiment', data=thyroid_df, palette="Set2")
plt.title("Sentiment Distribution for Thyroid Data")
plt.show()

# Plot emotion distribution for PCOS data
plt.figure(figsize=(8, 6))
sns.countplot(x='emotion', data=pcos_df_filtered, palette="Set2", order=pcos_df_filtered['emotion'].value_counts().index)
plt.title("Emotion Distribution for PCOS Data")
plt.xlabel("Emotions")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Plot emotion distribution for Thyroid data
plt.figure(figsize=(8, 6))
sns.countplot(x='emotion', data=thyroid_df_filtered, palette="Set2", order=thyroid_df_filtered['emotion'].value_counts().index)
plt.title("Emotion Distribution for Thyroid Data")
plt.xlabel("Emotions")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Using boxplots to detect outliers
plt.figure(figsize=(8, 6))
sns.boxplot(x=pcos_df['upvotes'], color='blue')
plt.title("Boxplot for Upvotes (PCOS)")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x=thyroid_df['upvotes'], color='green')
plt.title("Boxplot for Upvotes (Thyroid)")
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Replace None values in 'selftext' with empty strings before applying TfidfVectorizer
pcos_df['selftext'] = pcos_df['selftext'].fillna('')
thyroid_df['selftext'] = thyroid_df['selftext'].fillna('')

# Vectorize the text data for PCOS dataset
vectorizer = TfidfVectorizer(stop_words='english')
pcos_vectors = vectorizer.fit_transform(pcos_df['selftext'])

# Vectorize the text data for Thyroid dataset using the same vectorizer
thyroid_vectors = vectorizer.transform(thyroid_df['selftext'])

# Check the shape of the pcos_vectors matrix
print(pcos_vectors.shape)

# Compute cosine similarity between PCOS and Thyroid vectors
cosine_sim = cosine_similarity(pcos_vectors, thyroid_vectors)

# Print the cosine similarity matrix
print(cosine_sim)

# Import necessary libraries
import pandas as pd
from transformers import pipeline

# Load data
pcos_df = pd.read_json("cure_pcos_data.json")
thyroid_df = pd.read_json("cure_thyroid_data.json")
data = pd.concat([pcos_df, thyroid_df], ignore_index=True)

# Fill missing or None values in 'selftext' with an empty string
data['selftext'] = data['selftext'].fillna('')

# Emotion detection pipeline using Hugging Face Transformers
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", truncation=True, max_length=512)

# Function for emotion detection
def detect_emotion(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neutral"
    result = emotion_analyzer(text)
    return result[0]['label']

# Apply emotion detection to the dataset
data['emotion'] = data['selftext'].apply(detect_emotion)

# Predefined remedies based on emotions
emotion_remedies = {
    'joy': "Maintain a positive outlook and continue healthy habits like exercise and social activities.",
    'sadness': "Consider therapy, meditation, and connecting with supportive friends or family.",
    'anger': "Practice deep breathing, mindfulness, and physical activities to release tension.",
    'fear': "Engage in relaxation techniques, talk to a counselor, and avoid stress triggers.",
    'neutral': "Maintain a balanced lifestyle with regular exercise, sleep, and a healthy diet.",
    'disgust': "Try mindfulness techniques and focus on activities that bring comfort and relaxation.",
    'surprise': "Channel your surprise into curiosity and learning new things."
}

# Function to match emotions to remedies
def match_remedy(emotion):
    return emotion_remedies.get(emotion, "No remedy available for this emotion.")

# Apply the remedy matching function
data['remedy'] = data['emotion'].apply(match_remedy) # Now 'data' DataFrame has 'emotion' column

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Prepare features and labels
X = data['selftext']
y = data['emotion']

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X)

# Encode labels
label_mapping = {label: idx for idx, label in enumerate(y.unique())}
y_encoded = y.map(label_mapping)

# Check class distribution
print("Class distribution before SMOTE:")
print(y_encoded.value_counts())

# Set k_neighbors dynamically based on the smallest class size
min_class_size = y_encoded.value_counts().min()
k_neighbors = min(3, min_class_size - 1)  # Ensure k_neighbors is less than the smallest class size

# Balance classes with SMOTE
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_res, y_res = smote.fit_resample(X_tfidf, y_encoded)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Hyperparameter tuning for Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg_param_grid = {
    'penalty': ['l2', 'none'],
    'C': [0.1, 1, 10],
    'solver': ['lbfgs', 'saga']
}
logreg_grid = GridSearchCV(logreg, logreg_param_grid, cv=3, scoring='accuracy')
logreg_grid.fit(X_train, y_train)
logreg_best = logreg_grid.best_estimator_

# Hyperparameter tuning for SVC
svc = SVC()
svc_param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}
svc_grid = GridSearchCV(svc, svc_param_grid, cv=3, scoring='accuracy')
svc_grid.fit(X_train, y_train)
svc_best = svc_grid.best_estimator_

# Hyperparameter tuning for XGBoost
xgb = XGBClassifier(eval_metric='mlogloss')
xgb_param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200]
}
xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=3, scoring='accuracy')
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_

# Evaluate tuned Logistic Regression model
y_pred_logreg = logreg_best.predict(X_test)
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
print("Logistic Regression Accuracy:", logreg_accuracy)
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_logreg))

# Evaluate tuned SVC model
y_pred_svc = svc_best.predict(X_test)
svc_accuracy = accuracy_score(y_test, y_pred_svc)
print("SVC Accuracy:", svc_accuracy)
print("SVC Classification Report:\n", classification_report(y_test, y_pred_svc))

# Evaluate tuned XGBoost model
y_pred_xgb = xgb_best.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Accuracy:", xgb_accuracy)
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

# Save the results to a new JSON file
data.to_json("data_with_emotions_and_remedies.json", orient="records")
pd.read_json("data_with_emotions_and_remedies.json").to_csv("data_with_emotions_and_remedies.csv", index=False)

