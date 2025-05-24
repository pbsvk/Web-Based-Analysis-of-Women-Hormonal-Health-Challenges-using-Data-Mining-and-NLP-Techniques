# Web-based Analysis of Women’s Hormonal Health Challenges Using NLP & Machine Learning

## Overview

This project presents a web-based solution to analyze real-world emotional experiences shared by women dealing with hormonal health issues, primarily Polycystic Ovary Syndrome (PCOS) and thyroid disorders. Leveraging Natural Language Processing (NLP) and machine learning, the project extracts insights from Reddit discussions and aligns user emotions with medically recommended remedies from trusted health platforms like Mayo Clinic and Healthline.

## Key Features

- **Data Collection**  
  Scraped real user stories and discussions from Reddit subreddits (`r/PCOS`, `r/thyroidhealth`) and extracted medical advice from Mayo Clinic, WebMD, and Healthline using BeautifulSoup and PRAW.

- **Sentiment Analysis**  
  Used the VADER model to classify text as positive, negative, or neutral to understand community tone.

- **Emotion Detection**  
  Applied a transformer-based model (DistilRoBERTa from Hugging Face) to detect fine-grained emotions such as sadness, anger, fear, joy, disgust, and surprise.

- **Remedy Matching Engine**  
  Mapped detected emotions to personalized support recommendations extracted from trusted health sources.

- **Machine Learning Classification**  
  Trained and evaluated ML models (Logistic Regression, SVC, XGBoost) to automate emotion classification:
  - Best model: **Support Vector Classifier (SVC)** with **86% accuracy**
  - Addressed class imbalance using SMOTE
  - Features extracted via TF-IDF

## Dataset Sources

- **Reddit Discussions**  
  - [r/PCOS](https://www.reddit.com/r/PCOS)  
  - [r/thyroidhealth](https://www.reddit.com/r/thyroidhealth)

- **Medical References**  
  - [Mayo Clinic](https://www.mayoclinic.org)  
  - [Healthline](https://www.healthline.com)  
  - [WebMD](https://www.webmd.com)

## Tech Stack

- **Languages**: Python  
- **NLP Libraries**: NLTK (VADER), Hugging Face Transformers  
- **Web Scraping**: PRAW, BeautifulSoup  
- **ML Models**: Logistic Regression, SVC, XGBoost  
- **Vectorization**: TF-IDF  
- **Imbalance Handling**: SMOTE  
- **Visualization**: Matplotlib, Seaborn  
- **Environment**: Jupyter Notebook  

## Insights

- **Dominant Emotions**: Sadness and anger were most common in discussions of PCOS and thyroid health, reflecting emotional frustration and lack of clinical support.  
- **Sentiment Distribution**: Over 45% of posts reflected negative sentiment, highlighting a need for empathetic digital health solutions.

## Future Enhancements

- Real-time emotion-aware chatbot for health support  
- Expand data sources to include forums, blogs, and medical Q&A platforms  
- Integrate deeper LLM-based sentiment understanding (e.g., GPT-based emotion ranking)  
- Build an interactive dashboard for healthcare providers and support groups  
