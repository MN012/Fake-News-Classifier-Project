# Fake News Classifier Project

## Overview
A Python-based machine learning project for classifying news articles as "FAKE" or "REAL" using Natural Language Processing (NLP) techniques and a Linear Support Vector Classifier.

## Dependencies
- `pandas`
- `nltk`
- `scikit-learn`

## Dataset
The project uses a CSV file named `fake_or_real_news.csv` which should contain at least two columns:
- `text`: the news article content
- `label`: the classification label ("FAKE" or "REAL")

## Project Structure

### 1. Data Loading and Cleaning
```python
import pandas as pd
import nltk

df = pd.read_csv('fake_or_real_news.csv')

# Clean data by removing NaN values
df_cleaned = df.dropna()
df_cleaned = df_cleaned.reset_index(drop=True)
