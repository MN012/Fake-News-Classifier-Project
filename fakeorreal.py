import pandas as pd
import nltk

df = pd.read_csv('fake_or_real_news.csv')


df_cleaned= df.dropna()
df_cleaned= df_cleaned.reset_index(drop=True)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


df_cleaned["fake"] = df_cleaned["label"].apply(lambda x: 1 if x == "FAKE" else 0)

df_cleaned.drop("label", axis=1, inplace=True)

x = df_cleaned["text"]
y = df_cleaned["fake"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
x_train_vectors = vectorizer.fit_transform(x_train)
x_test_vectors = vectorizer.transform(x_test)

clf = LinearSVC()
clf.fit(x_train_vectors, y_train)
clf.score(x_test_vectors, y_test)

from nltk.tokenize import sent_tokenize

sentence = split_into_sentences("This is a fake news. This is real news.")

def split_into_sentences(sentence):
    return sent_tokenize(sentence)
