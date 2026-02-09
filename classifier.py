import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords




def load_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "data", "linguistics_dataset.csv")
    df = pd.read_csv(path)
    return df["text"].tolist(), df["label"].tolist()


def vectorize_texts(texts):
    spanish_stopwords = stopwords.words("spanish")
    vectorizer = TfidfVectorizer(
        stop_words=spanish_stopwords,
        ngram_range=(1,2),
        max_df=0.9,
        min_df=2
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def train_classifier(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

def run_classification_pipeline():
    texts, labels = load_dataset()
    X, vectorizer = vectorize_texts(texts)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    model = train_classifier(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    return model, vectorizer

if __name__ == "__main__":
    run_classification_pipeline()
