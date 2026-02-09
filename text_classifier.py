import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_classifier(csv_path):
    df = pd.read_csv(csv_path)

    texts = df["text"]
    labels = df["label"]

    vectorizer = TfidfVectorizer(
        stop_words="spanish",
        ngram_range=(1,2),
        max_df=0.9,
        min_df=2
    )

    X = vectorizer.fit_transform(texts)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    report = classification_report(y_test, predictions)
    return model, vectorizer, report
