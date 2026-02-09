from sklearn.feature_extraction.text import TfidfVectorizer

def top_tfidf_terms(text, top_n=20):
    vectorizer = TfidfVectorizer(
        stop_words='spanish',
        ngram_range=(1,2),
        max_df=0.9,
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = tfidf_matrix.toarray()[0]
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
