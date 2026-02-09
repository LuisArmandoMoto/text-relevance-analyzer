from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_keywords(text, keywords):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([text])
    vocab = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()[0]
    result = {}
    for kw in keywords:
        if kw.lower() in vocab:
            idx = list(vocab).index(kw.lower())
            result[kw] = scores[idx]
        else:
            result[kw] = 0.0
    return result, sum(result.values())

def top_tfidf_terms(text, top_n=15):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([text])
    vocab = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()[0]
    pairs = list(zip(vocab, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_n]
