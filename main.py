from text_loader import load_text
from preprocessing import preprocess
from keyword_scoring import keyword_frequency
from tfidf_scoring import tfidf_keywords, top_tfidf_terms

def run_analysis(file_path, keywords):
    raw_text = load_text(file_path)
    tokens = preprocess(raw_text)
    freq_scores, freq_total = keyword_frequency(tokens, keywords)
    tfidf_scores, tfidf_total = tfidf_keywords(raw_text, keywords)
    return raw_text, freq_scores, freq_total, tfidf_scores, tfidf_total

if __name__ == "__main__":
    path = input("Path to text or PDF: ").strip('"')
    keywords = [k.strip() for k in input("Keywords (comma separated): ").split(",")]

    raw_text, freq_scores, freq_total, tfidf_scores, tfidf_total = run_analysis(path, keywords)

    print("\nFrequency-based relevance:")
    for k, v in freq_scores.items():
        print(f"{k}: {v:.6f}")
    print(f"Global frequency score: {freq_total:.6f}")

    print("\nTF-IDF-based relevance:")
    for k, v in tfidf_scores.items():
        print(f"{k}: {v:.6f}")
    print(f"Global TF-IDF score: {tfidf_total:.6f}")

    print("\nTop TF-IDF terms:")
    for term, score in top_tfidf_terms(raw_text, top_n=15):
        print(f"{term}: {score:.4f}")
