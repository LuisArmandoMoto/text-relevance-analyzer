from collections import Counter

def keyword_frequency(tokens, keywords):
    total = len(tokens)
    counts = Counter(tokens)
    scores = {}
    for kw in keywords:
        scores[kw] = counts.get(kw.lower(), 0) / total if total > 0 else 0
    return scores, sum(scores.values())
