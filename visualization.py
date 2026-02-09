import matplotlib.pyplot as plt
import numpy as np

def plot_class_probabilities(model, vectorizer, text):
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    labels = model.classes_

    indices = np.argsort(probs)[::-1]
    probs = probs[indices]
    labels = labels[indices]

    plt.figure(figsize=(10,5))
    plt.bar(labels, probs)
    plt.ylabel("Probability")
    plt.xlabel("Linguistic area")
    plt.title("Linguistic classification profile of the text")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
