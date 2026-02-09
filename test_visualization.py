from classifier import run_classification_pipeline
from visualization import plot_class_probabilities

sample_text = """
Language on the internet evolves rapidly due to social interaction,
contextual meaning, and pragmatic negotiation between users.
"""

model, vectorizer = run_classification_pipeline()
plot_class_probabilities(model, vectorizer, sample_text)
