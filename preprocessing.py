import re

def preprocess(text):
    text = text.lower()
    tokens = re.findall(r"[a-záéíóúüñ]+", text)
    return tokens
