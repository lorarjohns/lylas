from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pandas as pd
import re

with open('all_text.txt') as f:
    corpus = json.load(f)

vect = TfidfVectorizer()

def clean_text(doc):
    doc = re.sub(r"[0-9]", "", doc)
    return doc

corpus = [clean_text(doc) for doc in corpus]
# TF-IDF (scikit-learn's default implementation)
#for doc in corpus:
#    try:
#        vect.fit(doc)
#    except Exception as e:
#        print(corpus.index(doc), e)

pd.DataFrame(vect.fit_transform(corpus).toarray(), columns=vect.get_feature_names())


