# https://data.cityofnewyork.us/resource/nu7n-tubp.json

import pandas as pd
import json
import re
import os
from sodapy import Socrata
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

key = os.environ["API_KEY"]
username = os.environ["API_USERNAME"]
password = os.environ["API_PASSWORD"]

ENDPOINTS = {
    "citywide-payroll":"k397-673e",
    "nyc-jobs":"kpav-sd4t",
    "civil-service-titles":"nzjr-3966"
        }


class SoPy:
    def __init__(self, 
                key=key,
                username=username,
                password=password):

        self.key=key 
        self.username=username
        self.password=password
        self.client = Socrata("data.cityofnewyork.us", 
                    self.key,
                    self.username,
                    self.password
                    )
    def get(self, endpoint, df=True, **kwargs):
        results = self.client.get(endpoint, **kwargs)
        if not df:
            return results
        else:
            return self.results_df(results)

    def results_df(self, results):
        results_df = pd.DataFrame.from_records(results)
        return results_df

sp = SoPy()

pd.set_option('expand_frame_repr', True)
pd.options.display.max_colwidth = None
res_df = pd.read_csv("tech_jobs_nyc.csv")

#print(res_df.isna().any())
for col in ['minimum_qual_requirements','preferred_skills','additional_information']:
    res_df[col].fillna(value=" ", inplace=True)
#print(res_df.isna().any())

res_df['all_text'] = list(res_df['business_title'] + " " + res_df['civil_service_title'] + " " + res_df['job_description'] + " " + res_df['minimum_qual_requirements'] + " " + res_df['preferred_skills'])
corpus = res_df['all_text'].values.tolist()

import spacy
from html import unescape

# create a spaCy tokenizer
spacy.load('en')
nlp = spacy.lang.en.English()

# remove html entities from docs and
# set everything to lowercase
import unicodedata

def my_preprocessor(doc):
    doc = doc.encode("latin-1").decode("utf-8")
    doc = re.sub(r"[\d!@#$%^&*()\";:~`]", "", doc)
    return unescape(doc).lower()

# tokenize the doc and lemmatize its tokens
def my_tokenizer(doc):
    tokens = nlp(doc)
    return [token.text for token in tokens if not token.is_stop and not token.is_punct]


vect = TfidfVectorizer(encoding='utf-8',
                       #decode_error='strict',
                       strip_accents="unicode",
                       lowercase=True,
                       preprocessor=my_preprocessor, 
                       tokenizer=my_tokenizer,
                       # analyzer='word', 
                       #stop_words="english",
                       token_pattern=r"(?u)\b[A-Za-z-\'\"\s][A-Za-z-\'\"\s]+\b", #[A-Za-z-\'\"\s][A-Za-z-\'\"\s]
                       ngram_range=(1, 2),
                       max_df=0.9,
                       min_df=2,
                       max_features=None,
                       norm='l2',
                       use_idf=True,
                       smooth_idf=True,
                       sublinear_tf=False)

def wm2df(wm, feat_names):
    
    # create an index for each row
    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names,
                      columns=feat_names)
    return(df)

dtm = vect.fit_transform(corpus)
#dense = dtm.toarray().transpose()
feats = vect.get_feature_names()

#wm2df(dtm, feats).to_csv('tfidf_words.csv')

from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(
    #encoding='utf-8',
    #decode_error='strict',
    strip_accents="unicode",
    lowercase=True,
    preprocessor=my_preprocessor, 
    tokenizer=my_tokenizer,
    # analyzer='word', 
    #stop_words="english",
    token_pattern=r"(?u)\b[A-Za-z-\'\"\s][A-Za-z-\'\"\s]+\b", #[A-Za-z-\'\"\s][A-Za-z-\'\"\s]
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=2,
    max_features=None,
    #vocabulary=vect.vocabulary_
)
# Helper function
def print_topics(model, vectorizer, n_top_words):
    words = vectorizer.get_feature_names()
    string = ""
    for topic_idx, topic in enumerate(model.components_):
        string += "\nTopic #%d:" % topic_idx
        string += " ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
        string += "\n"
    return string
        
# Tweak the two parameters below
number_topics = 7
number_words = 10
# Create and fit the LDA model

counts = cv.fit_transform(corpus)
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(counts)
# Print the topics found by the LDA model
string = f"Topics found via LDA: {print_topics(lda, cv, number_words)}"
#print(string)

#with open("sample_topic_model.txt", "w") as f:
#    f.write(string)

#print(pd.DataFrame(dense, index=feats))

from fuzzywuzzy import fuzz 
from fuzzywuzzy import process  
import gensim
#matching = []
#for row in res_df['business_title']:
#    for found, score, matchrow in process.extract(row, res_df['business_title'], limit=1):
#        if 75 < score < 100 :
#            print('%d%% partial match: "%s" with "%s" ' % (score, row, found))
#            m = [row, score, found]
#            m.append(matching)
##print(matching)


import numpy as np

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

lda_output = lda.fit_transform(cv.fit_transform(res_df['minimum_qual_requirements'].tolist()))
# Construct the k-means clusters
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=10, random_state=100).fit_predict(lda_output)

# Build the Singular Value Decomposition(SVD) model
svd_model = TruncatedSVD(n_components=5)  # 2 components
lda_output_svd = svd_model.fit_transform(lda_output)

# X and Y axes of the plot using SVD decomposition
x = lda_output_svd[:, 0]
y = lda_output_svd[:, 1]

# Weights for the 15 columns of lda_output, for each component
print("Component's weights: \n", np.round(svd_model.components_, 2))

# Percentage of total information in 'lda_output' explained by the two components
print("Perc of Variance Explained: \n", np.round(svd_model.explained_variance_ratio_, 2))

plt.figure(figsize=(12, 12))
plt.scatter(x, y, c=clusters)
plt.xlabel('Component 2')
plt.xlabel('Component 1')
plt.title("Segregation of Topic Clusters", )
plt.show()


df_topic_keywords = pd.DataFrame(lda.components_)

f_topic_keywords = pd.DataFrame(lda.components_)

# Assign Column and Index
df_topic_keywords.columns = cv.get_feature_names()
df_topic_keywords.index = ["Topic" + str(i) for i in range(lda.n_components)]

# View
print(df_topic_keywords.head())



from sklearn.metrics.pairwise import euclidean_distances
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

#nlp = spacy.load('en', disable=['parser', 'ner'])
def predict_topic(text, nlp=nlp):
    global sent_to_words
    global lemmatization

    # Step 1: Clean with simple_preprocess
    mytext_2 = list(sent_to_words(text))

    # Step 2: Lemmatize
    mytext_3 = lemmatization(mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Step 3: Vectorize transform
    mytext_4 = cv.transform(mytext_3)

    # Step 4: LDA Transform
    topic_probability_scores = lda.transform(mytext_4)
    topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), :].values.tolist()
    return topic, topic_probability_scores

# Predict the topic
mytext = ["example"]
topic, prob_scores = predict_topic(text = mytext)
print(prob_scores)



def show_topics(vectorizer=cv, lda_model=lda, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=cv, lda_model=lda, n_words=15)       
print(topic_keywords) 

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords

#corpus = [clean_text(doc) for doc in corpus]
# TF-IDF (scikit-learn's default implementation)

#print(pd.DataFrame(vect.fit_transform(corpus).toarray(), columns=vect.get_feature_names()))

__name__ == "__main__"