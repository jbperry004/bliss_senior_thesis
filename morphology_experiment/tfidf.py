from sklearn.feature_extraction.text import TfidfVectorizer
from cltk.stop.greek.stops import STOPS_LIST
from greek_accentuation.characters import strip_accents
import os
import numpy as np

tfidf = TfidfVectorizer(
    input="filename",
    lowercase=False,
    stop_words=list(map(strip_accents, STOPS_LIST)),
    max_df=0.85
)

tfidf_lemma = TfidfVectorizer(
    input="filename",
    lowercase=False,
    stop_words=list(map(strip_accents, STOPS_LIST)),
    max_df=0.85
) 

DOCUMENT_PATH = "./data/documents/"
DOCUMENT_PATH_LEMMA = "./data/documents_lemma/"

documents = list(map(lambda x: os.path.join(DOCUMENT_PATH, x), os.listdir(DOCUMENT_PATH)))
documents_lemma = list(map(lambda x: os.path.join(DOCUMENT_PATH_LEMMA, x), os.listdir(DOCUMENT_PATH_LEMMA)))

X =  tfidf.fit_transform(documents)
X_lemma = tfidf_lemma.fit_transform(documents_lemma)

top_five = X.toarray().argsort(axis=1)[:, -10:][::-1]
top_five_lemma = X_lemma.toarray().argsort(axis=1)[:, -10:][::-1]

words = tfidf.get_feature_names()
words_lemma = tfidf_lemma.get_feature_names()


for i, (row1, row2) in enumerate(zip(top_five, top_five_lemma)):
    print(documents[i])

    for x1, x2 in zip(row1, row2):
        print(words[x1], '\t', words_lemma[x2])

    print('\n')