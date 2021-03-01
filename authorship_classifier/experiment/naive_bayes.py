from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from cltk.stop.greek.stops import STOPS_LIST
from greek_accentuation.characters import strip_accents

import pandas as pd
import numpy as np


class NaiveBayes():
    def __init__(self, division="sents", ngram=1):

        self.df_train = pd.read_csv(f"../data/{division}_train.csv", sep='\t', names=['sentence', 'author', 'work'])
        self.df_val = pd.read_csv(f"../data/{division}_val.csv", sep='\t', names=['sentence', 'author', 'work'])
        self.df_test = pd.read_csv(f"../data/{division}_test.csv", sep='\t', names=['sentence', 'author', 'work'])

        self.df_spurious = pd.read_csv(f"../data/{division}_spurious.csv", sep='\t', names=['sentence', 'work'])
        self.df_epistles = self.df_spurious[self.df_spurious['work'] == 36]
        self.df_spurious = self.df_spurious[self.df_spurious['work'] != 36]

        self.tfidf = TfidfVectorizer(lowercase=False, stop_words=list(map(strip_accents, STOPS_LIST)), ngram_range=(1, ngram))
        
        self.tfidf_train = self.tfidf.fit_transform(self.df_train['sentence'])
        self.tfidf_val = self.tfidf.transform(self.df_val['sentence'])
        self.tfidf_test = self.tfidf.transform(self.df_test['sentence'])
        self.tfidf_spurious = self.tfidf.transform(self.df_spurious['sentence'])
        self.tfidf_epistles = self.tfidf.transform(self.df_epistles['sentence'])



        self.label = LabelEncoder()
        self.author_train = self.label.fit_transform(self.df_train['author'])

        self.author_val = self.label.transform(self.df_val['author'])
        self.author_test = self.label.transform(self.df_test['author'])


        self.nb = ComplementNB()
        self.nb.fit(self.tfidf_train, self.author_train)


    def eval(self):
        author_train_pred = self.nb.predict(self.tfidf_train)
        author_val_pred = self.nb.predict(self.tfidf_val)
        author_test_pred = self.nb.predict(self.tfidf_test)

        print(classification_report(self.author_train, author_train_pred))
        print(classification_report(self.author_val, author_val_pred))

    def predict(self):
        epistles_labels = self.label.inverse_transform(self.nb.predict(self.tfidf_epistles))
        print((epistles_labels == "Plato").mean())
        print(epistles_labels)

        spurious_labels = self.label.inverse_transform(self.nb.predict(self.tfidf_spurious))
        print((spurious_labels == "Plato").mean())
        






    