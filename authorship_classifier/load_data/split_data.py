import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_data():
    sentences = pd.read_csv('sents.csv', sep='\t', names=['sentence', 'author', 'work'])

    sentences_train, sentences_test = train_test_split(sentences, test_size=0.2, stratify=sentences['work'], random_state=300)
    sentences_valid, sentences_test = train_test_split(sentences_test, test_size=0.50, stratify=sentences_test['work'], random_state=300)

    sentences_train.to_csv('sents_train.csv', index=False, sep='\t')
    sentences_valid.to_csv('sents_val.csv', index=False, sep='\t')
    sentences_test.to_csv('sents_test.csv', index=False, sep='\t')

    groups = pd.read_csv('groups.csv', sep='\t', names=['sentence', 'author', 'work'])

    groups_train, groups_test = train_test_split(groups,test_size=0.2, stratify=groups['work'], random_state=300)
    groups_valid, groups_test = train_test_split(groups_test, test_size=0.50, stratify=groups_test['work'], random_state=300)

    groups_train.to_csv('groups_train.csv', index=False, sep='\t')
    groups_valid.to_csv('groups_val.csv', index=False, sep='\t')
    groups_test.to_csv('groups_test.csv', index=False, sep='\t')