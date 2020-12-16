import torch
import torchtext as tt
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import csv

from rnn import LSTMClassifier

class PlaintextClassifier():
    def __init__(self):
        self.BATCH_SIZE = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.TEXT = tt.data.Field(sequential=True, include_lengths=True)
        self.LABEL = tt.data.Field(sequential=False, unk_token=None)

        self.train, self.val, self.test = tt.data.TabularDataset.splits(
                path='../data/',
                format='csv',
                train="greek_corpus_train_lemma.csv",
                validation="greek_corpus_valid_lemma.csv", 
                test="greek_corpus_test_lemma.csv",
                fields=[('text', TEXT), ('label', LABEL)],
                csv_reader_params={'delimiter':'\t'},
                )
        

        self.apology = tt.data.TabularDataset(
            path='../data/greek_corpus_apology_lemma.csv',
            format="csv",
            csv_reader_params={'delimiter':'\t'},
            fields=[('text', TEXT), ('label', LABEL)]
        )

        self.TEXT.build_vocab(train.text)
        self.LABEL.build_vocab(train.label)

        self.train_iter, self.val_iter, self.test_iter = tt.data.BucketIterator.splits(
            (train, val, test), 
            batch_size=BATCH_SIZE, 
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            device=device)

        self.apology_iter = tt.data.BucketIterator(
            apology,
            batch_size=BATCH_SIZE,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            device=device)

train_accs = []
test_accs = []
apology_accs = []
for i in range(10):
    classifier = LSTMClassifier(TEXT, LABEL, embedding_size=64, hidden_size=32, experiment=f"runs/lemma_fixed_{i}").to(device)
    classifier.train_all(train_iter, val_iter, epochs=15, learning_rate=0.0001)
    classifier.load_state_dict(classifier.best_model)

    train_acc = classifier.evaluate(train_iter)
    test_acc = classifier.evaluate(test_iter)
    apology_acc = classifier.evaluate(apology_iter)

    print(f"Bootstrap {i}\n")
    print(f'Training accuracy: {train_acc:.3f}\n'
        f'Test accuracy:     {test_acc:.3f}\n'
        f'Apology accuracy:  {apology_acc:.3f}\n')
    
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    apology_accs.append(apology_acc)

train_accs = np.array(train_accs)
test_accs = np.array(test_accs)
apology_accs = np.array(apology_accs)

print(f"Lemma:")
print(f"Training accuracy has mean {train_accs.mean()} and standard deviation {train_accs.std()}")
print(f"Test accuracy has mean {test_accs.mean()} and standard deviation {test_accs.std()}")
print(f"Apology accuracy has mean {apology_accs.mean()} and standard deviation {apology_accs.std()}")

with open('./results.csv', 'a') as results_csv:
    writer = csv.writer(results_csv)
    writer.writerow(["lemma_fixed", train_accs.mean(), train_accs.std(), test_accs.mean(), test_accs.std(), apology_accs.mean(), apology_accs.std()])