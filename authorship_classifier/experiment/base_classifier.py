import torch
import torchtext as tt
import os
import numpy as np
from rnn import LSTMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from cltk.stop.greek.stops import STOPS_LIST
from greek_accentuation.characters import strip_accents

class BaseClassifier():
    def __init__(self, batch_size=32, embedding_size=64, hidden_size=32, division="sents", method="plaintext", experiment=f"runs/original", epochs=10, learning_rate=0.0001):
        self.bsz = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.method = method
        self.division = division

        self.TEXT = tt.data.Field(sequential=True, include_lengths=True)
        self.AUTHOR = tt.data.Field(sequential=False, unk_token=None)
        self.WORK = tt.data.Field(sequential=False)

        self.train, self.val, self.test = tt.data.TabularDataset.splits(
                path='../data/',
                format='csv',
                train=f"{self.division}_train.csv",
                validation=f"{self.division}_val.csv", 
                test=f"{self.division}_test.csv",
                fields=[('text', self.TEXT), ('author', self.AUTHOR), ('work', self.WORK)],
                csv_reader_params={'delimiter':'\t'}
                )
        

        self.disputed = tt.data.TabularDataset(
            path=f'../data/{self.division}_spurious.csv',
            format="csv",
            csv_reader_params={'delimiter':'\t'},
            fields=[('text', self.TEXT), ('work', self.WORK)]
        )

        self.TEXT.build_vocab(self.train.text)
        self.AUTHOR.build_vocab(self.train.author)
        self.WORK.build_vocab(self.train.work)


        self.train_iter, self.val_iter, self.test_iter = tt.data.BucketIterator.splits(
            (self.train, self.val, self.test), 
            batch_size=self.bsz, 
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            device=self.device)

        self.spurious_iter = tt.data.BucketIterator(
            self.disputed,
            batch_size=self.bsz,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            device=self.device)

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.experiment = experiment

    def run_bootstrap(self, num_bootstraps):
        train_accs = []
        test_accs = []
        for i in range(num_bootstraps):
            classifier = LSTMClassifier(self.TEXT, self.AUTHOR, embedding_size=self.embedding_size, hidden_size=self.hidden_size, experiment=(self.experiment + f"_{i}")).to(self.device)
            classifier.train_all(self.train_iter, self.val_iter, epochs=self.epochs, learning_rate=self.learning_rate)
            classifier.load_state_dict(classifier.best_model)

            train_acc = classifier.evaluate(self.train_iter)
            test_acc = classifier.evaluate(self.test_iter)

            print(f"Bootstrap {i}\n")
            print(f'Training accuracy: {train_acc:.3f}\n'
                f'Test accuracy:     {test_acc:.3f}\n')
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)

        train_accs = np.array(train_accs)
        test_accs = np.array(test_accs)

        print(f"Original:")
        print(f"Training accuracy has mean {train_accs.mean()} and standard deviation {train_accs.std()}")
        print(f"Test accuracy has mean {test_accs.mean()} and standard deviation {test_accs.std()}")

        with open(f'../results/{self.method}_results.csv', 'w') as results_csv:
            writer = csv.writer(results_csv)
            writer.writerow(["Bootstrap", "Train Accuracy", "Test Accuracy"])
            for i, (train, test) in enumerate(zip(train_accs, test_accs)):
                writer.writerow([i, train, test])

        with open('../results/bootstrap_results.csv', 'a') as results_csv:
            writer = csv.writer(results_csv)
            if self.method == "plaintext":
                writer.writerow(["Model Type", "Train Accuracy Mean", "Train Accuracy Standard Deviation", "Test Accuracy Mean", "Test Accuracy Standard Deviation"])
            writer.writerow([self.method, train_accs.mean(), train_accs.std(), test_accs.mean(), test_accs.std()])