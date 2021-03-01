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

DOCUMENT_PATH = "./data/authors/"
DOCUMENT_PATH_LEMMA = "./data/authors_lemma/"

method_dict = {
    "plaintext": "Plaintext",
    "lemma": "Lemmatization",
    "lemma_concat": ""
}

class BaseClassifier():
    def __init__(self, batch_size=32, embedding_size=64, hidden_size=32, method="plaintext", experiment=f"runs/original", epochs=10, learning_rate=0.0001, level_tfidf=False):
        self.bsz = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.method = method

        pipeline = None
        if level_tfidf:
            important_words = self.calculate_top_tfidf_words()

            def preprocess(token):
                if token in important_words:
                    return '<imp>'
                return token

            pipeline = tt.data.Pipeline(preprocess)

        self.TEXT = tt.data.Field(sequential=True, include_lengths=True, preprocessing=pipeline)
        self.LABEL = tt.data.Field(sequential=False, unk_token=None)

        

        self.train, self.val, self.test = tt.data.TabularDataset.splits(
                path='./data/',
                format='csv',
                train=f"greek_corpus_train_{self.method}.csv",
                validation=f"greek_corpus_valid_{self.method}.csv", 
                test=f"greek_corpus_test_{self.method}.csv",
                fields=[('text', self.TEXT), ('label', self.LABEL)],
                csv_reader_params={'delimiter':'\t'}
                )
        

        self.apology = tt.data.TabularDataset(
            path=f'./data/greek_corpus_apology_{self.method}.csv',
            format="csv",
            csv_reader_params={'delimiter':'\t'},
            fields=[('text', self.TEXT), ('label', self.LABEL)]
        )

        self.TEXT.build_vocab(self.train.text)
        self.LABEL.build_vocab(self.train.label)

        self.train_iter, self.val_iter, self.test_iter = tt.data.BucketIterator.splits(
            (self.train, self.val, self.test), 
            batch_size=self.bsz, 
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            device=self.device)

        self.apology_iter = tt.data.BucketIterator(
            self.apology,
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

    def calculate_top_tfidf_words(self):
        top_words = set()
        if self.method in ['plaintext', 'lemma_concat']:
            tfidf = TfidfVectorizer(
                input="filename",
                lowercase=False,
                stop_words=list(map(strip_accents, STOPS_LIST)),
                max_df=0.10
            )
        
            scores = tfidf.fit_transform(list(map(lambda x: os.path.join(DOCUMENT_PATH, x), os.listdir(DOCUMENT_PATH))))
            words = tfidf.get_feature_names()

            top_words.update(np.array(words)[(scores > 0.10).toarray().any(axis=0)])
        
        if self.method in ['lemma', 'lemma_concat']:
            tfidf_lemma = TfidfVectorizer(
                input="filename",
                lowercase=False,
                stop_words=list(map(strip_accents, STOPS_LIST)),
                max_df=0.20
            )
        
            scores_lemma = tfidf_lemma.fit_transform(list(map(lambda x: os.path.join(DOCUMENT_PATH_LEMMA, x), os.listdir(DOCUMENT_PATH_LEMMA))))
            words_lemma = tfidf_lemma.get_feature_names()

            top_words.update(np.array(words_lemma)[(scores_lemma > 0.15).toarray().any(axis=0)])

        if self.method in ['encoded_labeled']:
            tfidf = TfidfVectorizer(
                lowercase=False,
                stop_words=list(map(strip_accents, STOPS_LIST)),
                max_df=0.5
            )

            byte_pair = pd.read_csv('./data/greek_corpus_train_encoded_labeled.csv', names=['sentence', 'author'], delimiter='\t')
            plato = byte_pair[byte_pair['author'] == 'plato']['sentence']
            xenophon = byte_pair[byte_pair['author'] == 'xenophon']['sentence']

            scores_encoded = tfidf.fit_transform([' '.join(plato), ' '.join(xenophon)])
            words_encoded = tfidf.get_feature_names()

            top_words.update(np.array(words_encoded)[(X > 0).toarray().any(axis=0)])

            print(filtered)
        
        return list(top_words)

    def run_bootstrap(self, num_bootstraps):
        train_accs = []
        test_accs = []
        apology_accs = []
        for i in range(num_bootstraps):
            classifier = LSTMClassifier(self.TEXT, self.LABEL, embedding_size=self.embedding_size, hidden_size=self.hidden_size, experiment=(self.experiment + f"_{i}")).to(self.device)
            classifier.train_all(self.train_iter, self.val_iter, epochs=self.epochs, learning_rate=self.learning_rate)
            classifier.load_state_dict(classifier.best_model)

            train_acc = classifier.evaluate(self.train_iter)
            test_acc = classifier.evaluate(self.test_iter)
            apology_acc = classifier.evaluate(self.apology_iter)

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

        print(f"Original:")
        print(f"Training accuracy has mean {train_accs.mean()} and standard deviation {train_accs.std()}")
        print(f"Test accuracy has mean {test_accs.mean()} and standard deviation {test_accs.std()}")
        print(f"Apology accuracy has mean {apology_accs.mean()} and standard deviation {apology_accs.std()}")

        with open(f'./results/{self.method}_results.csv', 'w') as results_csv:
            writer = csv.writer(results_csv)
            writer.writerow(["Bootstrap", "Train Accuracy", "Test Accuracy", "Apology Accuracy"])
            for i, (train, test, apology) in enumerate(zip(train_accs, test_accs, apology_accs)):
                writer.writerow([i, train, test, apology])

        with open('./results/bootstrap_results.csv', 'a') as results_csv:
            writer = csv.writer(results_csv)
            if self.method == "plaintext":
                writer.writerow(["Model Type", "Train Accuracy Mean", "Train Accuracy Standard Deviation", "Test Accuracy Mean", "Test Accuracy Standard Deviation", "Apology Accuracy Mean", "Apology Accuracy Standard Deviation"])
            writer.writerow([self.method, train_accs.mean(), train_accs.std(), test_accs.mean(), test_accs.std(), apology_accs.mean(), apology_accs.std()])