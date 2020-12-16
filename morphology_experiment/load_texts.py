from cltk.corpus.readers import get_corpus_reader
from cltk.corpus.utils.formatter import tlg_plaintext_cleanup, cltk_normalize
from cltk.stem.lemma import LemmaReplacer
from greek_accentuation.characters import strip_accents
from sklearn.feature_extraction.text import TfidfVectorizer
from cltk.tokenize.greek.sentence import SentenceTokenizer
from cltk.tokenize.word import WordTokenizer
import pandas as pd
import os
import csv
import numpy as np
from collections import Counter
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

TRAIN_CORPUS_FILE_NAME = "./data/greek_corpus_train_plaintext.csv"
VALID_CORPUS_FILE_NAME = "./data/greek_corpus_valid_plaintext.csv"
TEST_CORPUS_FILE_NAME = "./data/greek_corpus_test_plaintext.csv"
APOLOGY_CORPUS_FILE_NAME = "./data/greek_corpus_apology_plaintext.csv"

TRAIN_CORPUS_ACCENTS_FILE_NAME = "./data/greek_corpus_accents_train.csv"
VALID_CORPUS_ACCENTS_FILE_NAME = "./data/greek_corpus_accents_valid.csv"
TEST_CORPUS_ACCENTS_FILE_NAME = "./data/greek_corpus_accents_test.csv"
APOLOGY_CORPUS_ACCENTS_FILE_NAME = "./data/greek_corpus_accents_apology.csv"

TRAIN_CORPUS_ENCODED_FILE_NAME = "./data/greek_corpus_train_encoded.txt"
VALID_CORPUS_ENCODED_FILE_NAME = "./data/greek_corpus_valid_encoded.txt"
TEST_CORPUS_ENCODED_FILE_NAME = "./data/greek_corpus_test_encoded.txt"
APOLOGY_CORPUS_ENCODED_FILE_NAME = "./data/greek_corpus_apology_encoded.txt"

TRAIN_CORPUS_ENCODED_LABELED_FILE_NAME = "./data/greek_corpus_train_encoded_labeled.csv"
VALID_CORPUS_ENCODED_LABELED_FILE_NAME = "./data/greek_corpus_valid_encoded_labeled.csv"
TEST_CORPUS_ENCODED_LABELED_FILE_NAME = "./data/greek_corpus_test_encoded_labeled.csv"
APOLOGY_CORPUS_ENCODED_LABELED_FILE_NAME = "./data/greek_corpus_apology_encoded_labeled.csv"

TRAIN_SENTENCES_FILE_NAME = "./data/greek_sentences_training.txt"
VALID_SENTENCES_FILE_NAME = "./data/greek_sentences_valid.txt"
TEST_SENTENCES_FILE_NAME = "./data/greek_sentences_test.txt"
APOLOGY_SENTENCES_FILE_NAME = "./data/greek_sentences_apology.txt"

TRAIN_AUTHORS_FILE_NAME = "./data/greek_authors_training.txt"
VALID_AUTHORS_FILE_NAME = "./data/greek_authors_valid.txt"
TEST_AUTHORS_FILE_NAME = "./data/greek_authors_test.txt"
APOLOGY_AUTHORS_FILE_NAME = "./data/greek_authors_apology.txt"

ENCODING_FILE_NAME = "./data/greek_byte_pair_encoding.txt"

TRAIN_CORPUS_LEMMA_FILE_NAME = "./data/greek_corpus_train_lemma.csv"
VALID_CORPUS_LEMMA_FILE_NAME = "./data/greek_corpus_valid_lemma.csv"
TEST_CORPUS_LEMMA_FILE_NAME = "./data/greek_corpus_test_lemma.csv"
APOLOGY_CORPUS_LEMMA_FILE_NAME = "./data/greek_corpus_apology_lemma.csv"

TRAIN_CORPUS_LEMMA_CONCAT_FILE_NAME = "./data/greek_corpus_train_lemma_concat.csv"
VALID_CORPUS_LEMMA_CONCAT_FILE_NAME = "./data/greek_corpus_valid_lemma_concat.csv"
TEST_CORPUS_LEMMA_CONCAT_FILE_NAME = "./data/greek_corpus_test_lemma_concat.csv"
APOLOGY_CORPUS_LEMMA_CONCAT_FILE_NAME = "./data/greek_corpus_apology_lemma_concat.csv"

DOCUMENT_PATH = "./data/documents/"
DOCUMENT_PATH_LEMMA = "./data/documents_lemma/"

class TextLoader():
    def __init__(self):
        self.sent_tokenizer = SentenceTokenizer()
        self.word_tokenizer = WordTokenizer('greek')
        self.corpus_reader = get_corpus_reader(corpus_name='greek_text_perseus', language='greek')
        self.lemmatizer = LemmaReplacer('greek')
        self.tfidf_vectorizer = TfidfVectorizer(
            input="filename"
        )

    def process_document(self, doc):
        cleaned_sents = []
        for paragraph in doc['text'].values():
            if type(paragraph) != str:
                paragraph = paragraph.values()
            else:
                paragraph = self.sent_tokenizer.tokenize(paragraph)
            for sent in paragraph:
                if type(sent) is dict:
                    for subsent in sent.values():
                        tokenized = self.sent_tokenizer.tokenize(subsent)
                        for token in tokenized:
                            cleaned = tlg_plaintext_cleanup(token, rm_punctuation=True, rm_periods=True)
                            sentence = cltk_normalize(cleaned)
                            if len(self.word_tokenizer.tokenize(sentence)) > 5:
                                cleaned_sents.append(sentence)
                else:
                    tokenized = self.sent_tokenizer.tokenize(sent)
                    for token in tokenized:
                        cleaned = tlg_plaintext_cleanup(token, rm_punctuation=True, rm_periods=True)
                        sentence = cltk_normalize(cleaned)
                        if len(self.word_tokenizer.tokenize(sentence)) > 5:
                            cleaned_sents.append(sentence)
        return cleaned_sents
        
    def strip_accents_from_sentence(self, sent):
        words = sent.split()
        words = list(map(lambda x: strip_accents(x), words))
        return " ".join(words)

    def write_csv(self, filename, examples, labels, strip_accents=True):
        with open(filename, 'w') as f:
            csv_writer = csv.writer(f, delimiter='\t')
            for sent, author in zip(examples, labels):
                if strip_accents:
                    sent = self.strip_accents_from_sentence(sent)
                csv_writer.writerow([sent, author])
    
    def write_txt(self, filename, examples, strip_accents=True):
        with open(filename, 'w') as f:   
            for sent in examples:
                if strip_accents:
                    sent = self.strip_accents_from_sentence(sent)
                f.write(sent)
                f.write("\n")

    def read_txt(self, filename):
        lines = []
        with open(filename, 'r') as f:
            for line in f:
                lines.append(line)
        return lines
    
    def read_csv(self, filename):
        sentences = []
        authors = []
        with open(filename, 'r') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            for row in csv_reader:
                sentences.append(row[0])
                authors.append(row[1])
        return sentences, authors

    def label_file(self, in_filename, out_filename, labels_filename):
        labels = self.read_txt(labels_filename)
        with open(in_filename, 'r') as f_in:
            with open(out_filename, 'w') as f_out:
                csv_writer = csv.writer(f_out, delimiter='\t')
                for line, label in zip(f_in, labels):
                    print(line.strip(), label)
                    csv_writer.writerow([line.strip(), label.strip()])

    def calculate_top_tfidf_words(self):
        tfidf = self.tfidf_vectorizer.fit_transform(os.listdir(DOCUMENT_PATH))
        print(tfidf)

    def load_corpus(self): 
        authors = []
        sentences = []
        authors_apology = []
        sentences_apology = []
        titles = []

        documents = {}
        for doc in self.corpus_reader.docs():
            if doc['author'] in ['plato', 'xenophon']:
                document = []
                for sent in self.process_document(doc):
                    if doc['englishTitle'] in ['Apology']:
                        authors_apology.append(doc['author'])
                        sentences_apology.append(sent)
                        document.append(sent)
                    else:
                        sentences.append(sent)
                        authors.append(doc['author'])
                        titles.append(doc['englishTitle'])
                        document.append(sent)
                documents[f"{doc['englishTitle'].replace(' ', '')}_{doc['author']}"] = document

        for title, document in documents.items():
            self.write_txt(DOCUMENT_PATH + title + ".txt", document)
            self.write_txt(DOCUMENT_PATH_LEMMA + title + ".txt", self.lemmatize(document))

        sentences_train, sentences_test, authors_train, authors_test = train_test_split(sentences, authors, test_size=0.20, stratify=authors)
        sentences_valid, sentences_test, authors_valid, authors_test = train_test_split(sentences_test, authors_test, test_size=0.50, stratify=authors_test)

        sentences_apology, authors_apology = shuffle(sentences_apology, authors_apology)

        self.write_txt(TRAIN_SENTENCES_FILE_NAME, sentences_train)
        self.write_txt(VALID_SENTENCES_FILE_NAME, sentences_valid)
        self.write_txt(TEST_SENTENCES_FILE_NAME, sentences_test)
        self.write_txt(APOLOGY_SENTENCES_FILE_NAME, sentences_apology)

        self.write_txt(TRAIN_AUTHORS_FILE_NAME, authors_train)
        self.write_txt(VALID_AUTHORS_FILE_NAME, authors_valid)
        self.write_txt(TEST_AUTHORS_FILE_NAME, authors_test)
        self.write_txt(APOLOGY_AUTHORS_FILE_NAME, authors_apology)

        self.write_csv(TRAIN_CORPUS_FILE_NAME, sentences_train, authors_train)
        self.write_csv(VALID_CORPUS_FILE_NAME, sentences_valid, authors_valid)
        self.write_csv(TEST_CORPUS_FILE_NAME, sentences_test, authors_test)
        self.write_csv(APOLOGY_CORPUS_FILE_NAME, sentences_apology, authors_apology)

        self.write_csv(TRAIN_CORPUS_ACCENTS_FILE_NAME, sentences_train, authors_train, strip_accents=False)
        self.write_csv(VALID_CORPUS_ACCENTS_FILE_NAME, sentences_valid, authors_valid, strip_accents=False)
        self.write_csv(TEST_CORPUS_ACCENTS_FILE_NAME, sentences_test, authors_test, strip_accents=False)
        self.write_csv(APOLOGY_CORPUS_ACCENTS_FILE_NAME, sentences_apology, authors_apology, strip_accents=False)
    
    def byte_pair_encoding(self, num_iterations=2500):
        os.system(f"subword-nmt learn-bpe -s {num_iterations} < {TRAIN_SENTENCES_FILE_NAME} > {ENCODING_FILE_NAME}")
        os.system(f"subword-nmt apply-bpe -c {ENCODING_FILE_NAME} < {TRAIN_SENTENCES_FILE_NAME} > {TRAIN_CORPUS_ENCODED_FILE_NAME}")
        os.system(f"subword-nmt apply-bpe -c {ENCODING_FILE_NAME} < {VALID_SENTENCES_FILE_NAME} > {VALID_CORPUS_ENCODED_FILE_NAME}")
        os.system(f"subword-nmt apply-bpe -c {ENCODING_FILE_NAME} < {TEST_SENTENCES_FILE_NAME} > {TEST_CORPUS_ENCODED_FILE_NAME}")
        os.system(f"subword-nmt apply-bpe -c {ENCODING_FILE_NAME} < {APOLOGY_SENTENCES_FILE_NAME} > {APOLOGY_CORPUS_ENCODED_FILE_NAME}")

        self.label_file(TRAIN_CORPUS_ENCODED_FILE_NAME, TRAIN_CORPUS_ENCODED_LABELED_FILE_NAME, TRAIN_AUTHORS_FILE_NAME)
        self.label_file(TEST_CORPUS_ENCODED_FILE_NAME, TEST_CORPUS_ENCODED_LABELED_FILE_NAME, TEST_AUTHORS_FILE_NAME)
        self.label_file(VALID_CORPUS_ENCODED_FILE_NAME, VALID_CORPUS_ENCODED_LABELED_FILE_NAME, VALID_AUTHORS_FILE_NAME)
        self.label_file(APOLOGY_CORPUS_ENCODED_FILE_NAME, APOLOGY_CORPUS_ENCODED_LABELED_FILE_NAME, APOLOGY_AUTHORS_FILE_NAME)
    
    def lemmatize(self, sentences, concat=False):
        lemmatized_sents = []
        for sent in sentences:
            lemmatized = []
            for word in sent.split():
                lemmatized.append(self.lemmatizer.lemmatize(word)[0])
                if concat:
                    lemmatized.append(word)
            lemmatized_sent = " ".join(lemmatized)
            lemmatized_sents.append(lemmatized_sent)
        return lemmatized_sents
    
    def lemmatize_csv(self, in_filename, out_filename, concat=False):
        sentences, authors = self.read_csv(in_filename)
        lemmatized_sents = self.lemmatize(sentences, concat=concat)
        
        self.write_csv(out_filename, lemmatized_sents, authors)

    def lemmatized(self):
        self.lemmatize_csv(TRAIN_CORPUS_ACCENTS_FILE_NAME, TRAIN_CORPUS_LEMMA_FILE_NAME)
        self.lemmatize_csv(TEST_CORPUS_ACCENTS_FILE_NAME, TEST_CORPUS_LEMMA_FILE_NAME)
        self.lemmatize_csv(VALID_CORPUS_ACCENTS_FILE_NAME, VALID_CORPUS_LEMMA_FILE_NAME)
        self.lemmatize_csv(APOLOGY_CORPUS_ACCENTS_FILE_NAME, APOLOGY_CORPUS_LEMMA_FILE_NAME)
    
    def lemmatized_concat(self):
        self.lemmatize_csv(TRAIN_CORPUS_ACCENTS_FILE_NAME, TRAIN_CORPUS_LEMMA_CONCAT_FILE_NAME, concat=True)
        self.lemmatize_csv(VALID_CORPUS_ACCENTS_FILE_NAME, VALID_CORPUS_LEMMA_CONCAT_FILE_NAME, concat=True)
        self.lemmatize_csv(TEST_CORPUS_ACCENTS_FILE_NAME, TEST_CORPUS_LEMMA_CONCAT_FILE_NAME, concat=True)
        self.lemmatize_csv(APOLOGY_CORPUS_ACCENTS_FILE_NAME, APOLOGY_CORPUS_LEMMA_CONCAT_FILE_NAME, concat=True)
    
    def load_texts(self):
        print("Loading corpus...")
        self.load_corpus()
    
        print("Byte pair encoding...")
        self.byte_pair_encoding()

        print("Lemma...")
        self.lemmatized()

        print("Lemma Concat...")
        self.lemmatized_concat()