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

TRAIN_CORPUS_FILE_NAME = "../data/greek_corpus_train.csv"
VALID_CORPUS_FILE_NAME = "../data/greek_corpus_valid.csv"
TEST_CORPUS_FILE_NAME = "../data/greek_corpus_test.csv"
APOLOGY_CORPUS_FILE_NAME = "../data/greek_corpus_apology.csv"
TRAIN_CORPUS_ENCODED_FILE_NAME = "../data/greek_corpus_train_encoded.txt"
VALID_CORPUS_ENCODED_FILE_NAME = "../data/greek_corpus_valid_encoded.txt"
TEST_CORPUS_ENCODED_FILE_NAME = "../data/greek_corpus_test_encoded.txt"
APOLOGY_CORPUS_ENCODED_FILE_NAME = "../data/greek_corpus_apology_encoded.txt"
TRAIN_CORPUS_ENCODED_LABELED_FILE_NAME = "../data/greek_corpus_train_encoded_labeled.csv"
VALID_CORPUS_ENCODED_LABELED_FILE_NAME = "../data/greek_corpus_valid_encoded_labeled.csv"
TEST_CORPUS_ENCODED_LABELED_FILE_NAME = "../data/greek_corpus_test_encoded_labeled.csv"
APOLOGY_CORPUS_ENCODED_LABELED_FILE_NAME = "../data/greek_corpus_apology_encoded_labeled.csv"
TRAIN_ENCODING_FILE_NAME = "../data/greek_byte_pair_training.txt"
VALID_ENCODING_FILE_NAME = "../data/greek_byte_pair_valid.txt"
TEST_ENCODING_FILE_NAME = "../data/greek_byte_pair_test.txt"
APOLOGY_ENCODING_FILE_NAME = "../data/greek_byte_pair_apology.txt"
ENCODING_FILE_NAME = "../data/greek_byte_pair_encoding.txt"
TRAIN_CORPUS_CHARNGRAM_FILE_NAME = "../data/greek_corpus_train_charngram.csv"
VALID_CORPUS_CHARNGRAM_FILE_NAME = "../data/greek_corpus_valid_charngram.csv"
TEST_CORPUS_CHARNGRAM_FILE_NAME = "../data/greek_corpus_test_charngram.csv"
APOLOGY_CORPUS_CHARNGRAM_FILE_NAME = "../data/greek_corpus_apology_charngram.csv"
TRAIN_CORPUS_CHARNGRAMNOWORD_FILE_NAME = "../data/greek_corpus_train_charngram_noword.csv"
VALID_CORPUS_CHARNGRAMNOWORD_FILE_NAME = "../data/greek_corpus_valid_charngram_noword.csv"
TEST_CORPUS_CHARNGRAMNOWORD_FILE_NAME = "../data/greek_corpus_test_charngram_noword.csv"
APOLOGY_CORPUS_CHARNGRAMNOWORD_FILE_NAME = "../data/greek_corpus_apology_charngram_noword.csv"
TRAIN_CORPUS_LEMMA_FILE_NAME = "../data/greek_corpus_train_lemma.csv"
VALID_CORPUS_LEMMA_FILE_NAME = "../data/greek_corpus_valid_lemma.csv"
TEST_CORPUS_LEMMA_FILE_NAME = "../data/greek_corpus_test_lemma.csv"
APOLOGY_CORPUS_LEMMA_FILE_NAME = "../data/greek_corpus_apology_lemma.csv"
TRAIN_CORPUS_LEMMA_CONCAT_FILE_NAME = "../data/greek_corpus_train_lemma_concat.csv"
VALID_CORPUS_LEMMA_CONCAT_FILE_NAME = "../data/greek_corpus_valid_lemma_concat.csv"
TEST_CORPUS_LEMMA_CONCAT_FILE_NAME = "../data/greek_corpus_test_lemma_concat.csv"
APOLOGY_CORPUS_LEMMA_CONCAT_FILE_NAME = "../data/greek_corpus_apology_lemma_concat.csv"

np.random.seed(300)

sent_tokenizer = SentenceTokenizer()
word_tokenizer = WordTokenizer('greek')

perseus_reader = get_corpus_reader(corpus_name='greek_text_perseus', language='greek')

def process_document(doc):
    cleaned_sents = []
    for paragraph in doc['text'].values():
        if type(paragraph) != str:
            paragraph = paragraph.values()
        else:
            paragraph = sent_tokenizer.tokenize(paragraph)
        for sent in paragraph:
            if type(sent) is dict:
                for subsent in sent.values():
                    tokenized = sent_tokenizer.tokenize(subsent)
                    for token in tokenized:
                        cleaned = tlg_plaintext_cleanup(token, rm_punctuation=True, rm_periods=True)
                        sentence = cltk_normalize(cleaned)
                        if len(word_tokenizer.tokenize(sentence)) > 5:
                            cleaned_sents.append(sentence)
            else:
                tokenized = sent_tokenizer.tokenize(sent)
                for token in tokenized:
                    cleaned = tlg_plaintext_cleanup(token, rm_punctuation=True, rm_periods=True)
                    sentence = cltk_normalize(cleaned)
                    if len(word_tokenizer.tokenize(sentence)) > 5:
                        cleaned_sents.append(sentence)
    return cleaned_sents

def strip_accents_from_sentence(sent):
    words = sent.split()
    words = list(map(lambda x: strip_accents(x), words))
    return " ".join(words)

authors = []
sentences = []
authors_apology = []
sentences_apology = []
titles = []

for doc in perseus_reader.docs():
    if doc['author'] in ['plato', 'xenophon']:
        for sent in process_document(doc):
            if doc['englishTitle'] in ['Apology']:
                authors_apology.append(doc['author'])
                sentences_apology.append(sent)
            else:
                sentences.append(sent)
                authors.append(doc['author'])
                titles.append(doc['englishTitle'])

print(Counter(titles))
print(Counter(authors))
print(Counter(authors_apology))

print(sentences_valid)

sentences_train, sentences_test, authors_train, authors_test = train_test_split(sentences, authors, test_size=0.20, stratify=authors)
sentences_valid, sentences_test, authors_valid, authors_test = train_test_split(sentences_test, authors_test, test_size=0.50, stratify=authors_test)

sentences_apology, authors_apology = shuffle(sentences_apology, authors_apology)

print(len(sentences_train), len(sentences_valid), len(sentences_test), len(sentences_apology))

with open(TRAIN_ENCODING_FILE_NAME, 'w') as train_encoding_file:   
    for sent in sentences_train:
        train_encoding_file.write(strip_accents_from_sentence(sent))
        train_encoding_file.write("\n")

with open(VALID_ENCODING_FILE_NAME, 'w') as valid_encoding_file:   
    for sent in sentences_valid:
        valid_encoding_file.write(strip_accents_from_sentence(sent))
        valid_encoding_file.write("\n")

with open(TEST_ENCODING_FILE_NAME, 'w') as test_encoding_file:   
    for sent in sentences_test:
        test_encoding_file.write(strip_accents_from_sentence(sent))
        test_encoding_file.write("\n")

with open(APOLOGY_ENCODING_FILE_NAME, 'w') as apology_encoding_file:   
    for sent in sentences_apology:
        apology_encoding_file.write(strip_accents_from_sentence(sent))
        apology_encoding_file.write("\n")

with open(TRAIN_CORPUS_FILE_NAME, 'w') as train_file:
    csv_writer = csv.writer(train_file, delimiter='\t')
    for sent, author in zip(sentences_train, authors_train):
        csv_writer.writerow([strip_accents_from_sentence(sent), author])

with open(VALID_CORPUS_FILE_NAME, 'w') as valid_file:
    csv_writer = csv.writer(valid_file, delimiter='\t')
    for sent, author in zip(sentences_valid, authors_valid):
        csv_writer.writerow([strip_accents_from_sentence(sent), author])

with open(TEST_CORPUS_FILE_NAME, 'w') as test_file:
    csv_writer = csv.writer(test_file, delimiter='\t')
    for sent, author in zip(sentences_test, authors_test):
        csv_writer.writerow([strip_accents_from_sentence(sent), author])

with open(APOLOGY_CORPUS_FILE_NAME, 'w') as apology_file:
    csv_writer = csv.writer(apology_file, delimiter='\t')
    for sent, author in zip(sentences_apology, authors_apology):
        csv_writer.writerow([strip_accents_from_sentence(sent), author])

os.system(f"subword-nmt learn-bpe -s 2500 < {TRAIN_ENCODING_FILE_NAME} > {ENCODING_FILE_NAME}")
os.system(f"subword-nmt apply-bpe -c {ENCODING_FILE_NAME} < {TRAIN_ENCODING_FILE_NAME} > {TRAIN_CORPUS_ENCODED_FILE_NAME}")
os.system(f"subword-nmt apply-bpe -c {ENCODING_FILE_NAME} < {VALID_ENCODING_FILE_NAME} > {VALID_CORPUS_ENCODED_FILE_NAME}")
os.system(f"subword-nmt apply-bpe -c {ENCODING_FILE_NAME} < {TEST_ENCODING_FILE_NAME} > {TEST_CORPUS_ENCODED_FILE_NAME}")
os.system(f"subword-nmt apply-bpe -c {ENCODING_FILE_NAME} < {APOLOGY_ENCODING_FILE_NAME} > {APOLOGY_CORPUS_ENCODED_FILE_NAME}")

with open(TRAIN_CORPUS_ENCODED_FILE_NAME, 'r') as train_file_in:
    with open(TRAIN_CORPUS_ENCODED_LABELED_FILE_NAME, 'w') as train_file_out:
        csv_writer = csv.writer(train_file_out, delimiter='\t')
        for line, author in zip(train_file_in, authors_train):
            csv_writer.writerow([line.strip(), author])


with open(VALID_CORPUS_ENCODED_FILE_NAME, 'r') as valid_file_in:
    with open(VALID_CORPUS_ENCODED_LABELED_FILE_NAME, 'w') as valid_file_out:
        csv_writer = csv.writer(valid_file_out, delimiter='\t')
        for line, author in zip(valid_file_in, authors_valid):
            csv_writer.writerow([line.strip(), author])

with open(TEST_CORPUS_ENCODED_FILE_NAME, 'r') as test_file_in:
    with open(TEST_CORPUS_ENCODED_LABELED_FILE_NAME, 'w') as test_file_out:
        csv_writer = csv.writer(test_file_out, delimiter='\t')
        for line, author in zip(test_file_in, authors_test):
            csv_writer.writerow([line.strip(), author])

with open(APOLOGY_CORPUS_ENCODED_FILE_NAME, 'r') as apology_file_in:
    with open(APOLOGY_CORPUS_ENCODED_LABELED_FILE_NAME, 'w') as apology_file_out:
        csv_writer = csv.writer(apology_file_out, delimiter='\t')
        for line, author in zip(apology_file_in, authors_apology):
            csv_writer.writerow([line.strip(), author])

def character_n_grams(corpus, authors, output, ngram_range=(3,6)):
    csv_writer = csv.writer(output, delimiter='\t')
    for sent, author in zip(corpus, authors):
        new_sent = ""
        for word in sent.split():
            word_augmented = "<" + strip_accents(word) + ">"
            for n in range(NGRAM_RANGE[0], NGRAM_RANGE[1]):
                new_sent += " " + " ".join([word_augmented[i:i+n] for i in range(len(word_augmented) - n + 1)])
            if len(word) < NGRAM_RANGE[0] or len(word) >= NGRAM_RANGE[1]:
                new_sent += " " + word_augmented
        csv_writer.writerow([new_sent, author])

NGRAM_RANGE = (3, 6)
with open(TRAIN_CORPUS_CHARNGRAM_FILE_NAME, 'w') as train_file:
    character_n_grams(sentences_train, authors_train, train_file, NGRAM_RANGE)

with open(VALID_CORPUS_CHARNGRAM_FILE_NAME, 'w') as valid_file:
    character_n_grams(sentences_valid, authors_valid, valid_file, NGRAM_RANGE)

with open(TEST_CORPUS_CHARNGRAM_FILE_NAME, 'w') as test_file:
    character_n_grams(sentences_test, authors_test, test_file, NGRAM_RANGE)

with open(APOLOGY_CORPUS_CHARNGRAM_FILE_NAME, 'w') as apology_file:
    character_n_grams(sentences_apology, authors_apology, apology_file, NGRAM_RANGE)

def character_n_grams_no_word(corpus, authors, output, ngram_range=(3,6)):
    csv_writer = csv.writer(output, delimiter='\t')
    for sent, author in zip(corpus, authors):
        new_sent = ""
        for word in sent.split():
            word_augmented = "<" + strip_accents(word) + ">"
            for n in range(NGRAM_RANGE[0], NGRAM_RANGE[1]):
                new_sent += " " + " ".join([word_augmented[i:i+n] for i in range(len(word_augmented) - n + 1)])
        csv_writer.writerow([new_sent, author])

NGRAM_RANGE = (3, 6)
with open(TRAIN_CORPUS_CHARNGRAMNOWORD_FILE_NAME, 'w') as train_file:
    character_n_grams_no_word(sentences_train, authors_train, train_file, NGRAM_RANGE)

with open(VALID_CORPUS_CHARNGRAMNOWORD_FILE_NAME, 'w') as valid_file:
    character_n_grams_no_word(sentences_valid, authors_valid, valid_file, NGRAM_RANGE)

with open(TEST_CORPUS_CHARNGRAMNOWORD_FILE_NAME, 'w') as test_file:
    character_n_grams_no_word(sentences_test, authors_test, test_file, NGRAM_RANGE)

with open(APOLOGY_CORPUS_CHARNGRAMNOWORD_FILE_NAME, 'w') as apology_file:
    character_n_grams_no_word(sentences_apology, authors_apology, apology_file, NGRAM_RANGE)

lemmatizer = LemmaReplacer('greek')
def lemmatized(corpus, authors, output):
    csv_writer = csv.writer(output, delimiter='\t')
    for sent, author in zip(corpus, authors):
        lemmatized = []
        for word in sent.split():
            lemmatized.append(strip_accents(lemmatizer.lemmatize(word)[0]))
        lemmatized_sent = " ".join(lemmatized)
        csv_writer.writerow([lemmatized_sent, author])

def lemmatized_concat(corpus, authors, output):
    csv_writer = csv.writer(output, delimiter='\t')
    for sent, author in zip(corpus, authors):
        lemmatized = []
        for word in sent.split():
            lemmatized.append(strip_accents(lemmatizer.lemmatize(word)[0]))
            lemmatized.append(strip_accents(word))
        lemmatized_sent = " ".join(lemmatized)
        csv_writer.writerow([lemmatized_sent, author])

with open(TRAIN_CORPUS_LEMMA_FILE_NAME, 'w') as train_file:
    lemmatized(sentences_train, authors_train, train_file)

with open(VALID_CORPUS_LEMMA_FILE_NAME, 'w') as valid_file:
    lemmatized(sentences_valid, authors_valid, valid_file)

with open(TEST_CORPUS_LEMMA_FILE_NAME, 'w') as test_file:
    lemmatized(sentences_test, authors_test, test_file)

with open(APOLOGY_CORPUS_LEMMA_FILE_NAME, 'w') as apology_file:
    lemmatized(sentences_apology, authors_apology, apology_file)

with open(TRAIN_CORPUS_LEMMA_CONCAT_FILE_NAME, 'w') as train_file:
    lemmatized_concat(sentences_train, authors_train, train_file)

with open(VALID_CORPUS_LEMMA_CONCAT_FILE_NAME, 'w') as valid_file:
    lemmatized_concat(sentences_valid, authors_valid, valid_file)

with open(TEST_CORPUS_LEMMA_CONCAT_FILE_NAME, 'w') as test_file:
    lemmatized_concat(sentences_test, authors_test, test_file)

with open(APOLOGY_CORPUS_LEMMA_CONCAT_FILE_NAME, 'w') as apology_file:
    lemmatized_concat(sentences_apology, authors_apology, apology_file)






 

    








