from cltk.corpus.readers import get_corpus_reader
from cltk.corpus.utils.formatter import tlg_plaintext_cleanup, cltk_normalize
from cltk.stem.lemma import LemmaReplacer
import os

from sklearn.model_selection import train_test_split

TRAIN_CORPUS_FILE_NAME = "../data/greek_corpus_train.txt"
VALID_CORPUS_FILE_NAME = "../data/greek_corpus_valid.txt"
TEST_CORPUS_FILE_NAME = "../data/greek_corpus_test.txt"
TRAIN_CORPUS_ENCODED_FILE_NAME = "../data/greek_corpus_train_encoded.txt"
VALID_CORPUS_ENCODED_FILE_NAME = "../data/greek_corpus_valid_encoded.txt"
TEST_CORPUS_ENCODED_FILE_NAME = "../data/greek_corpus_test_encoded.txt"
ENCODING_FILE_NAME = "../data/greek_byte_pair_encoding.txt"
TRAIN_CORPUS_CHARNGRAM_FILE_NAME = "../data/greek_corpus_train_charngram.txt"
VALID_CORPUS_CHARNGRAM_FILE_NAME = "../data/greek_corpus_valid_charngram.txt"
TEST_CORPUS_CHARNGRAM_FILE_NAME = "../data/greek_corpus_test_charngram.txt"
TRAIN_CORPUS_LEMMA_FILE_NAME = "../data/greek_corpus_train_lemma.txt"
VALID_CORPUS_LEMMA_FILE_NAME = "../data/greek_corpus_valid_lemma.txt"
TEST_CORPUS_LEMMA_FILE_NAME = "../data/greek_corpus_test_lemma.txt"
TRAIN_CORPUS_LEMMA_CONCAT_FILE_NAME = "../data/greek_corpus_train_lemma_concat.txt"
VALID_CORPUS_LEMMA_CONCAT_FILE_NAME = "../data/greek_corpus_valid_lemma_concat.txt"
TEST_CORPUS_LEMMA_CONCAT_FILE_NAME = "../data/greek_corpus_test_lemma_concat.txt"



perseus_reader = get_corpus_reader(corpus_name='greek_text_perseus', language='greek')

def process_document(doc):
    cleaned_sents = []
    for paragraph in doc['text'].values():
        para_string = ""
        if type(paragraph) != str:
            for sent in paragraph.values():
                if type(sent) is dict:
                    for subsent in sent.values():
                        cleaned = tlg_plaintext_cleanup(subsent)
                        sentence = cltk_normalize(cleaned)
                        cleaned_sents.append(sentence)
                else:
                    cleaned = tlg_plaintext_cleanup(sent)
                    sentence = cltk_normalize(cleaned)
                    cleaned_sents.append(sentence)
    return cleaned_sents

sentences = []
for doc in perseus_reader.docs():
    if doc['author'] == 'xenophon':
        for sent in process_document(doc):
            sentences.append(sent)
    
sentences_train, sentences_test = train_test_split(sentences,test_size=0.1)
sentences_train, sentences_valid = train_test_split(sentences_train,test_size=0.1)


with open(TRAIN_CORPUS_FILE_NAME, 'w') as train_file:
    for sent in sentences_train:
        train_file.write(sent)
        train_file.write("\n")

with open(VALID_CORPUS_FILE_NAME, 'w') as valid_file:
    for sent in sentences_valid:
        valid_file.write(sent)
        valid_file.write("\n")

with open(TEST_CORPUS_FILE_NAME, 'w') as test_file:
    for sent in sentences_test:
        test_file.write(sent)
        test_file.write("\n")

os.system(f"subword-nmt learn-bpe -s 10000 < {TRAIN_CORPUS_FILE_NAME} > {ENCODING_FILE_NAME}")
os.system(f"subword-nmt apply-bpe -c {ENCODING_FILE_NAME} < {TRAIN_CORPUS_FILE_NAME} > {TRAIN_CORPUS_ENCODED_FILE_NAME}")
os.system(f"subword-nmt apply-bpe -c {ENCODING_FILE_NAME} < {VALID_CORPUS_FILE_NAME} > {VALID_CORPUS_ENCODED_FILE_NAME}")
os.system(f"subword-nmt apply-bpe -c {ENCODING_FILE_NAME} < {TEST_CORPUS_FILE_NAME} > {TEST_CORPUS_ENCODED_FILE_NAME}")

def character_n_grams(corpus, output, ngram_range=(3,6), ):
    for sent in corpus:
        new_sent = ""
        for word in sent.split():
            word_augmented = "<" + word + ">"
            for n in range(NGRAM_RANGE[0], NGRAM_RANGE[1]):
                new_sent += " " + " ".join([word_augmented[i:i+n] for i in range(len(word_augmented) - n + 1)])
            if len(word) < NGRAM_RANGE[0] or len(word) >= NGRAM_RANGE[1]:
                new_sent += " " + word_augmented
        output.write(new_sent)
        output.write("\n")

NGRAM_RANGE = (3, 6)
with open(TRAIN_CORPUS_CHARNGRAM_FILE_NAME, 'w') as train_file:
    character_n_grams(sentences_train, train_file, NGRAM_RANGE)

with open(VALID_CORPUS_CHARNGRAM_FILE_NAME, 'w') as valid_file:
    character_n_grams(sentences_valid, valid_file, NGRAM_RANGE)

with open(TEST_CORPUS_CHARNGRAM_FILE_NAME, 'w') as test_file:
    character_n_grams(sentences_test, test_file, NGRAM_RANGE)

lemmatizer = LemmaReplacer('greek')
def lemmatized(corpus, output):
    for sent in corpus:
        lemmatized = []
        for word in sent.split():
            lemmatized.append(lemmatizer.lemmatize(word)[0])
        print(lemmatized)
        lemmatized_sent = " ".join(lemmatized)
        output.write(lemmatized_sent)
        output.write("\n")

def lemmatized_concat(corpus, output):
    for sent in corpus:
        lemmatized = []
        for word in sent.split():
            lemmatized.append(lemmatizer.lemmatize(word)[0])
            lemmatized.append(word)
        lemmatized_sent = " ".join(lemmatized)
        output.write(lemmatized_sent)
        output.write("\n")

with open(TRAIN_CORPUS_LEMMA_FILE_NAME, 'w') as train_file:
    lemmatized(sentences_train, train_file)

with open(VALID_CORPUS_LEMMA_FILE_NAME, 'w') as valid_file:
    lemmatized(sentences_valid, valid_file)

with open(TEST_CORPUS_LEMMA_FILE_NAME, 'w') as test_file:
    lemmatized(sentences_test, test_file)

with open(TRAIN_CORPUS_LEMMA_CONCAT_FILE_NAME, 'w') as train_file:
    lemmatized_concat(sentences_train, train_file)

with open(VALID_CORPUS_LEMMA_CONCAT_FILE_NAME, 'w') as valid_file:
    lemmatized_concat(sentences_valid, valid_file)

with open(TEST_CORPUS_LEMMA_CONCAT_FILE_NAME, 'w') as test_file:
    lemmatized_concat(sentences_test, test_file)






 

    








