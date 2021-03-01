"""Creates the Ancient_Greek_ML dataset and then prepares the train, dev and test sets for the character-level BERT."""
from clean_data import clean_data
from sentence_tokenization import sentence_tokenize_corpus
from split_data import split_data
import os

os.chdir("../data")
clean_data()
sentence_tokenize_corpus()
split_data()
