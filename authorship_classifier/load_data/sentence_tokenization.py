"""Loads and uses a custom trained Punkt tokenizer. The tokenizer has been trained on the Ancient_Greek_ML dataset using the NLTK's Punkt implementation (https://www.nltk.org/_modules/nltk/tokenize/punkt.html)."""
from nltk.tokenize.punkt import PunktSentenceTokenizer
from cltk.tokenize.word import WordTokenizer
from cltk.corpus.utils.formatter import tlg_plaintext_cleanup
from greek_accentuation.characters import strip_accents
import csv
import sys
csv.field_size_limit(sys.maxsize)
import os

from utils import write_to_file, write_to_csv, write_spurious_to_csv
import pickle
import re

speakers = [
    "ΣΩ",
    "ΑΘ",
    "ΞΕ",
    "ΚΛ",
    "ΚΕΦ",
    "ΠΡΩ",
    "ΝΕ",
    "ΑΛ",
    "Ἐ",
    "ΦΑΙ",
    "ΜΕΝ",
    "ΜΕ",
    "ΚΑΛ",
    "ΕΤ",
    "ΚΡ",
    "ΤΙ",
    "Ἀ",
    "ΘΕΟ",
    "ΛΑ",
    "ΓΟΡ",
]
numbers = ["α ́", "β ́", "γ ́", "δ ́", "ε ́", "ζ ́", "η ́", "θ ́", "ι ́", "ς ́", "κ ́"]
other_abbreviations = [
    "ὑπ",
    "Ἰνδ",
    "γρ",
]
abbreviations = speakers + numbers + other_abbreviations
ABBREVIATIONS = [i.lower() for i in abbreviations]

word_tokenizer = WordTokenizer('greek')


def get_corpus():
    with open("Ancient_Greek_ML.txt", "r") as f:
        texts = f.read()
    return texts

def strip_accents_from_sentence(sent):
        words = sent.split()
        words = list(map(lambda x: strip_accents(x), words))
        return " ".join(words)


def non_destructive_split(t, delim):
    """Splits the text t after the delimiter delim, retaining delim with the part of t that preceeded it. Returns a list of strings"""
    split_texts = []
    initial_split = re.split(f"({delim})", t)
    for s in initial_split:
        if s != delim:
            split_texts.append(s)
        else:
            split_texts[-1] = split_texts[-1] + delim
    return split_texts


def my_split(texts, delim):
    """Splits non-destructively on delim."""
    tokenized_texts = []
    for t in texts:
        sentences = non_destructive_split(t, delim)
        for s in sentences:
            if s:
                tokenized_texts.append(s.strip(" "))
    return tokenized_texts


def additional_tokenization(texts):
    """Splits texts on semicola and middle dots."""
    tokenized_texts = my_split(texts, ";")
    tokenized_texts = my_split(tokenized_texts, "·")
    return tokenized_texts


def tokenize_with_custom_punkt_tokenizer(texts):
    """Sentence tokenizes texts using a custom Punkt tokenizer followed by additional tokenization on semicola and middle dots."""
    # a trainer is loaded rather than a tokenizer so that additional abbreviations can be added manually.
    with open("ancient_greek_punkt_trainer.pickle", "rb") as f:
        trainer = pickle.load(f)
    # add additional abbreviations
    for abbv in ABBREVIATIONS:
        trainer._params.abbrev_types.add(abbv)
    tokenizer = PunktSentenceTokenizer(trainer.get_params())
    tokenized_texts = tokenizer.tokenize(texts)
    tokenized_texts = additional_tokenization(tokenized_texts)
    return tokenized_texts


def sentence_tokenize_corpus():
    """Fetches and tokenizes the corpus then writes it back out."""
    with open("texts.csv", 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        print("Sentence tokenizing...")
        for i, row in enumerate(csv_reader):
            tokenized_texts = tokenize_with_custom_punkt_tokenizer(row[0])

            rc = 0
            group = ""
            for num, text in enumerate(tokenized_texts):
                text = strip_accents_from_sentence(text)
                text = tlg_plaintext_cleanup(text, rm_punctuation=True, rm_periods=True)
                wc = len(word_tokenizer.tokenize(text))
                if wc > 5:
                    write_to_csv("sents.csv", [text], [row[1]], [row[2]])

                rc += wc
                group += text
                if rc > 100:
                    write_to_csv("groups.csv", [group], [row[1]], [row[2]])
                    rc = 0
                    group = ""

            write_to_csv("groups.csv", [group], [row[1]], [row[2]])
    
    with open("spurious_texts.csv", 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        print("Sentence tokenizing...")
        for i, row in enumerate(csv_reader):
            tokenized_texts = tokenize_with_custom_punkt_tokenizer(row[0])

            rc = 0
            group = ""
            for num, text in enumerate(tokenized_texts):
                text = strip_accents_from_sentence(text)
                text = tlg_plaintext_cleanup(text, rm_punctuation=True, rm_periods=True)
                wc = len(word_tokenizer.tokenize(text))
                if wc > 5:
                    write_spurious_to_csv("sents_spurious.csv", [text], [row[1]])

                rc += wc
                group += " "
                group += text
                if rc > 100:
                    write_spurious_to_csv("groups_spurious.csv", [group], [row[1]])
                    rc = 0
                    group = ""

            write_spurious_to_csv("groups_spurious.csv", [group], [row[1]])


if __name__ == "__main__":
    sentence_tokenize_corpus()
