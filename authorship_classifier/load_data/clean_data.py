"""Parses (if needed) and cleans F1KG text files and Perseus Cannonical Greek Lit XML files for use with charater-based neural networks. Assumes the files have already been downloaded (if not, run download_data.py first)."""
import glob
import os
import re
from cltk.corpus.utils.formatter import cltk_normalize
from bs4 import BeautifulSoup
from utils import write_to_file, write_to_csv, write_spurious_to_csv

DUBIOUS_PLATO = [
    "tlg0059.tlg013.perseus-grc2.xml",
    "tlg0059.tlg014.perseus-grc2.xml",
    "tlg0059.tlg015.perseus-grc2.xml",
    "tlg0059.tlg016.perseus-grc2.xml",
    "tlg0059.tlg017.perseus-grc2.xml",
    "tlg0059.tlg028.perseus-grc2.xml",
    "tlg0059.tlg029.perseus-grc2.xml",
    "tlg0059.tlg033.perseus-grc2.xml",
    "tlg0059.tlg035.perseus-grc2.xml",
    "tlg0059.tlg036.perseus-grc2.xml",
]

CHARS_TO_REMOVE = """{}΄|Ϛݲ§ϛ♃5ᾱᾅὝᾂ̆ᾦ#Ἦ*ᾆ⟩ὋἎὒὮ′̣ϝὯἾ͵ῂüὬ⌋⌈‚•ä+̀ö&–ͅᾕë1͂ῲᾡἇἛἋGϠ¶%ῢ^ἊἯæᾇ\\2ᾁῡἚ̋/⌉Ὢß!⌊Ἣ=ῗóΌ`3ï⌞⌟ᾲΆ65ϡ̈4∗Ὣq═òΈϞ○à7áΊᾒ‖~אϟΪϥ›\u200b⁄‹íé⋖ὊÍ9Ἲ̄8{ῧ}Üᾟᾍᾨ―ΉŕὟ⩹✶0Ώᾯᾥᾌ\x8d⟦⟧\x9a¦ᾬἻ£a⋇ῐ¬ÓbῚŒἿἏÉῌᾃ\x98°ΎῈÁ⨆�
ç↑ạὛ⏔⏑̅✝ú\x9dᾺụᾢᾓᾘῼùÒSῠϙ─\x90לṕᾣô\x9cῸᾜḿ$⦵⊏ī\x9eֹ\x8eÌćÆ\x8fǁ⊻@ū÷Ҁ∾ῺìῪ\x8c\x81ᾮᾈèÿœῩῊ\x88⊤З♀⊙\xadÄÖᾞߤ⁑⸨\x8aḍ⫯∼źẂ⋆★Ῑᾩ‵ᾎý√⏝⏓⏕ṃ×ȳហḾti¿⥽⥼⊣⊔ӄẉ͎\u070eҏďĎ̠◻ᾰ\ue1c2rƑ̧\x7fេឲិតា
ỳӕῘLᾙΫẃ☾☿♂♄⊢⋃Ā±TMĹ€║̇čō"""

CHARS_TO_REPLACE = {
    "∠": "Δ",
    "△": "Δ",
    "\ufeff": " ",  # ZERO WIDTH NO-BREAK SPACE (U+FEFF)
    "ῑ": "ϊ",
    "✝": "†",
    # regularize angled brackets
    "<": "⟨",
    ">": "⟩",
    "〈": "⟨",
    "〉": "⟩",
    # regularize quotation marks
    "“": '"',
    "”": '"',
    "«": '"',
    "»": '"',
    "„": '"',
    "‟": '"',
    "‹": "'",
    "›": "'",
    "‘": "'",
    "": "'",
}

author_index = {
    "0059": "Plato",
    "0032": "Xenophon",
    "0010": "Isocrates",
    "0014": "Demosthenes",
    "0026": "Aeschines",
    "0540": "Lycias",
    "0003": "Thucydides"
}

def get_files(directory, regex, files_to_exclude):
    """Finds files matching the regex in the specified directory except for those in the exclusion list. Returns a list of file paths."""
    authors = []
    works = []
    files = []
    spurious = []
    spurious_works = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            if re.search(regex, f): 
                if f not in files_to_exclude:
                    files.append(dirpath + "/" + f)
                    authors.append(author_index[f[3:7]])
                    works.append(f[11:14])
                else:
                    spurious.append(dirpath + '/' + f)
                    spurious_works.append(f[11:14])

    print(f"Number of texts found: {len(list(files))}")
    return files, authors, works, spurious, spurious_works


def parse_xml(fp):
    """Parses a Perseus XML file. The approch here is very rough. Ideally the nodes should be traversed and those containing a significan amount of non-greek characters removed (see: https://github.com/ThomasK81/TEItoCEX)"""
    soup = BeautifulSoup(fp, "xml")
    # remote 'note' tags
    for note_tag in soup.find_all("note"):
        note_tag.decompose()
    assert len(list(soup.find_all("note"))) == 0
    # get all remaining text
    raw_text = soup.find("text").get_text()
    return raw_text


def parse_perseus_xml(files):
    """Parses the given XML files. Returns the parsed files as well as a list of files which couldn't be parsed at all."""
    raw_texts = []
    failed_to_parse = []
    print("Parsing: ", end="")
    for i, f in enumerate(files):
        raw_text = ""
        with open(f) as fp:
            raw_text = parse_xml(fp)
        if not raw_text:
            failed_to_parse.append(f)
            print("x", end="", flush=True)
        else:
            raw_texts.append(raw_text)
            print(".", end="", flush=True)
    print(f"\nNumber of texts parsed: {len(raw_texts)}")
    print(f"Number of texts which could not be parsed: {len(failed_to_parse)}")
    return raw_texts, failed_to_parse


def remove_braces(text):
    """Removes various braces and brackets from the text."""
    # remove square and curly braces and their contents (if less than 40 chars)
    text = re.sub(r"\{[^}]{,40}\}", "", text)
    # remove text between ⟦ and ⟧ as the pairs were manually checked, there is no length restriction
    text = re.sub(r"⟦[^⟧]*⟧", "", text)
    return text


def remove_unwanted_chars(token, chars_to_remove):
    """Removes all characters is the list from the token."""
    output = filter(lambda c: c not in chars_to_remove, token)
    return "".join(output)


def replace_chars(token, chars_to_replace):
    """Replaces all characters in the dict with their specified replacement."""
    for c in token:
        if c in chars_to_replace:
            token = re.sub(c, chars_to_replace[c], token)
    return token


def clean_tokens(tokens, chars_to_remove, chars_to_replace):
    """Cleans a list of tokens."""
    cleaned_tokens = []
    for t in tokens:
        if t:
            # remove words in which latin characters appear
            if not re.search(r"\w+", t, re.ASCII):
                # remove tokens containing digits
                if not re.search(r"\d+", t):
                    # normalize
                    t = cltk_normalize(t)
                    t = t.strip("\t\r\n")
                    # remove unwanted chars
                    t = remove_unwanted_chars(t, chars_to_remove)
                    t = replace_chars(t, chars_to_replace)
                    # remove any inter-word hypens or en-dashes
                    t = re.sub(r"([^\s])(-|–)", r"\1", t)
                    # convert some other forms of whitespace to normal spaces
                    t = re.sub(r"\s+", " ", t)
                    # remove repeated whitespace
                    t = re.sub(r"\s{2,}", " ", t)
                    cleaned_tokens.append(t)
    return cleaned_tokens


def clean_texts(raw_texts, chars_to_remove, chars_to_replace):
    """Cleans a list of texts."""
    data = []
    print("Cleaning: ", end="")
    for i, raw_text in enumerate(raw_texts):
        text = ""
        raw_text = remove_braces(raw_text)
        raw_text = raw_text.splitlines()
        for line in raw_text:
            if line != "\n":
                tokens = line.split(" ")
                cleaned_tokens = clean_tokens(tokens, chars_to_remove, chars_to_replace)
                tokens = " ".join([t for t in cleaned_tokens])
                text += tokens.strip("\n") + " "
        # remove empty parentheses
        text = re.sub(r"\(\s*\)", "", text)
        # remove empty angled brackets
        text = re.sub(r"⟨\s*⟩", "", text)
        # final whitespace pass
        text = re.sub(r"\s{2,}", " ", text)
        data.append(text.strip(" "))
        print(".", end="", flush=True)
    print(f"\nNumber of texts cleaned: {len(data)}")
    return data


def clean_data():
    """Cleans the Perseus and F1KG data to produce the dataset."""
    print("Cleaning Perseus texts")
    perseus_dir = "canonical-greekLit/data"
    perseus_regex = re.compile("grc[0-9]*\.xml$")
    perseus_files, authors, works, spurious_text, spurious_works = get_files(perseus_dir, perseus_regex, DUBIOUS_PLATO)
    perseus_raw_texts, failed_to_parse = parse_perseus_xml(perseus_files)
    if failed_to_parse:
        write_to_file("perseus_parsing_failures.txt", failed_to_parse)
    spurious_raw_texts, failed_spurious = parse_perseus_xml(spurious_text)
    if failed_to_parse:
        write_to_file("perseus_parsing_spurious_failures.txt", failed_spurious)

    perseus_data = clean_texts(perseus_raw_texts, CHARS_TO_REMOVE, CHARS_TO_REPLACE)
    spurious_data = clean_texts(spurious_raw_texts, CHARS_TO_REMOVE, CHARS_TO_REPLACE)


    write_to_csv("texts.csv", perseus_data, authors, works)
    write_spurious_to_csv("spurious_texts.csv", spurious_data, spurious_works)



if __name__ == "__main__":
    clean_data()
