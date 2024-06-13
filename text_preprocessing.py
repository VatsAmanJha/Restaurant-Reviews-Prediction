import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from bs4 import BeautifulSoup
import contractions


stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def text_lower(text):
    return text.lower()


def remove_punctuation(text):
    translator = str.maketrans(" ", " ", string.punctuation)
    return text.translate(translator)


def remove_number(text):
    return re.sub(r"\d", " ", text)


def remove_whitespace(text):
    return " ".join(text.split())


def remove_contractions(text):
    return contractions.fix(text)


def remove_HTML_tag(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return " ".join(words)


def stem_words(text):
    words = text.split()
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)


def clean_text(dataframe, column_name):
    dataframe_copy = dataframe.copy()
    print("\n=== Cleaning Process ===")

    print("\n⬇️ Removing HTML Tags ⬇️")
    dataframe_copy[column_name] = dataframe_copy[column_name].apply(remove_HTML_tag)

    print("\n⬇️ Lowercasing Text ⬇️")
    dataframe_copy[column_name] = dataframe_copy[column_name].apply(text_lower)

    print("\n⬇️ Removing Punctuation ⬇️")
    dataframe_copy[column_name] = dataframe_copy[column_name].apply(remove_punctuation)

    print("\n⬇️ Removing Numbers ⬇️")
    dataframe_copy[column_name] = dataframe_copy[column_name].apply(remove_number)

    print("\n⬇️ Removing Whitespace ⬇️")
    dataframe_copy[column_name] = dataframe_copy[column_name].apply(remove_whitespace)

    print("\n⬇️ Expanding Contractions ⬇️")
    dataframe[column_name] = dataframe[column_name].apply(remove_contractions)

    print("\n⬇️ Removing Stopwords ⬇️")
    dataframe[column_name] = dataframe[column_name].apply(remove_stopwords)

    print("\n⬇️ Stemming Words ⬇️")
    dataframe_copy[column_name] = dataframe_copy[column_name].apply(stem_words)

    print("\n=== Cleaning Completed ===\n")
    return dataframe_copy
