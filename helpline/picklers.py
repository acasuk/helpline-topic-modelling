"""
Utility functions to prepare and pickle large numbers of documents.
"""

import pickle
import pandas as pd
import nltk

# Install/Update NLTK punkt package
nltk.download("punkt")

# Load BO's stopwords list
with open("BO_stopwords.pkl","rb") as infile:
    stop_words = pickle.load(infile)+["remove"] # To cover e.g.  "person removed", "name removed", "phone removed"

def lemmatise(filename):
    """
    Lemmatise a text
    """
    pass
