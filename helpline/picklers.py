"""
Utility functions to prepare and pickle large numbers of documents.
"""

import pickle
import pandas as pd
import nltk

# Install/Update NLTK punkt package
nltk.download("punkt")

# Load BO's stopwords list
with open("BO_stopwords.pkl", "rb") as infile:
    stop_words = pickle.load(infile) + [
        "remove"
    ]  # To cover e.g.  "person removed", "name removed", "phone removed"


def lemmatise_call(filename):
    """
    Lemmatise a text TODO
    """
    doc = pd.read_csv(filename)
    call = pd.DataFrame(nltk.sent_tokenize(text), columns=["Conversation"])
    call["Conversation"] = [
        re.sub("\s+", " ", sent) for sent in call["Conversation"]
    ]
    call["Conversation"] = [
        re.sub("'", "", sent) for sent in call["Conversation"]
    ]
    call_words = list(map(gensim.utils.simple_preprocess,
                          call["Conversation"]))

    # BO's n-gram generator
    def make_n_grams(texts):
        bigram = gensim.models.Phrases(
            texts, min_count=5,
            threshold=100)  # higher threshold fewer phrases.
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram = gensim.models.Phrases(bigram[texts], threshold=100)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        bigrams_text = [bigram_mod[doc] for doc in texts]
        trigrams_text = [trigram_mod[bigram_mod[doc]] for doc in bigrams_text]
        return trigrams_text

    ngrams = make_n_grams(call_words)

    # BO's stopword removal (gensim stopwords combined with BO's list)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def remove_stopwords(texts):
        return [[
            word for word in simple_preprocess(str(doc))
            if word not in gensim.parsing.preprocessing.STOPWORDS.union(
                set(stop_words))
        ] for doc in texts]

    # BO's lemmatisation function
    def lemmatise(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([
                token.lemma_ for token in doc if token.pos_ in allowed_postags
            ])
        return texts_out

    call_lem = lemmatise(ngrams,
                         allowed_postags=["NOUN", "VERB", "ADJ", "ADV"])
    call_lem = remove_stopwords(call_lem)

    return (text, " ".join(reduce(iconcat, call_lem, [])))
