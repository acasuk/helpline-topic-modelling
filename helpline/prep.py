"""
Utility functions to prepare large numbers of documents.
"""

import os
import pickle
import pandas as pd
import nltk

# Install/Update NLTK punkt package
nltk.download("punkt")

def _list_files(directory):
    """
    TODO
    """
    return [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(directory)) for f in fn]

def lemmatise_call(filename):
    """
    Lemmatise a text TODO
    """
    # Load BO's stopwords list
    with open("BO_stopwords.pkl", "rb") as infile:
        stop_words = pickle.load(infile) + [ "remove" ]  # To cover e.g.  "person removed", "name removed", "phone removed"

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

    return (text, call_lem, " ".join(reduce(iconcat, call_lem, [])))

def pickle_calls(files,outfile_prefix):
    """
    TODO
    """
    i = 0
    docs_raw = []
    docs_processed = []

    for f in files:
        i=i+1
        print(str(i)+" / "+str(len(files))+":\tProcessing:\t"+f)
        doc = lemmatise_call(f)
        docs_raw.append(doc[0])
        docs_processed.append(doc[2])
        print(str(i)+" / "+str(len(files))+":\tFinished:\t"+f)

    print("\nRaw:\t\t"+str(len(docs_raw))+" documents")
    print("Processed:\t"+str(len(docs_processed))+" documents\n")

    print("Pickling raw docs...")
    with open(outfile_prefix+"_raw_docs.pkl","wb") as f:
        pickle.dump(docs_raw,f,pickle.HIGHEST_PROTOCOL)
    print("Pickling processed docs...")
    with open(outfile_prefix+"_processed_docs.pkl","wb") as f:
        pickle.dump(docs_processed,f,pickle.HIGHEST_PROTOCOL)

    with open(outfile_prefix+"_raw_docs.pkl","rb") as f:
        assert len(pickle.load(f))==len(files)
    with open(outfile_prefix+"_processed_docs.pkl","rb") as f:
        assert len(pickle.load(f))==len(files)

    print("Finished!")
