"""
Functions relating to Phil's method of topic modelling, which works as
follows:

    1. For each document, perform topic modelling at the sentence
    level using GSDMM
    2. For each document, use Word2Vec to calculate the embeddings for
    each topic's keywords
    3. Calculate the centroid of the keywords for each topic, weighted
    by the number of sentences in which the keyword appeared
    4. Cluster analysis of topic centroids from all documents to
    identify meta-topics appearing across the entire corpus
"""

import numpy as np
import helpline.prep as prep
from gsdmm import MovieGroupProcess

def gsdmm_call(filename,n_topics=10,alpha=0.01,beta=0.01,n_iters=30,threshold=0.3): # BO's values by defaul
    """
    TODO
    """
    call_lem = prep.lemmatise_call(filename)[1]

    np.random.seed(42)

    mgp = MovieGroupProcess( K=n_topics, alpha=alpha, beta=beta, n_iters=n_iters)  # TODO: experiment with other values
    vocab = set(x for line in call_lem for x in line)
    n_terms = len(vocab)
    model = mgp.fit(call_lem, n_terms)

    for i, text in enumerate(call["Conversation"]):
        p = mgp.choose_best_label(call_lem[i])
        if p[1] >= threshold:
            call.at[i, "Topic"] = "Cluster " + str(p[0])
        else:
            call.at[i, "Topic"] = "Other"

    return (mgp, call)
