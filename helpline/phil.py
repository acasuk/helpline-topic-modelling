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
from itertools import repeat
from functools import reduce
from operator import iconcat


def gsdmm_call(filename,
               n_topics=10,
               alpha=0.01,
               beta=0.01,
               n_iters=30,
               threshold=0.3):  # BO's values by default
    """
    TODO
    """
    call_lem = prep.lemmatise_call(filename)[1]

    np.random.seed(42)

    mgp = MovieGroupProcess(
        K=n_topics, alpha=alpha, beta=beta,
        n_iters=n_iters)  # TODO: experiment with other values
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


def cwd_to_list(cwd):
    """
    TODO
    """
    return reduce(
        iconcat,
        list(
            map(lambda kv: repeat(kv[0], kv[1]), zip(cwd.keys(),
                                                     cwd.values()))),
        [],
    )


def centroid(points):
    """
    TODO
    """
    return np.average(np.stack(points),
                      axis=0)  # TODO: Rewrite in Futhark for speedup


def get_topic_centroids(mgp, model):
    """
    TODO
    """
    return list(
        map(
            lambda w: centroid(
                list(map(lambda w: model.get_vector(w), cwd_to_list(w)))),
            mgp.cluster_word_distribution,
        ))


def get_topic_centroid_words(mgp, model):
    """
    TODO
    """
    return list(
        map(lambda v: model.similar_by_vector(v, topn=1)[0],
            get_topic_centroids(mgp)))


def vector_to_word(v, model):
    """
    TODO
    """
    return model.similar_by_vector(v, topn=1)[0]


def topic_to_vectors_with_centroids(mgp, model):
    """
    TODO
    """
    topics = list(map(cwd_to_list, mgp.cluster_word_distribution))
    topics = list(filter(lambda l: l != [], topics))  # Remove empty topics
    for t in range(len(topics)):
        for w in range(len(topics[t])):
            # print(str(t)+": "+topics[t][w])
            try:
                topics[t][w] = model.get_vector(topics[t][w])
            except (
                    KeyError
            ):  # Was complaining when hitting a word with no vector ("manaja" in this case)
                topics[t][w] = None  # If word not found, return None

    # Remove any topics with all Nones
    dodgy = []
    for i in range(len(topics)):
        if all(list(map(lambda t: isinstance(t, type(None)), topics[i]))):
            dodgy.append(i)
    for i in dodgy:
        del topics[i]

    # Replace Nones with centroids
    for t in topics:
        vs = list(filter(lambda i: not (isinstance(i, type(None))), t))
        c = centroid(vs)
        for w in range(len(t)):
            if isinstance(t[w], type(None)):
                t[w] = c

    return topics
