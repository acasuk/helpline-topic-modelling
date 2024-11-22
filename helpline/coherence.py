"""
Implementation of several measures of topic coherence from scratch,
because this is available neither for BERTopic nor for Top2Vec.

Algorithms are drawn from section 2 of
https://arxiv.org/pdf/2305.14587
"""

import math, warnings
import numpy as np


def p_cooccurrence(w_i, w_j, document, window_size=2):
    assert window_size <= len(document)
    assert window_size > 1
    window_size -= 1
    windows = [
        document[i : i + window_size] for i in range(len(document) - window_size)
    ]
    return sum(
        list(map(lambda window: int((w_i in window) and (w_j in window)), windows))
    ) / len(windows)


def pmi_k(w_i, w_j, topic, k=1, window_size=2, epsilon=1):
    prob_w_i = topic.count(w_i) / len(topic)
    prob_w_j = topic.count(w_j) / len(topic)
    return math.log2(
        ((p_cooccurrence(w_i, w_j, topic, window_size=window_size) ^ k) + epsilon)
        / (prob_w_i * prob_w_j)
    )


def uci_single_topic(topic, epsilon=1):
    m = len(topic)
    pmis = [
        pmi_k(topic[i], topic[i + 1], topic, k=1, window_size=2, epsilon=epsilon)
        for i in range(m - 1)
    ]
    return sum(pmis) / (m / 2)


def uci_all_topics(topics, epsilon=1):
    n = len(topics)
    return sum(list(map(lambda t: uci_single_topic(t, epsilon=epsilon), topics))) / n


def umass(w_r, w_s, topic):
    prob_w_s = topic.count(w_s) / len(topic)
    return math.log(
        (p_cooccurrence(w_r, w_s, topic, window_size=2) + epsilon) / prob_w_s
    )


def npmi_k(w_r, w_s, topic, k=1, window_size=2, epsilon=1):
    if epsilon == 1:
        warnings.warn("NPMI: Recommend smaller epsilon values than default 1")
    pmi_r_s = pmi_k(w_r, w_s, topic, k=k, epsilon=epsilon)
    return pmi_r_s / (
        -math.log2(p_cooccurrence(w_r, w_s, topic, window_size=window_size) + epsilon)
    )


def cv(w_r, w_s, k=1, window_size=110, epsilon=1):
    return npmi_k(w_r, w_s, k=k, window_size=window_size, epsilon=epsilon)


def dwr(w_r, w_s, embed_function):
    e_r = embed_function(w_r)
    e_s = embed_function(w_s)
    return np.dot(e_r, e_s) / np.dot(np.linalg.norm(e_r), np.linalg.norm(e_s))
