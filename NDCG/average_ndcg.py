from typing import List

from functools import partial

import numpy as np

def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Normalized Discounted Cumulative Gain.

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    sorted_relevance = sorted(relevance, reverse=True)
    if k > len(relevance):
        k = len(relevance)
    ranks = relevance[:k]
    sorted_ranks = sorted_relevance[:k]
    gains = []
    sorted_gains = []
    if method == 'standard':
        for i in range(k):
            gains.append(ranks[i] / np.log2(2+i))
            sorted_gains.append(sorted_ranks[i] / np.log2(2+i))
    elif method == 'industry':
        for i in range(k):
            gains.append((2 ** ranks[i] - 1) / np.log2(2+i))
            sorted_gains.append((2 ** sorted_ranks[i] - 1) / np.log2(2+i))
    else:
        raise ValueError()
    dcg = np.sum(gains)
    ideal_dcg = np.sum(sorted_gains)
    score = dcg / ideal_dcg
    return score

def avg_ndcg(list_relevances: List[List[float]], k: int, method: str = 'standard') -> float:
    """Average nDCG

    Parameters
    ----------
    list_relevances : `List[List[float]]`
        Video relevance matrix for various queries
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values ​​\
        `standard` - adds weight to the denominator\
        `industry` - adds weights to the numerator and denominator\
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    mapfunc = partial(normalized_dcg, k = k, method = method)
    list_ndcg = map(mapfunc, list_relevances)
    score = sum(list_ndcg) / len(list_relevances)
    return score

