from typing import List

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
    ranks = relevance[:k]
    sorted_relevance = sorted(relevance, reverse=True)
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
relevance = [0.99, 0.94, 0.74, 0.88, 0.71, 0.68]
k = 5
method = 'standard'
print(normalized_dcg(relevance, k, method))

