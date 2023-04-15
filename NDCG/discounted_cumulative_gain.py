from typing import List

import numpy as np


def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values​​
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    ranks = relevance[:k]
    gains = []
    if method == 'standard':
        for i in range(k):
            gains.append(ranks[i] / np.log2(2+i))
        score = np.sum(gains)
    elif method == 'industry':
        for i in range(k):
            gains.append((2 ** ranks[i] - 1) / np.log2(2+i))
        score = np.sum(gains)
    else:
        raise ValueError()
    return score

relevance = [0.99, 0.94, 0.88, 0.74, 0.71, 0.68]
k = 5
method = 'standard'
print(discounted_cumulative_gain(relevance, k, method))

