from typing import List

import numpy as np


def cumulative_gain(relevance: List[float], k: int) -> float:
    """Score is cumulative gain at k (CG@k)

    Parameters
    ----------
    relevance:  `List[float]`
        Relevance labels (Ranks)
    k : `int`
        Number of elements to be counted

    Returns
    -------
    score : float
    """
    first_k_ranks = relevance[:k]
    score = np.sum(first_k_ranks)
    return score
