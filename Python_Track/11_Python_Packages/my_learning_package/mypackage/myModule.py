from heapq import nlargest
from typing import Iterable


def top_n(items, n):
    """
    Return the top n items in an array, in decending order.

    Args:
        items (array): list or array-like object containing numerical values.
        n (int): number of top items to return.

    Returns:
        array: top n items, in decending order.

    Examples:
        >>> top_n([8, 3, 2, 7, 4])
        [8, 7, 4]
    """

    for i in range(n):
        for j in range(len(items) -1-i):

            if items[j] > items[j+1]:
                items[j], items[j+1] = items[j+1], items[j]
        
    # Get the last items and return them
    top_n = items[-n:]

    return top_n[::-1]


def top_n_largest(items: Iterable[float], n: int) -> list[float]:
    """
    Return the largest n values in descending order.

    Args:
        items (Iterable[float]): values to search.
        n (int): number of values to return.

    Returns:
        list[float]: largest values in descending order.
    """
    if n <= 0:
        return []

    return nlargest(n, items)

