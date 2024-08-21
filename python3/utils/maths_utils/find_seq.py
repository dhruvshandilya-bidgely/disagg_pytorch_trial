"""
Author - Mayank Sharan
Date - 25/09/18
Return 4 column matrix with details of consecutive sequence of integers in a list
"""

# Import python packages

import numpy as np
from itertools import groupby


def find_seq(arr, min_seq_length=1):

    """
    Parameters:
        arr                 (np.ndarray)        : Array in which to find the sequences
        min_seq_length      (int)               : Set as the smallest length of the sequence to be considered a chunk

    Returns:
        res                 (np.ndarray)        : 4 column result, integer, start_idx, end_idx, num_in_seq
    """

    # Initialise the result array

    res = []
    start_idx = 0

    # Get groups

    group_list = groupby(arr)

    for seq_num, seq in group_list:

        # Get number of elements in the sequence

        seq_len = len(list(seq))

        # Discard single elements since they are not a sequence

        if seq_len <= min_seq_length:
            start_idx += seq_len
            continue
        else:
            temp_res = [seq_num, start_idx, start_idx + seq_len - 1, seq_len]
            start_idx += seq_len
            res.append(temp_res)

    return np.array(res)
