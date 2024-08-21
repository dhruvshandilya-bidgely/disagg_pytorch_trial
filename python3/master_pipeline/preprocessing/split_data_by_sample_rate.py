"""
Author - Mayank Sharan
Date - 25/09/18
Returns the split of the 21 column data by sampling rate
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.find_seq import find_seq


def split_data_by_sample_rate(input_data):

    """
    Parameters:
        input_data          (np.ndarray)        : 21 column input data

    Returns:
        seq_arr             (np.ndarray)        : 4 column output of merged sampling rate chunks
    """

    # Find the sequences of sampling rates in the data

    seq_arr = find_seq(np.diff(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]), 0)

    # Fix final indices since the indices are from a diff array

    seq_arr[:, 2] += 1

    return seq_arr
