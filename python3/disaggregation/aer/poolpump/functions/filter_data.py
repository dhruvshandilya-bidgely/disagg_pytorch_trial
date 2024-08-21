"""
Author - Mayank Sharan
Date - 11/1/19
Filter data by applying min max on top
"""

# Import python packages

import copy
import numpy as np
from skimage.morphology import erosion, dilation

# Import functions from within the project

from python3.disaggregation.aer.poolpump.functions.get_padded_signal import get_padded_signal


def filter_data(data_bl_removed, n_rows_pad, pp_config):
    """
    Parameters:
        data_bl_removed     (np.ndarray)        : Day wise data matrix with baseload subtracted
        n_rows_pad          (int)               : Number of rows to pad the data with for mi nmax
        pp_config           (dict)              : Dictionary containing all configuration variables for pool pump

    Returns:
        data_filtered       (np.ndarray)        : Day wise data after min max filtering
    """

    # Initialise variables for filtering

    sampling_rate = pp_config.get('sampling_rate')

    if sampling_rate == 900:
        n_cols_pad = pp_config.get('filtering_pad_cols_15_min')
    else:
        n_cols_pad = pp_config.get('filtering_pad_cols_others')

    neighbours = np.ones(shape=(n_rows_pad, n_cols_pad))
    n_rows, n_cols = data_bl_removed.shape

    # Get padded data and run min-max

    data_padded = get_padded_signal(copy.deepcopy(data_bl_removed), n_rows_pad, n_cols_pad)

    # Min max filtering

    data_min = erosion(data_padded, neighbours)
    data_min_max = dilation(data_min, neighbours)

    # Return the matrix back to original size removing the padding

    data_min_max = data_min_max[n_rows_pad: n_rows_pad + n_rows, n_cols_pad: n_cols_pad + n_cols]

    return data_min_max
