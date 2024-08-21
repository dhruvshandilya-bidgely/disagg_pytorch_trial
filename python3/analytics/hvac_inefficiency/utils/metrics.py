"""
Author  -   Anand Kumar Singh
Date    -   26th June 2021
This file contains code for finding saturation temperature
"""

# Import python packages

import numpy as np


def median_absolute_error(x, y):
    """
        This function estimates median absolute error

        Parameters:
            x       (np.ndarray)            array of expected values
            y       (np.ndarray)            array of predicted values
        Returns:
            mae     (float)                 median error
    """
    error_distribution = np.abs(x - y)
    mae = np.median(error_distribution)
    return mae
