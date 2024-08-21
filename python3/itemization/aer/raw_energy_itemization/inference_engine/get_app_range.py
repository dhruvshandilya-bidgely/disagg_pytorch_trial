
"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Initialize min and max ranges of appliance consumption
"""

# Import python packages

import copy
import numpy as np


def initialize_app_range(app_cons, app_conf, app_pot):

    """
    Initialize appliance ranges

    Parameters:
        app_cons            (np.ndarray)        : appliance ts level consumption
        app_conf            (np.ndarray)        : appliance ts level confidence
        app_pot             (np.ndarray)        : appliance ts level potential

    Returns:
        min_range           (np.ndarray)        : appliance ts level minimum consumption
        mid_range           (np.ndarray)        : appliance ts level mid level consumption
        max_range           (np.ndarray)        : appliance ts level maximum consumption
        app_cons            (np.ndarray)        : appliance ts level consumption
        app_conf            (np.ndarray)        : appliance ts level confidence
        app_pot             (np.ndarray)        : appliance ts level potential
    """

    app_cons = np.swapaxes(app_cons, 0, 2)
    app_cons = np.swapaxes(app_cons, 0, 1)

    app_conf = np.swapaxes(app_conf, 0, 2)
    app_conf = np.swapaxes(app_conf, 0, 1)

    app_pot = np.swapaxes(app_pot, 0, 2)
    app_pot = np.swapaxes(app_pot, 0, 1)

    max_range = copy.deepcopy(app_cons)
    mid_range = copy.deepcopy(app_cons)
    min_range = copy.deepcopy(app_cons)

    return  np.fmax(0, min_range), np.fmax(0, mid_range), np.fmax(0, max_range), app_conf, app_pot, app_cons
