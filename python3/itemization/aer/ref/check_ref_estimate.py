"""
Author - Nisha Agarwal
Date - 10/9/20
Check final estimated value
"""

# Import python packages

import numpy as np


def check_ref_estimate(day_estimate, item_input_object, ref_config):

    """
    Check final estimated value using decision graph

    Parameters:
        day_estimate            (int)              : Estimated day level ref output
        item_input_object     (dict)             : Dictionary containing all inputs
        ref_config              (dict)             : Dictionary containing all information

    Returns:
        day_estimate            (int)   `          : Modified day level ref output
    """

    # Safety checks for ref day level consumption

    if day_estimate == 0:
        return day_estimate

    day_estimate = np.maximum(day_estimate, ref_config.get("limits").get("min_limit"))

    day_estimate = np.minimum(day_estimate, ref_config.get("limits").get("max_limit"))

    # Safety checks based on meta features

    if item_input_object.get("home_meta_data").get("bedrooms") is not None and \
            item_input_object.get("home_meta_data").get("bedrooms") >= ref_config.get("limits").get("bedrooms_limit"):
        day_estimate = np.maximum(day_estimate, ref_config.get("limits").get("meta_data_limit"))

    if item_input_object.get("home_meta_data").get("numOccupants") is not None and \
            item_input_object.get("home_meta_data").get("numOccupants") >= ref_config.get("limits").get("occupants_limit"):
        day_estimate = np.maximum(day_estimate, ref_config.get("limits").get("meta_data_limit"))

    return day_estimate
