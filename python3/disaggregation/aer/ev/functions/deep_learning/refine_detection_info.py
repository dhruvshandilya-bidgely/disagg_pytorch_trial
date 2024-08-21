"""
Author - Sahana M
Date - 14-Nov-2023
Module which is called to refine the outputs obtained from the machine learning model and deep learning model
"""

# Import python packages
import numpy as np


def combine_detection_info(ml_debug, dl_debug):
    """
    Function to combine the final confidences
    Parameters:
        ml_debug                     (Dict)              : Machine learning Debug dictionary
        dl_debug                     (Dict)              : Deep learning Debug dictionary
    Returns:
        debug                       (Dict)               : Machine learning Debug dictionary
    """

    # Extract the required variables
    combined_confidence = dl_debug.get('combined_conf')
    combined_charger_type = dl_debug.get('combined_charger_type')

    # assign the final hld and confidences

    if combined_charger_type == 'L2':
        ml_debug['ev_hld'] = 1
        ml_debug['ev_probability'] = np.round(combined_confidence, 3)
        ml_debug['charger_type'] = 'L2'
    elif combined_charger_type == 'L1':
        ml_debug['ev_hld'] = 1
        ml_debug['ev_probability'] = np.round(combined_confidence, 3)
        ml_debug['charger_type'] = 'L1'
    else:
        ml_debug['ev_hld'] = 0
        ml_debug['ev_probability'] = np.round(combined_confidence, 3)
        ml_debug['charger_type'] = 'None'

    return ml_debug
