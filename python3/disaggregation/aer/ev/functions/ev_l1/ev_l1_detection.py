"""
Author -  Paras Tehria / Sahana M
Date - 1-Feb-2022
Module with calls for EV detection function
"""

# Import python packages

import logging
from copy import deepcopy

# Import functions from within the project

from python3.disaggregation.aer.ev.functions.remove_baseload import remove_baseload
from python3.disaggregation.aer.ev.functions.ev_l1.dynamic_box_fitting import dynamic_box_fitting
from python3.disaggregation.aer.ev.functions.detection.remove_missing_data import remove_missing_data
from python3.disaggregation.aer.ev.functions.ev_l1.append_missing_data import append_missing_data
from python3.disaggregation.aer.ev.functions.ev_l1.get_user_features import get_user_features
from python3.disaggregation.aer.ev.functions.ev_l1.home_level_detection import home_level_detection
from python3.disaggregation.aer.ev.functions.detection.remove_hvac import remove_hvac_using_potentials


def ev_l1_detection(in_data, ev_config, debug, error_list, logger_base):
    """
    Function to perform L1 detection
    Parameters:
        in_data                 (np.ndarray)        : Input data
        ev_config               (Dict)              : EV configurations
        debug                   (Dict)              : Debug dictionary
        error_list              (Dict)              : Error list
        logger_base             (Logger)            : Logger
    Returns:
        debug                   (Dict)              : Debug dictionary
        error_list              (Dict)              : Error list
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('ev_l1_detection')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking local copy of the input data

    input_data = deepcopy(in_data)

    # Removing missing data

    input_data, debug = remove_missing_data(input_data, debug, ev_config, logger_pass)

    # Remove baseload consumption from the input energy data

    input_data, debug = remove_baseload(input_data, debug, ev_config, logger_pass)

    # Remove HVAC using HVAC potentials

    input_data, debug = remove_hvac_using_potentials(input_data, debug, ev_config, logger_pass, 'l1')
    debug['hvac_removed_data_l1'] = deepcopy(input_data)

    # Dynamic box fitting for finding optimum energy amplitude boxes

    debug = dynamic_box_fitting(input_data, ev_config, debug, logger_pass)

    # Create user features using the final boxes

    debug = get_user_features(debug, ev_config)

    # Use saved model to detect the EV

    debug = home_level_detection(debug, ev_config, logger_pass)

    # Append back the missing data to the data used for detection

    debug = append_missing_data(debug)

    logger.info("Completed EV L1 detection | ")

    return debug, error_list
