"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module with calls for EV detection function
"""

# Import python packages
import logging
from copy import deepcopy
# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.ev.functions.remove_baseload import remove_baseload
from python3.disaggregation.aer.ev.functions.detection.dynamic_box_fitting import dynamic_box_fitting
from python3.disaggregation.aer.ev.functions.detection.remove_missing_data import remove_missing_data
from python3.disaggregation.aer.ev.functions.detection.remove_hvac import remove_hvac_using_potentials
from python3.disaggregation.aer.ev.functions.detection.append_missing_data import append_missing_data

from python3.disaggregation.aer.ev.functions.detection.get_user_features import get_user_features
from python3.disaggregation.aer.ev.functions.detection.get_recent_user_features import get_recent_user_features
from python3.disaggregation.aer.ev.functions.detection.home_level_detection import home_level_detection


def ev_detection(in_data, ev_config, debug, error_list, logger_base):
    """
    Function for EV detection

    Parameters:
        in_data             (np.ndarray)            : Input matrix
        ev_config            (dict)                  : Configuration for the algorithm
        debug               (dict)                  : Dictionary containing output of each step
        error_list          (list)                  : The list of handled errors
        logger_base         (logger)                : The logger object

    Returns:
        debug               (dict)                  : Output at each algo step
        error_list          (list)                  : List of handled errors encountered
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('ev_detection')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking local copy of the input data

    input_data = deepcopy(in_data)

    # Removing missing data

    input_data, debug = remove_missing_data(input_data, debug, ev_config, logger_pass)

    # Remove baseload consumption from the input energy data

    input_data, debug = remove_baseload(input_data, debug, ev_config, logger_pass)

    # Check the disagg mode, if MTD then use hsm

    if ev_config['disagg_mode'] == 'mtd':
        hsm_in = debug['hsm_in']
        logger.info("EV mtd run input hsm info: home level detection | {}".format(hsm_in['ev_hld']))
        logger.info("EV mtd run input hsm info: detection probability | {}".format(hsm_in['ev_probability']))

        ev_hld = hsm_in['ev_hld'][0]
        ev_probability = hsm_in['ev_probability'][0]

        debug['ev_hld'] = ev_hld
        debug['ev_probability'] = ev_probability
        debug['charger_type'] = hsm_in.get('charger_type')

        features_box_data = deepcopy(input_data)
        features_box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0
        debug['features_box_data'] = features_box_data

        if debug['charger_type'] == 'L1':
            debug['l1']['features_box_data'] = features_box_data

        debug = append_missing_data(debug)
        return debug, error_list

    # Remove HVAC

    input_data, debug = remove_hvac_using_potentials(input_data, debug, ev_config, logger_pass)
    debug['hvac_removed_data_l2'] = deepcopy(input_data)

    # Dynamic box fitting for finding optimum energy amplitude boxes

    debug = dynamic_box_fitting(input_data, ev_config, debug, logger_pass)

    # Creating the detection features of overall data

    debug = get_user_features(debug, ev_config, input_data)

    # Creating the detection features of recent data

    debug = get_recent_user_features(debug, ev_config, logger_pass, input_data)

    # Using saved model to detect the EV

    debug = home_level_detection(debug, ev_config, logger_pass)

    # Append back the missing data to the data used for detection

    debug = append_missing_data(debug)

    return debug, error_list
