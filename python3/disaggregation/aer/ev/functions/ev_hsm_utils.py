"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module contains operations related to HSM
"""

# Import python packages

import logging

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_ev_hsm(disagg_input_object, global_config, logger_base):
    """
    Parameters:
        disagg_input_object     (dict)          : Dictionary containing all inputs
        global_config           (dict)           : Dictionary containing all input configuration
        logger_base             (dict)          : Logger object

    Returns:
        hsm_in                  (dict)          : Electric Vehicle HSM (Home Specific Model)
        hsm_fail                (dict)          : If valid hsm is present for current run mode
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_hsm')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Try extracting the hsm from input object

    try:
        hsm_dict = disagg_input_object.get('appliances_hsm')
        hsm_in = hsm_dict.get('ev')

        logger.info('HSM retrieved from the disagg input object | ')
    except KeyError:
        # If hsm not found

        hsm_in = None
        logger.info('HSM not found in the disagg input object | ')

    # Get the disagg mode from the global config

    disagg_mode = global_config.get('disagg_mode')

    # Check if HSM attributes present for MTD mode

    hsm_fail = ((hsm_in is None) or (len(hsm_in) == 0) or (len(hsm_in.get('attributes')) == 0)) and \
               (disagg_mode == 'mtd')

    # If HSM is not None, then update the charger type

    if (disagg_mode == 'mtd') and not hsm_fail:
        if hsm_in.get('attributes').get('charger_type') == [1]:
            hsm_in['attributes']['charger_type'] = 'L1'
        elif hsm_in.get('attributes').get('charger_type') == [2]:
            hsm_in['attributes']['charger_type'] = 'L2'
        else:
            hsm_in['attributes']['charger_type'] = 'None'

    return hsm_in, hsm_fail


def make_hsm_attributes(debug, ev_config, logger):
    """
    Parameters:
        debug           (dict)                  : Contains data saved at each step of algorithm
        ev_config       (dict)                  : The config params for the algorithm
        logger          (logger)                : The logger object to write logs

    Returns:
        attributes      (dict)                  : Dictionary with hsm parameters
    """

    # Initialize the hsm attributes with hld values
    # ev_hld            : EV detection flag
    # ev_amplitude      : EV amplitude
    # ev_probability    : EV detection probability

    attributes = {
        'ev_hld': debug['ev_hld'],
        'ev_amplitude': debug['ev_amplitude'],
        'ev_probability': debug['ev_probability'],
        'recent_ev': debug['recent_ev'],
        'run_eu_model': debug['run_eu_model_hsm']
    }

    if debug['charger_type'] == 'L1':
        attributes['charger_type'] = 1
    elif debug['charger_type'] == 'L2':
        attributes['charger_type'] = 2
    else:
        attributes['charger_type'] = 0

    # Add values to hsm based on detection of EV

    if debug['ev_hld'] == 1:
        attributes['mean_duration'] = debug['mean_duration']
        attributes['lower_amplitude'] = debug['lower_amplitude']
        attributes['max_energy_allowed'] = debug['max_energy_allowed']
        attributes['max_deviation_allowed'] = debug['max_deviation_allowed']
        attributes['clean_box_tou'] = debug['clean_box_tou']
        attributes['clean_box_amp'] = debug['clean_box_amp']
        attributes['clean_box_auc'] = debug['clean_box_auc']
        attributes['clean_box_dur'] = debug['clean_box_dur']

    return attributes


def check_hsm_validity(debug, logger_base):
    """
    Check if a valid hsm present for MTD mode

    Parameters:
        debug           (dict)              : Algorithm intermediate steps output
        logger_base     (dict)              : Logger object

    Returns:
        valid_hsm       (bool)              : The boolean to mark validity of the HSM
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('check_hsm_validity')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve disagg_mode and hsm from debug object

    hsm_in = debug.get('hsm_in')
    disagg_mode = debug.get('disagg_mode')

    # Initialize default hsm validity to True

    valid_hsm = True

    # If the disagg mode is MTD

    if disagg_mode == 'mtd':
        # If the HSM is not None, check for ev amplitude values

        if hsm_in is not None:
            # If valid hsm present, check for EV amplitude value

            ev_hld = hsm_in.get('ev_hld')

            # HSM valid if EV detection value found

            if ev_hld is None:
                # Invalid EV detection value

                valid_hsm = False

                logger.info('HSM invalid because no valid EV amplitude values found | ')
        else:
            # If HSM None in MTD mode

            valid_hsm = False

            logger.info('HSM invalid because HSM is None for mode {} | '.format(disagg_mode))

    return valid_hsm


def make_hsm_from_debug(input_data, debug, ev_config, error_list, logger):
    """
    Parameters:
        input_data      (np.ndarray)            : Raw data input for the user
        debug           (dict)                  : The dictionary containing all module level output
        ev_config       (dict)                  : Water heater parameters
        error_list      (list)                  : List of errors encountered in the code run
        logger          (logger)                : The logger object to write logs

    Returns:
        ev_hsm          (dict)                  : The new hsm dictionary
        debug           (dict)                  : Updated debug object
        error_list      (list)                  : The list of errors
    """

    ev_hsm = dict({})

    # If the mode is historical/incremental, make hsm

    if debug['make_hsm']:
        # Extract the relevant values from debug dict to create hsm

        ev_hsm = {
            'timestamp': input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': make_hsm_attributes(debug, ev_config, logger)
        }

        # Saving new HSM to the debug object

        debug['hsm'] = ev_hsm

        logger.info('Writing new HSM for the run | ')
    else:
        logger.info('Not writing HSM for this mode | ')

    return ev_hsm, debug, error_list
