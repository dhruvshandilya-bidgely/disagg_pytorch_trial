"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
This module takes the input data and create features which are further fed to a pre-trained decision tree
to determine whether the user has a water heater or not (at a monthly level)
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy
from datetime import datetime

# Import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff

from python3.disaggregation.aer.waterheater.functions.thermostat.functions.model_features import train
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.hld_review import hld_review
from python3.disaggregation.aer.waterheater.functions.get_seasonal_segments import get_seasonal_segments
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.get_user_features import get_user_features
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.get_seasonal_features import get_seasonal_features
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.override_timed_wh_detection import override_timed_wh_detection


def thermostat_waterheater_detection(input, debug, wh_config, logger_base):
    """
    This module makes the features required for and home level detection of water heater and then
    using a pre trained decision tree returns the output if the user has a Water heater or not

    Parameters:
        debug           (dict)          : Debug dictionary with values recorded throughout the code run
        wh_config       (dict)          : parameters dictionary
        logger_base     (logger)        : logger object to save logs

    Returns:
        debug           (dict)          : Updated debug dict with Home Level Detection for the user (1 / 0)
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('thermostat_waterheater_detection')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Start time of detection module

    t_start = datetime.now()

    # Taking a deepcopy of input data to keep local instances

    input_data = deepcopy(input)

    # Check if MTD mode and hsm can be used to generate output

    if debug['use_hsm']:
        # If MTD run mode, get values from hsm

        debug['thermostat_hld'] = debug['hsm_in']['thermostat_hld'][0]

        # Find the season of the given data

        wtr_data, itr_data, smr_data, seasons = get_seasonal_segments(input_data, None, debug, logger_pass, wh_config,
                                                                      monthly=False, return_data=True, one_season=True)

        # Pick the latest season as the season of the given data chunk

        debug['season'] = wh_config['season_code'][int(seasons[-1, -1])]

        # Get user features for detection model

        current_season_features, debug = get_seasonal_features(wtr_data, itr_data, smr_data, wh_config, debug,
                                                               logger_pass)

    else:
        # If historical / incremental run mode

        # Get seasonal segments and data

        winter, intermediate, summer, seasons = get_seasonal_segments(input_data, None, debug, logger_pass, wh_config)

        debug['season_segments'] = seasons

        # Get user features for detection model

        season_features, debug = get_seasonal_features(winter, intermediate, summer, wh_config, debug, logger_pass)

        # Aggregate the monthly features to get user features

        features, debug = get_user_features(debug, season_features, wh_config)

        logger.info('User level feature set created successfully | ')

        # Check detection model validity

        valid_model = check_model(debug['models'])

        # Get home level detection

        if valid_model:
            # If valid model, go for prediction

            # Extract hld model

            hld_model = debug['models']['hld_model']

            # Get detection along with probability

            hld, hld_prob, debug = get_detection(features, wh_config, hld_model, debug, logger_pass)

            # Add features to the debug object

            debug['all_features'] = season_features
            debug['features'] = features

            debug['thermostat_hld'] = hld
            debug['thermostat_hld_prob'] = np.round(hld_prob, 4)
        else:
            # If model file not found, give default detection zero

            logger.info('HLD model file does not exist | ')

            debug['thermostat_hld'] = 0

            return debug

    # End time of detection module

    t_end = datetime.now()

    logger.info('Non-timed water heater detection | {}'.format(debug['thermostat_hld']))

    # Log the runtime of detection

    logger.info('Detection module time in seconds | {}'.format(get_time_diff(t_start, t_end)))

    return debug


def check_model(model_dict):
    """
    Parameters:
        model_dict          (dict)      : Model dictionary

    Returns:
        valid_model         (bool)      : Validity boolean
    """

    # Extract the hld model and make sure it is not None

    if (model_dict.get('hld_model') is None) or (len(model_dict.get('hld_model')) == 0):
        valid_model = False
    else:
        valid_model = True

    return valid_model


def get_detection(features, wh_config, model, debug, logger_base):
    """
    Parameters:
        features            (dict)          : User features
        model               (object)        : Model object to make prediction
        logger_base         (logger)        : Logger object

    Returns:
        hld                 (int)           : Home level detection
        hld_probability     (float)         : Detection probability
        debug               (dict)          : Algorithm intermediate steps output
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_detection')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Initialize the features array

    pred_features = np.zeros(shape=(1, len(train)))

    logger.info('The number of features | {}'.format(len(train)))
    logger.info('The detection features are as follows | ')

    # Extract each feature to be used for prediction

    for feat, i in zip(train, range(len(train))):
        # Populate the features array

        pred_features[0, i] = features[feat]

        logger.info('{} | {}'.format(feat, features[feat]))

    # Replace NaN with zero

    pred_features[np.isnan(pred_features)] = 0

    # Get detection probability

    hld_probability = model.predict_proba(pred_features)[0][1]

    # Based on probability and thin pulse features, give final hld

    hld, debug = hld_review(features, hld_probability, wh_config, debug, logger)

    logger.info('Water heater probability | {}'.format(hld_probability))

    # Check if timed water heater to be overwritten by non-timed

    if (hld == 1) and (debug['timed_hld'] == 1):
        # Check if both timed and non-timed water heater detected

        logger.info('Both timed and non-timed water heater detected | ')

        hld, debug = override_timed_wh_detection(debug, hld_probability, wh_config, logger_pass)
    else:
        # If no conflict of waterheater detections

        logger.info('Timed water heater override not required | ')

    return hld, hld_probability, debug
