"""
Author - Nikhil Singh Chauhan
Date - 16/10/18
Module with calls for EV detection function
"""

# Import python packages

import logging
import numpy as np
import pandas as pd
from copy import deepcopy

from python3.disaggregation.aer.ev.functions.detection.seasonality_checks import winter_seasonality_check


def home_level_detection(debug, ev_config, logger_base):
    """
    Perform EV L1 home level detection
    Parameters:
        debug                   (Dict)              : Debug dictionary
        ev_config               (Dict)              : EV configurations
        logger_base             (Logger)            : Logger
    Returns:
        debug                   (Dict)              : Debug dictionary
    """
    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('home_level_detection')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    recent_ev = 0

    all_probabilities = []

    # Extract the EV model from debug object

    hld_model = debug['models']['ev_hld']['l1']

    model_features = ev_config['model_features']

    # Use the complete data features

    user_features = deepcopy(debug['l1']['user_features'])

    user_features_df = pd.DataFrame(user_features, index=[0])

    user_model_features = model_features['l1']

    user_features_df = user_features_df[user_model_features]

    # Replace nan with zero

    user_features_df = user_features_df.replace(np.nan, 0)
    user_features_df = user_features_df.replace([np.inf, -np.inf], 5)

    # Logging features values

    for feature in user_model_features:
        feature_value = user_features[feature]

        logger.info('EV L1 Feature {} | {}'.format(feature, feature_value))

    ev_probability = hld_model.predict_proba(user_features_df)[0, 1]
    all_probabilities.append(ev_probability)

    logger.info('EV L1 detection probability | {}'.format(ev_probability))

    if ev_probability >= 0.50:
        ev_hld = 1
    else:
        ev_hld = 0

    # HLD Post processing checks to avoid False positives

    if ev_hld != 0 and ev_config.get('pilot_id') in ev_config.get('nsp_winter_seasonality_configs').get('wtr_seasonal_pilots'):

        winter_device, debug = winter_seasonality_check(debug['l1']['features_box_data'], ev_config, debug)

        if winter_device:
            ev_probability = 0.3
            ev_hld = 0

            logger.info('Winter WH/HVAC device FP detected in EV, changing the detection to 0 | ')

    # If not found use the recent features

    logger.info('EV L1 home level detection | {}'.format(ev_hld))
    logger.info('EV L1 detection probability | {}'.format(ev_probability))

    debug['l1']['ev_hld'] = ev_hld
    debug['l1']['ev_probability'] = round(float(ev_probability), 4)
    debug['l1']['recent_ev'] = recent_ev

    if ev_hld == 1:
        debug['ev_hld'] = ev_hld
        debug['ev_probability'] = round(float(ev_probability), 4)
        debug['charger_type'] = 'L1'
    else:
        debug['charger_type'] = 'None'

    return debug
