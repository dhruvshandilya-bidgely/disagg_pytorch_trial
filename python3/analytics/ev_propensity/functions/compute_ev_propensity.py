"""
Author - Paras Tehria
Date - 27-May-2021
This module is used to run the EV propensity module
"""

# Import python packages

import logging
import numpy as np

# Import functions from within the project

from python3.analytics.ev_propensity.functions.get_model_input_features import get_model_input_features
from python3.analytics.ev_propensity.functions.apply_post_process_rules import apply_post_process_rules


def get_propensity_from_model(input_features, model):
    """
    This is the function that uses EV propensity model to output propensity scored based on the input_features
    Parameters:
        input_features        (list)             : Input features to propensity model
        model                 (ABCMeta)          : EV propensity model
    Return:
        ev_propensity      (float)               : output propensity score
    """
    input_x = np.array(input_features).astype(float).reshape(1, -1)

    propensity_score_output = model.predict_proba(input_x)

    ev_propensity = propensity_score_output[0, 1]
    return ev_propensity


def compute_ev_propensity(ev_propensity_config, debug, logger_base):
    """
    This is the main function that computes EV propensity for a user
    Parameters:
        ev_propensity_config          (dict)             : config dict for EV propensity module
        debug                         (dict)             : Debug object for EV propensity
        logger_base                   (dict)             : logger object to generate logs
    Return:
        debug                         (dict)             : Debug object for EV propensity
    """
    # Initializing new logger child compute_ev_propensity

    logger_local = logger_base.get('logger_base').getChild('compute_ev_propensity')

    # Initializing new logger pass to be used by the internal functions of EV propensity

    logger_pass = {'logger_base': logger_local, 'logging_dict': logger_base.get('logging_dict')}

    # logger

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    logger.debug("Generating input features to be used for propensity computation | ")
    debug, input_feature_ls = get_model_input_features(debug, ev_propensity_config, logger_pass)

    ev_prop_model = debug.get('propensity_model')

    # Get ev propensity score from the trained random forest model
    ev_propensity_score = get_propensity_from_model(input_feature_ls, ev_prop_model)
    logger.info("EV propensity score from model | {}".format(ev_propensity_score))

    # Using meta data and solar presence to boost the ev propensity
    ev_propensity_score = apply_post_process_rules(ev_propensity_score, debug, ev_propensity_config, logger)
    logger.info("Final EV propensity score | {}".format(ev_propensity_score))

    debug['ev_propensity_score'] = np.round(float(ev_propensity_score), 2)

    return debug
