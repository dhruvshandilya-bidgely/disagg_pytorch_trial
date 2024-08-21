"""
Author - Paras Tehria
Date - 17-Nov-2020
This module is used to run the solar propensity module
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.master_pipeline.preprocessing.downsample_data import downsample_data
from python3.analytics.solar_propensity.functions.get_consum_feat import get_consum_feat
from python3.analytics.solar_propensity.functions.get_break_even_period import get_break_even_period


def get_propensity_from_model(input_features, model):
    """
    This is the function that uses solar propensity model to output propensity scored based on the input_features

    Parameters:
        input_features        (list)             : Input features to propensity model
        model                 (ABCMeta)          : solar propensity model

    Return:
        solar_propensity      (float)             : output propensity score
    """
    input_x = np.array(input_features).astype(float).reshape(1, -1)

    model.n_jobs = 1
    propensity_score_output = model.predict_proba(input_x)

    solar_propensity = propensity_score_output[0, 1]
    return solar_propensity


def compute_solar_propensity(solar_propensity_config, debug, analytics_input_object, logger_base):
    """
    This is the main function that computes solar propensity for a user

    Parameters:
        solar_propensity_config        (dict)             : global config
        debug                         (dict)             : Debug object for solar propensity
        analytics_input_object           (dict)             : disagg input object
        logger_base                   (dict)             : logger object to generate logs

    Return:
        debug                           (dict)             : Debug object for solar propensity
    """
    input_data = deepcopy(debug['input_data'])
    uuid = solar_propensity_config.get('uuid')

    # Initializing new logger child solar_propensity

    logger_local = logger_base.get('logger_base').getChild('solar_propensity')

    # Initializing new logger pass to be used by the internal functions of solar_propensity

    logger_pass = {'logger_pass': logger_local, 'logging_dict': logger_base.get('logging_dict')}

    # logger

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Downsampling the input data
    input_data = downsample_data(input_data, Cgbdisagg.SEC_IN_HOUR)

    # Not running propensity module, if a lot of negative consumption data present

    max_perc_neg_data = solar_propensity_config.get('max_perc_neg_data')
    if np.count_nonzero(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] < 0) > max_perc_neg_data * len(input_data):
        logger.debug("Negative in_data for uuid: {}. Skipping in_data prep.".format(uuid))
        debug['panel_capacity'] = None
        debug['break_even_period'] = None
        debug['propensity_score_model_1'] = None
        debug['propensity_score_model_2'] = None
        return debug

    dwelling_type = solar_propensity_config.get('dwelling')
    ownership_type = solar_propensity_config.get('ownershipType')

    logger.info("dwelling, ownership: | {}, {}".format(dwelling_type, ownership_type))

    # Getting daily savings potential array
    daily_savings_potential = get_consum_feat(input_data, solar_propensity_config)

    # Extracting useful info to be used as features for solar propensity model
    sun_50 = np.nanquantile(daily_savings_potential, 0.50)
    sun_75 = np.nanquantile(daily_savings_potential, 0.75)
    sun_std = np.std(daily_savings_potential)

    # If dwelling type is 1 and ownership is owned or not, this will also take care of nan and none values
    dwelling_type = int(dwelling_type == 1)
    ownership_type = int(dwelling_type == "Owned")
    logger.info("meta features calculated successfully |")

    # Calculating model 1 propensity score

    # Model 1 was trained using hand-tagged ground truth

    feat_arr_model_1 = [dwelling_type, ownership_type, sun_50, sun_std]

    logger.info("sun_50, sun_75, sun_std: | {}, {}, {}".format(sun_50, sun_75, sun_std))

    model_1 = analytics_input_object.get('loaded_files', {}).get('solar_files', {}).get('propensity_model').get('model_1')

    # propensity score is a probability value signifying the chances of adoption for a particular user
    propensity_score_model_1 = get_propensity_from_model(feat_arr_model_1, model_1)

    # Model 2 propensity score

    # Model 2 was trained using data of users who transitioned from non-solar to solar. Model 2 performed better on dev-qa set

    feat_arr_model_2 = [dwelling_type, ownership_type, sun_50, sun_75, sun_std]

    model_2 = analytics_input_object.get('loaded_files', {}).get('solar_files', {}).get('propensity_model').get('model_2')

    propensity_score_model_2 = get_propensity_from_model(feat_arr_model_2, model_2)

    logger.info("solar propensity model 1:, model 2: | {}, {} ".format(np.round(propensity_score_model_1, 2),
                                                                       np.round(propensity_score_model_2, 2)))

    # Getting the optimal panel capacity required and break-even period for adopting solar panel

    panel_capacity, break_even = get_break_even_period(input_data, solar_propensity_config, logger_pass=logger_pass)

    debug['panel_capacity'] = None if panel_capacity is None else float(panel_capacity)
    debug['break_even_period'] = None if break_even is None else float(np.round(break_even, 2))
    debug['propensity_score_model_1'] = float(np.round(propensity_score_model_1, 2))
    debug['propensity_score_model_2'] = float(np.round(propensity_score_model_2, 2))

    return debug
