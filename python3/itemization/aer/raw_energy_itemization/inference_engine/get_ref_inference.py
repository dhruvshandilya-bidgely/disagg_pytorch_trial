"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update laundry consumption ranges using inference rules
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants
from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.raw_energy_itemization.utils import update_hybrid_object

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config


def get_ref_inference(app_index, item_input_object, item_output_object, date_list, logger_pass):

    """
    Update ref consumption ranges using inference rules

    Parameters:
        app_index                   (int)       : Index of app in the appliance list
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        date_list                   (np.ndarray): list of target dates for heatmap dumping
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)      : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_ref_inference')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    config = get_inf_config().get("ref")

    # Fetching required inputs

    app_pot = item_output_object.get("inference_engine_dict").get("appliance_pot")[app_index]
    disagg_cons = item_output_object.get("inference_engine_dict").get("output_data")[app_index, :, :]
    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    ########################### RULE 1 - Adding seasonality in ref output ######################################

    # Calculating min, mid and max range for the appliance

    mid_cons = np.multiply(np.max(disagg_cons), app_pot)
    min_cons = np.multiply(np.max(disagg_cons), app_pot)
    max_cons = disagg_cons

    samples_per_hour = int(input_data.shape[1]/Cgbdisagg.HRS_IN_DAY)

    ########################### RULE 2 - Increasing consumption to meet  ###########################

    multiplier_to_resolve_underest = config.get(str(int(Cgbdisagg.SEC_IN_1_MIN/samples_per_hour)) + "_min_mul")

    if item_input_object.get("config").get("pilot_id") in PilotConstants.TIMED_WH_JAPAN_PILOTS:
        multiplier_to_resolve_underest = config.get('scaling_factor_for_japan_users')

    max_day_level_limit_for_scaling_ref = config.get('max_day_level_limit_for_scaling_ref')

    if (multiplier_to_resolve_underest is not None) and (np.mean(disagg_cons.sum(axis=1)) < max_day_level_limit_for_scaling_ref):
        max_cons = max_cons * multiplier_to_resolve_underest
        mid_cons = mid_cons * multiplier_to_resolve_underest
        min_cons = min_cons * multiplier_to_resolve_underest

    ########################### RULE 3 - Increasing consumption for high consumption homes ###########################

    mid_cons = np.minimum(mid_cons, input_data)
    max_cons = np.minimum(max_cons, input_data)
    min_cons = np.minimum(min_cons, input_data)

    # Updating the values in the original dictionary

    item_output_object = update_hybrid_object(app_index, item_output_object, mid_cons, max_cons, min_cons)

    item_output_object["inference_engine_dict"]["output_data"][app_index, :, :] = disagg_cons

    t_end = datetime.now()

    logger.debug("Ref inference calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object
