"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update ent consumption ranges using inference rules
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.get_config import get_hybrid_config

from python3.itemization.aer.functions.itemization_utils import get_index_array
from python3.itemization.aer.raw_energy_itemization.utils import update_hybrid_object

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config2 import get_inf_config2


def get_ent_inference(app_index, item_input_object, item_output_object, logger_pass):

    """
    Update ent consumption ranges using inference rules

    Parameters:
        app_index                   (int)       : Index of app in the appliance list
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)      : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_ent_inference')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    disagg_cons = item_output_object.get("inference_engine_dict").get("output_data")[app_index, :, :]
    mid_cons = item_output_object.get("inference_engine_dict").get("appliance_mid_values")[app_index, :, :]
    max_cons = item_output_object.get("inference_engine_dict").get("appliance_max_values")[app_index, :, :]
    min_cons = item_output_object.get("inference_engine_dict").get("appliance_min_values")[app_index, :, :]

    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    samples_per_hour = int(len(min_cons[0]) / Cgbdisagg.HRS_IN_DAY)

    ent_config = get_inf_config2(item_input_object).get("ent")

    mid_cons = np.maximum(mid_cons, disagg_cons*ent_config.get('min_disagg_frac'))
    max_cons = np.maximum(max_cons, disagg_cons*ent_config.get('min_disagg_frac'))
    min_cons = np.maximum(min_cons, disagg_cons*ent_config.get('min_disagg_frac'))

    if np.sum(disagg_cons) == 0:
        logger.info("Entertainment hybrid disagg is zero | ")
        return item_output_object

    ################## RULE 1 - less entertainment consumtion is television is absent ###########################

    # Fetching appliance profile data of television data

    app_profile = item_input_object.get("app_profile").get('television')

    if app_profile is not None:
        app_profile = app_profile.get("number", -1)
    else:
        app_profile = -1

    if app_profile == 0:
        logger.info("Television app profile is 0 | ")
        mid_cons = mid_cons * ent_config.get("non_television_cons_factor")
        min_cons = min_cons * ent_config.get("non_television_cons_factor")
        max_cons = max_cons * ent_config.get("non_television_cons_factor")

    ################## RULE 2 - Zero consumption during sleep hours ###########################################

    sleep_hours = copy.deepcopy(item_output_object.get("profile_attributes").get("sleep_hours"))

    default_sleep_hours = ent_config.get('default_sleep_hours')
    max_ent_cons_cap = ent_config.get('max_ent_cons_cap')
    max_ent_cons_offset = ent_config.get('max_ent_cons_offset')
    min_ent_cons_cap = ent_config.get('min_ent_cons_cap')
    min_ent_cons_offset = ent_config.get('min_ent_cons_offset')

    if np.all(sleep_hours == 0):
        sleep_hours = np.ones_like(sleep_hours)
        sleep_hours[default_sleep_hours] = 0
    else:
        sleep_hours[get_index_array(item_output_object.get("profile_attributes").get("sleep_time") * samples_per_hour,
                                    item_output_object.get("profile_attributes").get("sleep_time") * samples_per_hour +
                                    int(samples_per_hour) + 1,
                                    samples_per_hour * Cgbdisagg.HRS_IN_DAY)] = 1

    mid_cons[:, np.logical_not(sleep_hours)] = 0
    min_cons[:, np.logical_not(sleep_hours)] = 0
    max_cons[:, np.logical_not(sleep_hours)] = 0

    ############################# RULE 3 - Monthly consumption limit ##########################################

    min_cons, mid_cons, max_cons = \
        update_ent_cons_based_on_min_max_threshold(min_cons, mid_cons, max_cons, hybrid_config, disagg_cons, sleep_hours, ent_config)

    ################## RULE 4 - Modify consumption based on app profile ###########################################

    ent_app_prof_present_flag = not item_input_object.get("appliance_profile").get("default_ent_flag")

    if ent_app_prof_present_flag:
        ent_app_count = item_input_object.get("appliance_profile").get("ent")
        ent_app_type = item_input_object.get("appliance_profile").get("ent_type")
        appliance_consumption = [200, 200, 700]

        ent_app_count[ent_app_type == 0] = ent_app_count[ent_app_type == 0] * 0.1
        appliance_consumption = np.dot(appliance_consumption, ent_app_count)

        min_ent_limit = np.fmax(min_ent_cons_cap/samples_per_hour,
                                appliance_consumption / samples_per_hour - min_ent_cons_offset / samples_per_hour)

        mid_cons[mid_cons > 0] = np.fmax(min_ent_limit, mid_cons[mid_cons > 0])
        max_cons[max_cons > 0] = np.fmax(min_ent_limit, max_cons[max_cons > 0])
        min_cons[min_cons > 0] = np.fmax(min_ent_limit, min_cons[min_cons > 0])

        # calculate ts level min/max ent consumption based on app profile of the user

        max_ent_limit = (appliance_consumption - max_ent_cons_offset) / samples_per_hour
        max_ent_limit = np.fmax(max_ent_limit, max_ent_cons_cap/samples_per_hour)

        logger.info('Max ent ts level cap based on app profile | %s', max_ent_limit)
        logger.info('Min ent ts level cap based on app profile | %s', min_ent_limit)

        mid_cons = np.fmin(max_ent_limit, mid_cons)
        max_cons = np.fmin(max_ent_limit, max_cons)
        min_cons = np.fmin(max_ent_limit, min_cons)

    max_cons = np.minimum(max_cons, input_data)
    min_cons = np.minimum(min_cons, input_data)
    mid_cons = np.minimum(mid_cons, input_data)

    feeble_ent_cons_flag = np.sum(mid_cons.sum(axis=1) > 0) < len(mid_cons) * ent_config.get('zero_ent_days_frac')

    if feeble_ent_cons_flag:
        mid_cons[:, :] = 0
        min_cons[:, :] = 0
        max_cons[:, :] = 0
        logger.info('killing initial entertainment consumption due to feeble consumption | ')

    item_output_object = update_hybrid_object(app_index, item_output_object, mid_cons, max_cons, min_cons)

    t_end = datetime.now()

    logger.info("Entertainment inference calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object


def update_ent_cons_based_on_min_max_threshold(min_cons, mid_cons, max_cons, hybrid_config, disagg_cons, sleep_hours, ent_config):

    """
    adjust entertainment consumption based on min and max monthly limit

    Parameters:
        min_cons                    (np.ndarray)    : ts level min lighting consumption
        mid_cons                    (np.ndarray)    : ts level avg lighting consumption
        max_cons                    (np.ndarray)    : ts level max lighting consumption
        hybrid_config               (dict)          : hybrid v2 pilot config
        disagg_cons                 (np.ndarray)    : initial ent estimates of the user
        sleep_hours                 (np.ndarray)    : sleep hours of the user
        ent_config                  (logger)        : ent config

    Returns:
        min_cons                    (np.ndarray)    : ts level min laundry consumption
        mid_cons                    (np.ndarray)    : ts level avg laundry consumption
        max_cons                    (np.ndarray)    : ts level max laundry consumption
    """

    # fetching ent monthly max and min from hybrid config

    monthly_cons_max_limit = 0
    monthly_cons_min_limit = 0

    ld_idx = hybrid_config.get("app_seq").index('ent')
    have_min_cons = hybrid_config.get("have_min_cons")[ld_idx]
    min_cons_limit = hybrid_config.get("min_cons")[ld_idx]

    if have_min_cons and min_cons_limit > 0:
        monthly_cons_min_limit = min_cons_limit

    have_mid_cons = hybrid_config.get("have_max_cons")[ld_idx]
    mid_cons_lim = hybrid_config.get("max_cons")[ld_idx]

    if have_mid_cons and mid_cons_lim > 0:
        monthly_cons_max_limit = mid_cons_lim

    max_cons = max_cons * ent_config.get('max_range_multiplier')

    # killing ent at sleep hours and late night hours

    max_cons = np.maximum(max_cons, disagg_cons)

    max_cons[:, np.logical_not(sleep_hours)] = 0

    zero_ent_hours = ent_config.get('zero_ent_hours').astype(int)

    max_cons[:, zero_ent_hours] = 0
    mid_cons[:, zero_ent_hours] = 0
    min_cons[:, zero_ent_hours] = 0

    # adjusting entertainment consumption based on min and max monthly limit

    decrease_ent_cons = monthly_cons_max_limit and monthly_cons_max_limit <= ent_config.get('max_ent')

    if decrease_ent_cons:
        monthly_cons = ((np.sum(max_cons) / len(max_cons)) * Cgbdisagg.DAYS_IN_MONTH) / Cgbdisagg.WH_IN_1_KWH
        if monthly_cons > monthly_cons_max_limit:
            factor = monthly_cons_max_limit / monthly_cons
            max_cons = max_cons * factor
            mid_cons = np.minimum(mid_cons, max_cons)
            min_cons = np.maximum(max_cons, min_cons)

    increase_ent_cons = monthly_cons_min_limit and monthly_cons_min_limit >= ent_config.get('min_ent')

    if increase_ent_cons:
        monthly_cons = ((np.sum(min_cons) / len(min_cons)) * Cgbdisagg.DAYS_IN_MONTH) / Cgbdisagg.WH_IN_1_KWH
        if monthly_cons < monthly_cons_min_limit:
            factor = monthly_cons_min_limit / monthly_cons
            min_cons = min_cons * factor
            mid_cons = np.maximum(mid_cons, min_cons)
            max_cons = np.maximum(max_cons, min_cons)

    min_cons = np.nan_to_num(min_cons)
    max_cons = np.nan_to_num(max_cons)
    mid_cons = np.nan_to_num(mid_cons)

    perc_cap = ent_config.get('perc_cap_for_ts_level_cons')

    if np.sum(mid_cons):
        mid_cons = np.fmin(mid_cons, np.percentile(mid_cons[mid_cons > 0], perc_cap))
    if np.sum(min_cons):
        min_cons = np.fmin(min_cons, np.percentile(min_cons[min_cons > 0], perc_cap))
    if np.sum(max_cons):
        max_cons = np.fmin(max_cons, np.percentile(max_cons[max_cons > 0], perc_cap))

    return min_cons, mid_cons, max_cons
