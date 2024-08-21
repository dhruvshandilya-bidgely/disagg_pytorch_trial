"""
Author - Paras Tehria
Date - 17-Nov-2020
This module is used to compute optimal panel size for a user and corresponding break-even period
"""

# Import python packages

import copy
import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.maths_utils import create_pivot_table
from python3.utils.maths_utils.maths_utils import merge_np_arrays_bycol

from python3.analytics.solar_propensity.functions.get_irradiance_epoch_level import get_irradiance


def get_optimal_panel_capacity(cons_pivot, solar_propensity_config):
    """
    This function computes optimal panel capacity required for a user

    Parameters:
        cons_pivot                    (np.ndarray)       : Consumption pivot table with day as row and HOD as column
        solar_propensity_config        (dict)             : Solar propensity config parameters

    Return:
        panel_capacity                (int)              : Panel capacity in Wh
        panel_price                   (int)              : Panel price
    """

    max_consumption = np.nanquantile(cons_pivot, solar_propensity_config.get('max_consumption_quantile'))
    required_max_generation = max_consumption * solar_propensity_config.get('panel_size_safety_factor')

    panel_price_dict = solar_propensity_config.get('panel_price_dict')

    panel_capacity = int((np.ceil(required_max_generation / Cgbdisagg.WH_IN_1_KWH)) * Cgbdisagg.WH_IN_1_KWH)
    panel_capacity = max(panel_capacity, int(min(panel_price_dict, key=int)))
    panel_capacity = min(panel_capacity, int(max(panel_price_dict, key=int)))
    panel_price = panel_price_dict[str(panel_capacity)]

    return panel_capacity, panel_price


def compute_break_even_period(cons_pivot, irradiance_pivot, solar_propensity_config, panel_size=2500, panel_cost=8500):
    """
    This function computes break even period for given panel size and panel cost

    Parameters:
        cons_pivot                    (np.ndarray)       : Consumption pivot table with day as row and HOD as column
        irradiance_pivot              (np.ndarray)       : Irradiance pivot table with day as row and HOD as column
        solar_propensity_config        (dict)             : Solar propensity config parameters
        panel_size                    (int)              : Input panel size
        panel_cost                    (dict)             : input panel cost

    Return:
        break_even                    (float)             : break even period
    """

    rate_plan = solar_propensity_config.get('rate_plan')
    reverse_rate_plan = solar_propensity_config.get('reverse_rate_plan')

    # Solar generation array
    solar_gen_pivot = irradiance_pivot * panel_size

    # consumption minus generation
    post_gen_cons_matrix = cons_pivot - solar_gen_pivot

    positive_cons = np.fmax(post_gen_cons_matrix, 0)

    net_metered_cons = np.fmin(post_gen_cons_matrix, 0)

    # Calculating daily saved and earned dollars
    saved_cons = cons_pivot - positive_cons
    daily_saved_dollars = np.sum(saved_cons, axis=1) / Cgbdisagg.WH_IN_1_KWH * rate_plan
    daily_earned_dollars = np.sum(net_metered_cons, axis=1) / Cgbdisagg.WH_IN_1_KWH * -1 * reverse_rate_plan

    daily_total_dollars = daily_saved_dollars + daily_earned_dollars

    mean_dollar_savings = np.nanmean(daily_total_dollars)

    # break even period is calculated using mean saved dollars
    break_even = panel_cost / (mean_dollar_savings * Cgbdisagg.DAYS_IN_YEAR)

    return break_even


def get_break_even_period(input_data, solar_propensity_config, logger_pass):
    """
    This is the main function that computes solar propensity for a user

    Parameters:
        input_data                    (np.ndarray)       : input 21-column data
        solar_propensity_config        (dict)             : solar propensity config
        logger_pass                   (dict)             : logger pass for this function

    Return:
        panel_capacity                (int)              : Optimal panel capacity required for this user
        break_even                    (float)             : break even period
    """

    # Initializing new logger child solar_disagg

    logger_local = logger_pass.get('logger_pass').getChild('get_break_even_period')

    logger = logging.LoggerAdapter(logger_local, logger_pass.get('logging_dict'))

    # Getting data start and end timestamps
    data_start_ts = input_data[0, Cgbdisagg.INPUT_EPOCH_IDX]
    data_end_ts = input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX]

    # Getting irradiance using lat and long of the zip

    epoch_time_arr = np.arange(int(data_start_ts), int(data_end_ts), Cgbdisagg.SEC_IN_HOUR)

    longitude = solar_propensity_config.get('longitude')
    latitude = solar_propensity_config.get('latitude')
    timezone = solar_propensity_config.get('timezone')

    if latitude is None or longitude is None or np.isnan(longitude) or np.isnan(latitude):
        logger.info("One or both of latitude and longitude not available, skipping break even period calculation | ")
        irr_arr = []
    else:
        irr_arr = get_irradiance(epoch_time_arr, longitude, latitude, timezone)

    if len(irr_arr) == 0:
        logger.info("Not able to fetch irradiance, skipping break even period calculation | ")
        break_even = None
        panel_capacity = None

    else:

        irradiance_arr_col_dict = solar_propensity_config.get('irradiance_arr_col_dict')

        epoch_col = irradiance_arr_col_dict.get('epoch')
        irr_col = irradiance_arr_col_dict.get('irradiance')

        # normalizing irradiance by dividing with max_irradiance
        max_irradiance = np.nanquantile(irr_arr[:, irr_col], solar_propensity_config.get('max_irradiance_quantile'))
        irr_arr[:, irr_col] = np.fmin(irr_arr[:, irr_col], max_irradiance)

        irr_arr[:, irr_col] = irr_arr[:, irr_col] / max_irradiance

        # Merging irr_arr and input data
        merged_input_data = merge_np_arrays_bycol(input_data, irr_arr, Cgbdisagg.INPUT_EPOCH_IDX, epoch_col)

        cons_pivot, _, _ =\
            create_pivot_table(merged_input_data,
                               Cgbdisagg.INPUT_DAY_IDX, Cgbdisagg.INPUT_HOD_IDX, Cgbdisagg.INPUT_CONSUMPTION_IDX)

        irradiance_pivot, _, _ =\
            create_pivot_table(merged_input_data,
                               Cgbdisagg.INPUT_DAY_IDX, Cgbdisagg.INPUT_HOD_IDX, Cgbdisagg.INPUT_DIMENSION)

        # cons_pivot_copy contains consumption during the time when irradiance is >0
        cons_pivot_copy = copy.deepcopy(cons_pivot)
        cons_pivot_copy[~irradiance_pivot.astype(bool)] = 0

        # Computing the optimal panel capacity for the user
        panel_capacity, panel_price = get_optimal_panel_capacity(cons_pivot_copy, solar_propensity_config)

        # Computing break even period if the user buys the panel
        break_even = compute_break_even_period(cons_pivot, irradiance_pivot, solar_propensity_config,
                                               panel_size=panel_capacity, panel_cost=panel_price)

    return panel_capacity, break_even
