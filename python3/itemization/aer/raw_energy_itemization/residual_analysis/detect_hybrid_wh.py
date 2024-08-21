
"""
Author - Nisha Agarwal
Date - 9th Feb 20
Laundry detection module
"""

# Import python packages

import logging
import numpy as np
import os
import copy
import pandas as pd

# import functions from within the project

from python3.config.pilot_constants import PilotConstants

from python3.itemization.init_itemization_config import seq_config

from python3.config.Cgbdisagg import Cgbdisagg
from python3.itemization.aer.raw_energy_itemization.residual_analysis.config.get_residual_config import get_residual_config


def detect_hybrid_wh(item_input_object, item_output_object, disagg, logger):

    """
    Update wh hsm

    Parameters:
        final_tou_consumption     (np.ndarray)    : ts level itemization output
        ev_idx                    (int)           : wh appliance index
        length                    (int)           : count of non-vac days
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs

    Returns:
        item_output_object        (dict)          : Dict containing all hybrid outputs
    """

    pilot = item_input_object.get("config").get("pilot_id")

    input_data = item_input_object.get("item_input_params").get('day_input_data')

    days_count = len(input_data)

    wh_hybrid_hld = 0

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    box_seq = copy.deepcopy(item_output_object.get('box_dict').get('box_seq'))
    box_data = item_output_object.get('box_dict').get('box_cons')
    samples_per_hour = int(box_data.shape[1] / Cgbdisagg.HRS_IN_DAY)

    config = get_residual_config(samples_per_hour).get("wh_addition_dict")

    run_hybrid_v2 = item_input_object.get('item_input_params').get('run_hybrid_v2_flag')
    pilot_config = item_input_object.get('pilot_level_config')

    high_wh_pilot = pilot in config.get("high_wh_pilots")

    if run_hybrid_v2:
        cov = int(np.nan_to_num(pilot_config.get('wh_config').get('coverage')))
        fuel_type = str(np.nan_to_num(pilot_config.get('wh_config').get('type')))
        take_from_disagg_flag = int(pilot_config.get('wh_config').get('bounds').get('take_from_disagg'))
        high_wh_pilot = high_wh_pilot or ((cov >= 85) and fuel_type == 'ELECTRIC' and take_from_disagg_flag != 0)

    winter_month = config.get("winter_months")

    date_list = item_output_object.get("date_list")
    month_list = pd.DatetimeIndex(date_list).month.values

    # pick thin pulse type of boxes

    box_seq[box_seq[:, seq_len] >= max(0.75*samples_per_hour, 2), 0] = 0
    box_data = box_data.flatten()

    for i in range(len(box_seq)):
        if not box_seq[i, seq_label]:
            box_data[int(box_seq[i, seq_start]):int(box_seq[i, seq_end])+1] = 0

    box_data = np.reshape(box_data, disagg.shape)

    if (days_count > config.get("wh_days_thres")) and (not high_wh_pilot) and (pilot not in PilotConstants.SEASONAL_WH_ENABLED_PILOTS):
        return 0

    add_swh_for_less_days = ((days_count <= 150) and (pilot in PilotConstants.SEASONAL_WH_ENABLED_PILOTS) and (np.sum(disagg) == 0)) or \
                            ((days_count <= 300) and ((pilot in PilotConstants.SEASONAL_WH_ENABLED_PILOTS) and (pilot not in config.get("all_year_swh_pilots")))
                             and (np.sum(disagg) == 0) and (np.sum(item_output_object.get("hvac_dict").get('wh')) <  10000))

    # adding whs for pilots with high wh coverage, irrespective of app profiles

    if high_wh_pilot:

        wh_hybrid_hld = 1

        logger.info('Adding WH based on thin pulse detection')

    wh_hybrid_hld = \
        get_wh_hld_change_bool(wh_hybrid_hld, add_swh_for_less_days, month_list, winter_month, item_input_object, item_output_object, box_data, logger)

    return wh_hybrid_hld


def get_wh_hld_change_bool(wh_hybrid_hld, add_swh_for_less_days, month_list, winter_month, item_input_object, item_output_object, box_data, logger):

    """
    Update wh hsm

    Parameters:
        wh_hybrid_hld
        add_swh_for_less_days
        month_list
        winter_month
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        box_data
        logger

    Returns:
        wh_hybrid_hld
    """

    pilot = item_input_object.get("config").get("pilot_id")

    input_data = item_input_object.get("item_input_params").get('day_input_data')

    days_count = len(input_data)

    samples_per_hour = int(box_data.shape[1] / Cgbdisagg.HRS_IN_DAY)

    monthly_data = (np.sum(input_data) / len(input_data)) * Cgbdisagg.DAYS_IN_MONTH

    config = get_residual_config(samples_per_hour).get("wh_addition_dict")

    run_hybrid_v2 = item_input_object.get('item_input_params').get('run_hybrid_v2_flag')
    pilot_config = item_input_object.get('pilot_level_config')

    mid_wh_pilot = pilot in config.get("mid_wh_pilots")
    low_wh_pilot = pilot in config.get("low_wh_pilots")

    if run_hybrid_v2:
        cov = int(np.nan_to_num(pilot_config.get('wh_config').get('coverage')))
        fuel_type = str(np.nan_to_num(pilot_config.get('wh_config').get('type')))
        take_from_disagg_flag = int(pilot_config.get('wh_config').get('bounds').get('take_from_disagg'))

        mid_wh_pilot = mid_wh_pilot or ((cov >= 60) and (cov < 85) and fuel_type == 'ELECTRIC' and take_from_disagg_flag != 0)
        low_wh_pilot = low_wh_pilot or ((cov >= 40) and (cov < 60) and take_from_disagg_flag != 0)

    if len(input_data) < config.get('min_days_required'):

        wh_hybrid_hld = 0

        logger.info('Not adding wh since days count is less than threshold')

    # adding whs for pilots with electric WHS, irrespective of app profiles, for cases where disagg doesnt provide output

    elif days_count <= config.get("wh_days_thres") and (mid_wh_pilot or low_wh_pilot):

        thres = config.get('mid_cov_wh_pilots_min_days_frac') * (mid_wh_pilot) + \
                config.get('low_cov_wh_pilots_min_days_frac') * (low_wh_pilot)

        lap_data = box_data[:, config.get("low_wh_lap_hours")]

        label = np.logical_and(lap_data < config.get('thin_pulse_max_cons')/samples_per_hour,
                               lap_data > config.get('thin_pulse_min_cons')/samples_per_hour)

        label = np.sum(label, axis=1)

        wh_hybrid_hld = not ((monthly_data < config.get("wh_month_thres")) or (np.sum(label > 0) < thres*days_count))

        logger.info('Adding WH based on thin pulse detection | ')

    # adding whs for pilots with electric WHS, irrespective of app profiles, for cases where disagg doesnt provide output

    elif (days_count <= config.get("wh_days_thres")) and \
            (pilot in config.get("twh_pilots")) and (np.sum(item_output_object.get("timed_app_dict").get('twh')) == 0):

        lap_data = box_data[:, config.get("low_wh_lap_hours")]

        label = np.logical_and(lap_data < config.get('thin_pulse_max_cons')/samples_per_hour,
                               lap_data > config.get('thin_pulse_min_cons')/samples_per_hour)

        label = np.sum(label, axis=1)

        wh_hybrid_hld = not ((monthly_data < config.get("wh_month_thres")) or (np.sum(label > 3) < config.get('min_days_frac')*days_count))

        logger.info('Adding WH based on thin pulse detection | ')

    # adding whs for pilots with SWH, irrespective of app profiles,for cases seasonality detection is not possible

    elif add_swh_for_less_days:

        lap_data = box_data[:, config.get("swh_hours")]

        label = np.logical_and(lap_data < config.get('max_swh_amp')/samples_per_hour,
                               lap_data > config.get('min_swh_amp')/samples_per_hour)

        label = np.sum(label, axis=1)

        wh_hybrid_hld = not ((not np.any(np.isin(month_list, winter_month))) or
                             (monthly_data < config.get("swh_month_thres")) or
                             (np.sum(label[np.isin(month_list, winter_month)] > 0) <
                              config.get('min_days_frac')*np.sum(np.isin(month_list, winter_month))))

        logger.info('Adding WH for SWH pilot | ')

    return wh_hybrid_hld
