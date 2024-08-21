
"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update wh consumption ranges using inference rules
"""

# Import python packages

import copy
import logging
import numpy as np
import pandas as pd

from numpy.random import RandomState

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.pilot_constants import PilotConstants

from python3.itemization.init_itemization_config import random_gen_config

from python3.itemization.aer.functions.itemization_utils import fill_arr_based_seq_val
from python3.itemization.aer.functions.itemization_utils import fill_arr_based_seq_val_for_valid_boxes
from python3.itemization.aer.functions.itemization_utils import fill_circular_arr_based_seq_val_for_valid_boxes

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config


def postprocess_swh_output(app_ranges, swh_output, disagg_cons, app_index, residual_wh, residual, input_data,
                           item_input_object, item_output_object, season_potential):

    """
    Update wh consumption ranges using inference rules

    Parameters:
        app_ranges                  (list)            : list of wh min/mid/max ranges
        swh_output                  (np.ndarray)      : swh output
        disagg_cons                 (np.ndarray)      : app disagg cons
        app_index                   (int)             : Index of app in the appliance list
        residual_wh                 (np.ndarray)      : residual seasonal wh
        input_data                  (np.ndarray)      : input data
        item_input_object           (dict)            : Dict containing all hybrid inputs
        item_output_object          (dict)            : Dict containing all hybrid outputs
        season_potential            (np.ndarray)      : wh usage potential
    Returns:
        min_cons                    (np.ndarray)       : app min cons
        mid_cons                    (np.ndarray)       : app mid cons
        max_cons                    (np.ndarray)       : app max cons
    """

    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_label = seq_config.SEQ_LABEL
    seq_len = seq_config.SEQ_LEN

    min_cons, mid_cons, max_cons = app_ranges[0], app_ranges[1], app_ranges[2]

    pilot = item_input_object.get("config").get("pilot_id")

    if item_input_object.get('item_input_params').get('swh_hld'):
        mid_cons = np.maximum(mid_cons, disagg_cons)
        max_cons = np.maximum(max_cons, disagg_cons)
        min_cons = np.maximum(min_cons, disagg_cons)

    wh_days = np.sum(residual_wh > 0, axis=1) > 0
    wh_days = fill_arr_based_seq_val(wh_days, wh_days, 3, 1, 1, overnight_tag=0)

    seq = find_seq(wh_days, np.zeros_like(wh_days), np.zeros_like(wh_days), overnight=0)
    wh_tou = np.sum(mid_cons, axis=0) > 0
    wh_amp = np.median(mid_cons[mid_cons > 0])

    # fill the gaps in swh output

    for i in range(len(seq)):

        seq_start_idx = seq[i, seq_start]
        seq_end_idx = seq[i, seq_end]

        extended_swh_cons = np.zeros_like(mid_cons)

        # if gap is present and wh is seasonal wh type

        fill_gap_in_swh_output = \
            seq[i, seq_label] == 0 and seq[i, seq_len] < 70 and seq[i, seq_len] > 3 and \
            (swh_output or (pilot in PilotConstants.SEASONAL_WH_ENABLED_PILOTS and np.sum(mid_cons) > 0 and len(disagg_cons) > 200) and np.sum(residual_wh))

        if fill_gap_in_swh_output:

            # adding additional SWH at valid time of day

            extended_swh_cons[input_data > wh_amp * 0.9] = 1
            extended_swh_cons[:, np.logical_not(wh_tou)] = 0
            extended_swh_cons[extended_swh_cons > 0] = residual[extended_swh_cons > 0]
            extended_swh_cons = np.fmax(0, extended_swh_cons)
            item_output_object["inference_engine_dict"]["appliance_conf"][app_index][extended_swh_cons > 0] = 1.1
            mid_cons[seq_start_idx: seq_end_idx] = mid_cons[seq_start_idx: seq_end_idx] + extended_swh_cons[seq_start_idx: seq_end_idx]
            min_cons[seq_start_idx: seq_end_idx] = min_cons[seq_start_idx: seq_end_idx] + extended_swh_cons[seq_start_idx: seq_end_idx]
            max_cons[seq_start_idx: seq_end_idx] = max_cons[seq_start_idx: seq_end_idx] + extended_swh_cons[seq_start_idx: seq_end_idx]

    # fill the gaps in wh output where wh is present throughout the year

    min_cons, mid_cons, max_cons = postprocess_yearly_swh_output(app_ranges, disagg_cons, app_index, residual,
                                                                 input_data, item_input_object, item_output_object, season_potential)

    min_cons, mid_cons, max_cons = \
        update_swh_output(item_input_object, item_output_object, min_cons, mid_cons, max_cons, disagg_cons, season_potential)

    return min_cons, mid_cons, max_cons


def postprocess_yearly_swh_output(app_ranges, disagg_cons, app_index, residual, input_data, item_input_object, item_output_object, season_potential):

    """
    Update wh consumption ranges using inference rules

      Parameters:
        app_ranges                  (list)            : list of wh min/mid/max ranges
        disagg_cons                 (np.ndarray)      : app disagg cons
        app_index                   (int)             : Index of app in the appliance list
        residual                    (np.ndarray)      : residual data
        input_data                  (np.ndarray)      : input data
        item_input_object           (dict)            : Dict containing all hybrid inputs
        item_output_object          (dict)            : Dict containing all hybrid outputs
        season_potential            (np.ndarray)      : wh usage potential

    Returns:
        min_cons                    (np.ndarray)       : app min cons
        mid_cons                    (np.ndarray)       : app mid cons
        max_cons                    (np.ndarray)       : app max cons
    """

    config = get_inf_config(int(disagg_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)).get("wh")

    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_label = seq_config.SEQ_LABEL

    min_cons, mid_cons, max_cons = app_ranges[0], app_ranges[1], app_ranges[2]

    pilot = item_input_object.get("config").get("pilot_id")
    samples_per_hour = int(mid_cons.shape[1]/Cgbdisagg.HRS_IN_DAY)

    wh_tou = np.sum(mid_cons, axis=0) > 0

    tou_seq = find_seq(wh_tou, np.zeros_like(wh_tou), np.zeros_like(wh_tou), overnight=0).astype(int)

    for i in range(len(tou_seq)):
        if tou_seq[i, seq_label] > 0:
            wh_tou[tou_seq[i, seq_end]:tou_seq[i, seq_end] +samples_per_hour+ 1] = 1
            wh_tou[tou_seq[i, seq_start]-samples_per_hour:tou_seq[i, seq_start]] = 1

    # fill the gaps in swh output for pilot with wh usage throughout the year wheere disagg is non-zero

    if pilot in config.get("all_year_wh") and np.sum(mid_cons) > 0 and len(disagg_cons) > 100 and (np.sum(disagg_cons) == 0):

        wh_amp = np.median(mid_cons[mid_cons > 0])

        wh_picked_from_res = np.zeros_like(mid_cons)
        wh_picked_from_res[input_data > wh_amp * config.get('wh_extention_amp_thres')] = 1
        wh_picked_from_res[:, np.logical_not(wh_tou)] = 0
        wh_picked_from_res[wh_picked_from_res > 0] = input_data[wh_picked_from_res > 0]
        wh_picked_from_res = np.fmax(0, wh_picked_from_res)
        wh_picked_from_res[:, :5 * samples_per_hour] = 0
        item_output_object["inference_engine_dict"]["appliance_conf"][app_index][wh_picked_from_res > 0] = 1.1
        mid_cons[:] = mid_cons[:] + wh_picked_from_res[:]
        min_cons[:] = min_cons[:] + wh_picked_from_res[:]
        max_cons[:] = max_cons[:] + wh_picked_from_res[:]

    # fill the gaps in swh output for pilot with wh usage throughout the year where disagg is 0

    if pilot in config.get("all_year_wh") and np.sum(mid_cons) > 0 and len(disagg_cons) > 100 and (np.sum(disagg_cons) > 0):

        wh_tou[:6 * samples_per_hour + 1] = 0
        wh_tou[12 * samples_per_hour:] = 0

        wh_tou = wh_tou.astype(bool)

        amp = np.median(mid_cons[:, wh_tou][mid_cons[:, wh_tou] > 0])
        wh_picked_from_res = np.zeros_like(mid_cons)
        wh_picked_from_res[input_data > amp * config.get('wh_extention_amp_thres')] = 1
        wh_picked_from_res[:, np.logical_not(wh_tou)] = 0
        wh_picked_from_res[wh_picked_from_res > 0] = input_data[wh_picked_from_res > 0]
        wh_picked_from_res = np.fmax(0, wh_picked_from_res)
        wh_picked_from_res[:, :5 * samples_per_hour] = 0
        item_output_object["inference_engine_dict"]["appliance_conf"][app_index][wh_picked_from_res > 0] = 1.1
        mid_cons[:] = mid_cons[:] + wh_picked_from_res[:]
        min_cons[:] = min_cons[:] + wh_picked_from_res[:]
        max_cons[:] = max_cons[:] + wh_picked_from_res[:]

    min_cons, mid_cons, max_cons = \
        update_swh_output(item_input_object, item_output_object, min_cons, mid_cons, max_cons, disagg_cons, season_potential)

    return min_cons, mid_cons, max_cons


def postprocess_wh_output(wh_disagg, item_output_object, item_input_object, residual, residual_cons, residual_wh,
                          wh_ranges, app_index, swh_output, flow_wh):

    """
    Update wh consumption ranges using inference rules

    Parameters:
        wh_disagg                 (np.ndarray)       : true disagg WH output
        item_output_object        (dict)             : Dict containing all hybrid outputs
        item_input_object         (dict)             : Dict containing all hybrid inputs
        residual                  (np.ndarray)       : residual data
        wh_ranges                 (list)             : wh min/max/mid ranges
        app_index                 (int)              : Index of app in the appliance list
        swh_output                (np.ndarray)       :  swh output
        logger                    (logger)           : logger object

    Returns:
        min_cons                  (np.ndarray)       : app min cons
        mid_cons                  (np.ndarray)       : app mid cons
        max_cons                  (np.ndarray)       : app max cons

    """

    seq_len = seq_config.SEQ_LEN

    min_cons = wh_ranges[0]
    mid_cons = wh_ranges[1]
    max_cons = wh_ranges[2]

    pilot = item_input_object.get("config").get("pilot_id")
    disagg_cons = item_output_object.get("inference_engine_dict").get("output_data")[app_index, :, :]
    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    samples_per_hour = int(disagg_cons.shape[1]/Cgbdisagg.HRS_IN_DAY)

    config = get_inf_config(samples_per_hour).get("wh")

    morning_hours = np.arange(int(config.get('min_swh_usage_hour') * samples_per_hour), samples_per_hour * Cgbdisagg.HRS_IN_DAY)

    # Filling the gaps in residual WH boxes

    if not np.all(disagg_cons == 0):
        wh_days = np.sum(disagg_cons, axis=1) > 0

        wh_days_seq = find_seq(wh_days, np.zeros(wh_days.shape), np.zeros(wh_days.shape))
        wh_days_seq[np.logical_and(wh_days_seq[:, seq_len] < 40, wh_days_seq[:, 0] == 0), 0] = 1

        wh_days = fill_circular_arr_based_seq_val_for_valid_boxes(wh_days_seq, wh_days_seq[:, 0], wh_days, len(disagg_cons), 1, 1)

        residual_cons[np.logical_not(wh_days)] = 0

    # Removing residual WH boxes on vacation days
    residual_cons[vacation_days] = 0

    if np.sum(wh_disagg) > 0:
        residual_cons = np.fmax(0, np.minimum(residual_cons, item_output_object.get("inference_engine_dict").get("residual_data")))

    residual[:, morning_hours] = residual[:, morning_hours] - residual_cons[:, morning_hours]

    item_output_object["inference_engine_dict"].update({
        "residual_data": residual + np.fmin(0, item_output_object.get("inference_engine_dict").get("residual_data"))
    })

    non_wh_hours = config.get('non_wh_hours')

    if item_input_object.get('item_input_params').get('swh_hld') > 0 or pilot in PilotConstants.SEASONAL_WH_ENABLED_PILOTS:
        non_wh_hours = config.get('non_swh_hours')

    residual_cons[:, non_wh_hours] = 0
    residual_cons = residual_cons * (pilot not in config.get("japan_pilots"))

    storage_type_wh_flag = not (flow_wh and (pilot in PilotConstants.SEASONAL_WH_ENABLED_PILOTS))

    # adding residual WH boxes into WH estimates at probable WH hours

    if storage_type_wh_flag:
        mid_cons[:, morning_hours] = mid_cons[:, morning_hours] + residual_cons[:, morning_hours]
        max_cons[:, morning_hours] = max_cons[:, morning_hours] + residual_cons[:, morning_hours]
        min_cons[:, morning_hours] = min_cons[:, morning_hours] + residual_cons[:, morning_hours]

        additional_night_hours = config.get('additional_night_hours')

        mid_cons[:, additional_night_hours] = mid_cons[:, additional_night_hours] + residual_cons[:, additional_night_hours]
        max_cons[:, additional_night_hours] = max_cons[:, additional_night_hours] + residual_cons[:, additional_night_hours]
        min_cons[:, additional_night_hours] = min_cons[:, additional_night_hours] + residual_cons[:, additional_night_hours]

    ################ RULE 4 - postprocessing for swh output, to fill days with high wh usage potential ############

    season_potential = item_output_object.get("wh_pot")[:, 0]

    min_cons, mid_cons, max_cons = postprocess_swh_output([min_cons, mid_cons, max_cons], swh_output, disagg_cons,
                                                          app_index, residual_wh, residual, input_data,
                                                          item_input_object, item_output_object, season_potential)

    ################# RULE 5 - Check if wh consumption follows expected seasonality pattern #######################

    min_cons, mid_cons, max_cons, item_output_object = \
        wh_seasonality_check(min_cons, mid_cons, max_cons, disagg_cons, item_output_object, app_index,
                             residual_cons, vacation_days)

    return min_cons, mid_cons, max_cons, residual_cons


def update_swh_output(item_input_object, item_output_object, min_cons, mid_cons, max_cons, disagg_cons, season_potential):

    """
    This function performs postprocessing on ts level WH estimates after adding seasonal signature into SWH output

    Parameters:
        item_input_object           (dict)             : Dict containing all hybrid inputs
        item_output_object          (dict)             : Dict containing all hybrid outputs
        min_cons                    (np.ndarray)       : app min cons
        mid_cons                    (np.ndarray)       : app mid cons
        max_cons                    (np.ndarray)       : app max cons
        disagg_cons                 (np.ndarray)       : app disagg cons
        season_potential            (np.ndarray)       : wh usage potential

    Returns:
        min_cons                    (np.ndarray)       : app min cons
        mid_cons                    (np.ndarray)       : app mid cons
        max_cons                    (np.ndarray)       : app max cons
    """

    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END

    config = get_inf_config(int(disagg_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)).get("wh")
    date_list = item_output_object.get("date_list")
    pilot = item_input_object.get("config").get("pilot_id")

    samples_per_hour = int(mid_cons.shape[1]/Cgbdisagg.HRS_IN_DAY)

    # removing outlier consumption points if SWH is added from hybrid instead of disagg

    swh_added_from_hybrid = pilot in PilotConstants.SEASONAL_WH_ENABLED_PILOTS and (np.sum(disagg_cons) == 0)

    if swh_added_from_hybrid:
        mid_cons[mid_cons > config.get('swh_max_amp') / samples_per_hour] = 0
        mid_cons[mid_cons < config.get('swh_min_amp') / samples_per_hour] = 0
        max_cons[max_cons > config.get('swh_max_amp') / samples_per_hour] = 0
        min_cons[min_cons < config.get('swh_min_amp') / samples_per_hour] = 0
        max_cons[max_cons > config.get('swh_max_amp') / samples_per_hour] = 0
        max_cons[max_cons < config.get('swh_min_amp') / samples_per_hour] = 0

        if np.sum(mid_cons) > 0:
            item_input_object['item_input_params']['swh_hld'] = 1

    # remove swh based on month info

    if (item_input_object.get('item_input_params').get('swh_hld')) and (np.sum(disagg_cons) > 0) and (
            pilot not in config.get("all_year_wh")):
        month_list = pd.DatetimeIndex(date_list).month.values

        non_wh_months = config.get("non_swh_months")

        min_cons[np.isin(month_list, non_wh_months)] = 0
        mid_cons[np.isin(month_list, non_wh_months)] = 0
        max_cons[np.isin(month_list, non_wh_months)] = 0

    swh_pilots = (pilot in PilotConstants.SEASONAL_WH_ENABLED_PILOTS) and (pilot not in config.get("all_year_wh"))

    # remove wh from days that were less probable for having swh

    swh_pot_days_buffer = config.get('swh_pot_days_buffer')

    if (season_potential is not None) and (np.sum(season_potential) > 0) and swh_pilots and (not np.all(season_potential > 0)):
        seq = find_seq(season_potential > 0, np.zeros_like(season_potential), np.zeros_like(season_potential), overnight=0)

        zero_wh = season_potential == 0

        for i in range(len(seq)):
            if seq[i, 0]:
                zero_wh[get_index_array(seq[i, seq_start] - swh_pot_days_buffer, seq[i, seq_start], len(season_potential))] = 0
                zero_wh[get_index_array(seq[i, seq_end], seq[i, seq_end] + swh_pot_days_buffer, len(season_potential))] = 0

        zero_wh[disagg_cons.sum(axis=1) > 0] = 0

        min_cons[zero_wh > 0] = 0
        mid_cons[zero_wh > 0] = 0
        max_cons[zero_wh > 0] = 0

    # remove swh (added from hybrid) in late night hours

    if swh_pilots:

        swh_start_hour = 5*samples_per_hour + 1
        swh_end_hour = 22 * samples_per_hour

        mid_cons[:, :swh_start_hour] = np.minimum(mid_cons[:, :swh_start_hour], disagg_cons[:, :swh_start_hour])
        min_cons[:, :swh_start_hour] = np.minimum(min_cons[:, :swh_start_hour], disagg_cons[:, :swh_start_hour])
        max_cons[:, :swh_start_hour] = np.minimum(max_cons[:, :swh_start_hour], disagg_cons[:, :swh_start_hour])

        mid_cons[:, swh_end_hour:] = np.minimum(mid_cons[:, swh_end_hour:], disagg_cons[:, swh_end_hour:])
        min_cons[:, swh_end_hour:] = np.minimum(min_cons[:, swh_end_hour:], disagg_cons[:, swh_end_hour:])
        max_cons[:, swh_end_hour:] = np.minimum(max_cons[:, swh_end_hour:], disagg_cons[:, swh_end_hour:])

    return min_cons, mid_cons, max_cons


def wh_seasonality_check(min_cons, mid_cons, max_cons, disagg_cons, item_output_object, app_index, residual_cons, vacation_days):

    """
    Check if wh consumption follows expected seasonality pattern

    Parameters:
        min_cons                  (np.ndarray)       : app min cons
        mid_cons                  (np.ndarray)       : app mid cons
        max_cons                  (np.ndarray)       : app max cons
        disagg_cons               (np.ndarray)       : disagg wh
        item_output_object        (dict)             : Dict containing all hybrid outputs
        app_index                 (int)              : Index of wh app in the appliance list
        residual_cons             (np.ndarray)       : additional wh added from hybrid
        vacation_days             (np.ndarray)       : vacation data

    Returns:
        min_cons                  (np.ndarray)       : app min cons
        mid_cons                  (np.ndarray)       : app mid cons
        max_cons                  (np.ndarray)       : app max cons

    """

    season = item_output_object.get("season")

    season = season.astype(int)

    config = get_inf_config(int(disagg_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)).get("wh")

    block_wh = (np.sum(disagg_cons) == 0) and (vacation_days.sum() > 0.8 * len(vacation_days))

    # opposite seasonality detected in WH output

    block_wh = block_wh or (np.all(disagg_cons == 0) and
                            len(disagg_cons) > config.get('min_days_to_check_seasonality') and
                            np.any(season >= 0) and
                            np.any(season < 0) and
                            np.sum(mid_cons[season >= 0]) > 0 and
                            np.sum(mid_cons[season < 0]) == 0)

    min_cons = min_cons * (1 - int(block_wh))
    mid_cons = mid_cons * (1 - int(block_wh))
    max_cons = max_cons * (1 - int(block_wh))

    item_output_object["inference_engine_dict"]["appliance_conf"][app_index][residual_cons > 0] =  1.1

    return min_cons, mid_cons, max_cons,  item_output_object


def check_seasonality_and_m2m(wh_pot, disagg_cons, item_input_object, item_output_object, residual_cons, possible_others, non_wh_hours):

    """
    This function updates residual WH cons to maintain expected seasonality and consistency in monthly WH output

    Parameters:
        wh_pot                    (np.ndarray)       : weather analytics based WH potential
        disagg_cons               (np.ndarray)       : disagg wh
        item_input_object         (dict)             : Dict containing all hybrid inputs
        item_output_object        (dict)             : Dict containing all hybrid outputs
        residual_cons             (np.ndarray)       : additional wh added from hybrid
        possible_others           (np.ndarray)       : amount of raw energy data that can be added into WH
        non_wh_hours              (np.ndarray)       : inactive hours of the user

    Returns:
        residual_cons             (np.ndarray)       : additional wh added from hybrid

    """

    config = get_inf_config(int(disagg_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)).get("wh")

    date_list = item_output_object.get("date_list")
    month_list = pd.DatetimeIndex(date_list).month.values

    seasonality = np.array([0.2, 0.2, 0.1,0.0,-0.1,-0.2, -0.2,-0.2, -0.2, 0.0, 0.1, 0.2]) * 100

    allowed_deviation = config.get('allowed_deviation')

    # calculate month level WH potential based on weather analytics output

    for bc_idx in range(12):
        if np.any(month_list == (bc_idx + 1)):
            seasonality[bc_idx] = np.round(((wh_pot[month_list == (bc_idx + 1)].sum()) / 100), 2) * 100

    seasonality = np.array(seasonality)
    seasonality = seasonality - np.min(seasonality) + 10

    # preparing list of unique billing cycles

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    unique_bc = unique_bc[counts > 5]

    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    # preparing monthly level scaling factor based on expected seasonality and number of days in the billing cycle

    scaling_factor, monthly_cons = get_monthly_scaling_factor(item_input_object, residual_cons, disagg_cons, vacation_days, month_list, seasonality)

    median_cons = np.median(monthly_cons[monthly_cons > 0])
    median_val = np.median(residual_cons[residual_cons > 0])

    if np.sum(residual_cons > 0) == 0:
        median_val = config.get('default_wh_amp_for_box_addition')

    for bc_idx in range(len(unique_bc)):

        cons = monthly_cons[bc_idx]

        additional_cons = cons - median_cons

        if scaling_factor[bc_idx] != 0:
            additional_cons = additional_cons / scaling_factor[bc_idx]

        # reduce WH residual consumption if extra consumption is present in target billing cycle

        if additional_cons > allowed_deviation and residual_cons[bc_list == unique_bc[bc_idx]].sum() > 0:

            residual_cons = remove_extra_wh_boxes(additional_cons, bc_list, unique_bc, residual_cons, bc_idx)

        # add additional box type WH residual consumption if extra consumption is required in target billing cycle

        if additional_cons < -allowed_deviation:

            temp_cons = copy.deepcopy(possible_others[bc_list == unique_bc[bc_idx]])

            temp_cons = add_boxes_to_maintain_consistency(temp_cons, median_val, disagg_cons,
                                                          additional_cons, vacation_days, non_wh_hours, bc_idx, bc_list, unique_bc)

            residual_cons[bc_list == unique_bc[bc_idx]] = residual_cons[bc_list == unique_bc[bc_idx]] + \
                                                          temp_cons.reshape(residual_cons[bc_list == unique_bc[bc_idx]].shape)

    residual_cons[vacation_days] = 0

    return residual_cons


def check_m2m(wh_pot, disagg_cons, item_input_object, item_output_object, residual_cons, possible_others, non_wh_hours):

    """
    This function updates residual WH cons to maintain consistency in monthly WH output

    Parameters:
        wh_pot                    (np.ndarray)       : weather analytics based WH potential
        disagg_cons               (np.ndarray)       : disagg wh
        item_input_object         (dict)             : Dict containing all hybrid inputs
        item_output_object        (dict)             : Dict containing all hybrid outputs
        residual_cons             (np.ndarray)       : additional wh added from hybrid
        possible_others           (np.ndarray)       : amount of raw energy data that can be added into WH
        non_wh_hours              (np.ndarray)       : inactive hours of the user

    Returns:
        residual_cons             (np.ndarray)       : additional wh added from hybrid

    """

    seasonality = np.array([0.2, 0.2, 0.1,0.0,-0.1,-0.2, -0.2,-0.2, -0.2, 0.0, 0.1,0.2]) * 100

    date_list = item_output_object.get("date_list")
    month_list = pd.DatetimeIndex(date_list).month.values

    config = get_inf_config(int(disagg_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)).get("wh")

    # calculate month level WH potential based on weather analytics output

    for bc_idx in range(12):
        if np.any(month_list == (bc_idx + 1)):
            seasonality[bc_idx] = np.round(((wh_pot[month_list == (bc_idx + 1)].sum()) / 100), 2)

    seasonality = np.array(seasonality)
    seasonality = seasonality - np.min(seasonality) + 10

    # preparing list of unique billing cycles

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    unique_bc = unique_bc[counts > 5]

    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    # preparing monthly level scaling factor based on expected seasonality and number of days in the billing cycle

    scaling_factor, monthly_cons = get_monthly_scaling_factor(item_input_object, residual_cons, disagg_cons, vacation_days, month_list, seasonality)

    allowed_deviation = config.get('allowed_deviation')

    median_val = np.median(residual_cons[residual_cons > 0])

    if np.sum(residual_cons > 0) == 0:
        median_val = config.get('default_wh_amp_for_box_addition')

    for bc_idx in range(0, len(unique_bc)):

        if len(unique_bc) < 2:
            continue

        if bc_idx == 0:
            additional_cons = monthly_cons[bc_idx] - monthly_cons[bc_idx + 1]
        elif bc_idx == (len(unique_bc)-1):
            additional_cons = monthly_cons[bc_idx] - monthly_cons[bc_idx - 1]
        else:
            additional_cons = monthly_cons[bc_idx] - monthly_cons[(bc_idx - 1):(bc_idx +2)].mean()

        if scaling_factor[bc_idx] != 0:
            additional_cons = additional_cons / scaling_factor[bc_idx]

        # reduce WH residual consumption if extra consumption is present in target billing cycle

        if additional_cons > allowed_deviation and residual_cons[bc_list == unique_bc[bc_idx]].sum() > 0:

            residual_cons = remove_extra_wh_boxes(additional_cons, bc_list, unique_bc, residual_cons, bc_idx)

        # add additional box type WH residual consumption if extra consumption is required in target billing cycle

        if additional_cons < -allowed_deviation:

            temp_cons = copy.deepcopy(possible_others[bc_list == unique_bc[bc_idx]])

            temp_cons = add_boxes_to_maintain_consistency(temp_cons, median_val, disagg_cons,
                                                          additional_cons, vacation_days, non_wh_hours, bc_idx, bc_list, unique_bc)

            residual_cons[bc_list == unique_bc[bc_idx]] = residual_cons[bc_list == unique_bc[bc_idx]] + \
                                                          temp_cons.reshape(residual_cons[bc_list == unique_bc[bc_idx]].shape)

    residual_cons[vacation_days] = 0

    return residual_cons


def get_monthly_scaling_factor(item_input_object, residual_cons, disagg_cons, vacation_days, month_list, seasonality):

    """
    This function prepares billing cycle level scaling factor based on expected seasonality and
    number of days in the billing cycle

    Parameters:
        item_input_object         (dict)             : Dict containing all hybrid inputs
        residual_cons             (np.ndarray)       : additional wh added from hybrid
        disagg_cons               (np.ndarray)       : disagg wh
        vacation_days             (np.ndarray)       : vacation data
        month_list                (np.ndarray)       : list of months of all target days
        seasonality               (np.ndarray)       : monthly seasonality trend expected in WH output

    Returns:
        scaling_factor            (np.ndarray)       : scaling factor based on seasonality
        monthly_cons              (np.ndarray)       : scaling factor based on days count

    """

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    unique_bc = unique_bc[counts > 5]

    monthly_cons = np.zeros_like(unique_bc)
    scaling_factor = np.ones_like(unique_bc)

    for i in range(len(unique_bc)):

        monthly_cons[i] = ((residual_cons + disagg_cons)[bc_list == unique_bc[i]].sum() / Cgbdisagg.WH_IN_1_KWH) * \
                          (Cgbdisagg.DAYS_IN_MONTH / np.sum(bc_list == unique_bc[i]))

        if (vacation_days[bc_list == unique_bc[i]].sum() / np.sum(bc_list == unique_bc[i])) < 1:
            monthly_cons[i] = monthly_cons[i] / (1 - (vacation_days[bc_list == unique_bc[i]].sum() / np.sum(bc_list == unique_bc[i])))
            scaling_factor[i] = scaling_factor[i] * (1 - (vacation_days[bc_list == unique_bc[i]].sum() / np.sum(bc_list == unique_bc[i])))
        else:
            monthly_cons[i] = 0
            scaling_factor[i] = 0

        season_val = seasonality[(month_list[bc_list == unique_bc[i]] - 1).astype(int)].mean() / 100

        if season_val == 0:
            season_val = 1
        else:
            season_val = 1 / season_val

        monthly_cons[i] = monthly_cons[i] * season_val

        scaling_factor[i] = scaling_factor[i] * season_val

    return scaling_factor, monthly_cons


def check_min_cons(wh_pot, disagg_cons, item_input_object, item_output_object, residual_cons, possible_others, non_wh_hours, min_cons):

    """
    Check if wh consumption follows expected seasonality pattern

    Parameters:
        wh_pot                    (np.ndarray)       : weather analytics based WH potential
        disagg_cons               (np.ndarray)       : disagg wh
        item_input_object         (dict)             : Dict containing all hybrid inputs
        item_output_object        (dict)             : Dict containing all hybrid outputs
        residual_cons             (np.ndarray)       : additional wh added from hybrid
        possible_others           (np.ndarray)       : amount of raw energy data that can be added into WH
        non_wh_hours              (np.ndarray)       : inactive hours of the user
        min_cons                  (int)              : min WH consumption required

    Returns:
        residual_cons             (np.ndarray)       : additional wh added from hybrid

    """

    date_list = item_output_object.get("date_list")
    month_list = pd.DatetimeIndex(date_list).month.values

    config = get_inf_config(int(disagg_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)).get("wh")

    # calculate month level WH potential based on weather analytics output

    seasonality = np.zeros(12)

    for bc_idx in range(12):

        if np.any(month_list == (bc_idx + 1)):
            seasonality[bc_idx] = np.round(((wh_pot[month_list == (bc_idx + 1)].sum()) / 100), 2) * 100

    if np.max(seasonality) != np.min(seasonality):
        seasonality = (seasonality - np.min(seasonality)) / (np.max(seasonality) - np.min(seasonality)) * 100

    seasonality = np.fmin(100, seasonality + 10)

    # preparing list of unique billing cycles

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    unique_bc = unique_bc[counts > 5]

    monthly_cons = np.zeros_like(unique_bc)

    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    # preparing monthly level scaling factor based on number of days in the billing cycle

    for bc_idx in range(len(unique_bc)):

        monthly_cons[bc_idx] = ((residual_cons + disagg_cons)[bc_list == unique_bc[bc_idx]].sum() / 1000) * (30 / np.sum(bc_list == unique_bc[bc_idx]))

        if (vacation_days[bc_list == unique_bc[bc_idx]].sum() / np.sum(bc_list == unique_bc[bc_idx])) < 1:
            monthly_cons[bc_idx] = monthly_cons[bc_idx] / (1 - (vacation_days[bc_list == unique_bc[bc_idx]].sum() / np.sum(bc_list == unique_bc[bc_idx])))
        else:
            monthly_cons[bc_idx] = 0

    median_val = np.median(residual_cons[residual_cons > 0])

    if np.sum(residual_cons > 0) == 0:
        median_val = config.get('default_wh_amp_for_box_addition')

    for bc_idx in range(len(unique_bc)):

        cons = monthly_cons[bc_idx]

        # increasing residual consumption is monthly consumption is less than a required consumption

        diff = cons - min_cons * seasonality[(month_list[bc_list == unique_bc[bc_idx]] - 1).mean().astype(int)]/100

        if diff < -1:

            temp_cons = copy.deepcopy(possible_others[bc_list == unique_bc[bc_idx]])

            temp_cons = add_boxes_to_maintain_consistency(temp_cons, median_val, disagg_cons, diff, vacation_days,
                                                          non_wh_hours, bc_idx, bc_list, unique_bc)

            residual_cons[bc_list == unique_bc[bc_idx]] = residual_cons[bc_list == unique_bc[bc_idx]] +\
                                                          temp_cons.reshape(residual_cons[bc_list == unique_bc[bc_idx]].shape)

    residual_cons[vacation_days] = 0

    return residual_cons


def add_boxes_to_maintain_consistency(additional_wh_cons, median_val, disagg_cons, additonal_cons,
                                      vacation_days, non_wh_hours, bc_idx, bc_list, unique_bc):

    """
    This function adds cons into WH estimates to maintain consistency

    Parameters:
        additional_wh_cons        (np.ndarray)       : additional WH boxed picked from residual
        median_val                (int)              : WH amplitude
        disagg_cons               (np.ndarray)       : WH disagg output
        additonal_cons            (np.ndarray)       : amount of monthly cons to be added in the WH estimates
        vacation_days             (np.ndarray)       : vacation data
        non_wh_hours              (np.ndarray)       : inactive hours of the user
        bc_idx                    (int)              : current billing cycle index
        bc_list                   (np.ndarray)       : billing cycle data of all target days
        unique_bc                 (np.ndarray)       : list of unique billing cycles

    Returns:
        temp_cons                (np.ndarray)       : additional wh added from hybrid
    """

    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_label = seq_config.SEQ_LABEL
    seq_len = seq_config.SEQ_LEN

    # preparing probable consumption that can be added into WH estimates

    additional_wh_cons[vacation_days[bc_list == unique_bc[bc_idx]]] = 0

    additional_wh_cons[additional_wh_cons < 0.9 * median_val] = 0
    additional_wh_cons[:, non_wh_hours] = 0

    additional_wh_cons = np.fmin(additional_wh_cons, 1.1 * median_val).flatten()
    additional_wh_cons[additional_wh_cons > 0] = median_val

    # preparing sequence of probable WH boxes

    temp_seq = find_seq(additional_wh_cons > 0, np.zeros_like(additional_wh_cons), np.zeros_like(additional_wh_cons), overnight=0)

    samples = int(disagg_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)

    # Removing high duration boxes

    for j in range(len(temp_seq)):
        if temp_seq[j, seq_label] > 0 and temp_seq[j, seq_len] > 4 * samples:
            additional_wh_cons[temp_seq[j, seq_start]: temp_seq[j, seq_end] + 1] = 0

    if np.sum(additional_wh_cons) == 0:
        return additional_wh_cons

    # removing extra boxes from the sequence
    # to maintain total WH consumption similar to what is required to be added

    extra_wh_frac = 1 - min(0.99, (-additonal_cons / (additional_wh_cons.sum() / Cgbdisagg.WH_IN_1_KWH)))

    pot_box_seq = np.where(additional_wh_cons > 0)[0]

    seed = RandomState(random_gen_config.seed_value)

    remove_wh_frac = int((extra_wh_frac) * len(pot_box_seq))
    remove_wh_frac = min(remove_wh_frac, len(pot_box_seq))

    remove_boxes = seed.choice(np.arange(len(pot_box_seq)), remove_wh_frac, replace=False)

    additional_wh_cons[pot_box_seq[remove_boxes]] = 0

    return additional_wh_cons


def remove_extra_wh_boxes(additional_cons, bc_list, unique_bc, residual_cons, bc_idx):

    """
    This function removes additional cons from WH estimates to maintain consistency

    Parameters:
        additonal_cons            (np.ndarray)       : amount of monthly cons to be removed from the WH estimates
        bc_list                   (np.ndarray)       : billing cycle data of all target days
        unique_bc                 (np.ndarray)       : list of unique billing cycles
        residual_cons             (np.ndarray)       : additional WH boxed picked from residual
        bc_idx                    (int)              : current billing cycle index

    Returns:
        residual_cons             (np.ndarray)       : additional wh added from hybrid
    """

    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END

    potential_boxes = residual_cons[bc_list == unique_bc[bc_idx]]

    if np.sum(potential_boxes) == 0:
        return residual_cons

    potential_boxes_1d = residual_cons[bc_list == unique_bc[bc_idx]].flatten()

    factor = additional_cons / (potential_boxes.sum() / 1000)

    # preparing WH residual boxes

    potential_boxes = potential_boxes.flatten()
    pot_box_seq = find_seq(potential_boxes > 0, np.zeros_like(potential_boxes), np.zeros_like(potential_boxes))
    pot_box_seq = pot_box_seq[pot_box_seq[:, 0] > 0]

    seed = RandomState(random_gen_config.seed_value)

    # removing boxes from WH estimates to maintain a maximum consumption

    remove_wh_frac = int((factor) * len(pot_box_seq))
    remove_wh_frac = min(remove_wh_frac, len(pot_box_seq))

    remove_boxes = seed.choice(np.arange(len(pot_box_seq)), remove_wh_frac, replace=False)

    for k in range(len(remove_boxes)):
        potential_boxes_1d[pot_box_seq[remove_boxes[k], seq_start]: pot_box_seq[remove_boxes[k], seq_end] + 1] = 0

    residual_cons[bc_list == unique_bc[bc_idx]] = potential_boxes_1d.reshape(residual_cons[bc_list == unique_bc[bc_idx]].shape)

    return residual_cons

