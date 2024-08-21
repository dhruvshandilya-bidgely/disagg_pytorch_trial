"""
Author - Sahana M
Date - 15-May-2020
Module to remove havc boxes
"""

# Import python packages
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from numpy.linalg import norm

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import get_day_data_2d


def remove_hvac(in_data, debug):
    """
    Function to find high energy boxes in consumption data

        Parameters:
            in_data                   (np.ndarray)        : Current box data
            debug                     (dict)              : Dict containing hsm_in, bypass_hsm, make_hsm

        Returns:
            debug                     (object)            : Object containing all important data/values as well as HSM

    """

    # Taking local copy of input data

    input_data = deepcopy(in_data)

    # Retrieve the pool pump info

    hvac_output = np.sum(debug['other_output']['hvac'], axis=1)

    input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] -= hvac_output

    input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = np.fmax(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], 0)

    return input_data


def remove_hvac_using_potentials(input_data, debug, ev_config, logger_base, charger_type='l2'):
    """
    Function to remove HVAC utilising the HVAC potentials
    Parameters:
        input_data                  (np.ndarray)            : Input data
        debug                       (Dict)                  : Debug dictionary
        ev_config                   (Dict)                  : EV configurations dictionary
        logger_base                 (Logger)                : Logger object
        charger_type                (string)                : Charger type
    Returns:
        input_data                  (np.ndarray)            : Input data
        debug                       (Dict)                  : Debug dictionary
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('remove_baseload')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Get the 2D matrix

    data_matrices, row_idx, col_idx = get_day_data_2d(input_data, ev_config)

    raw_data_np = data_matrices[Cgbdisagg.INPUT_CONSUMPTION_IDX]
    heat_pot_data = data_matrices[Cgbdisagg.INPUT_HEATING_POTENTIAL_IDX]
    cool_pot_data = data_matrices[Cgbdisagg.INPUT_COOLING_POTENTIAL_IDX]
    rows, cols = raw_data_np.shape
    heat_pot_data = np.asarray(pd.DataFrame(heat_pot_data.flatten()).ffill()).reshape(rows, cols)
    cool_pot_data = np.asarray(pd.DataFrame(cool_pot_data.flatten()).ffill()).reshape(rows, cols)

    # Remove HVAC using HVAC potential

    raw_data_np = remove_hvac_using_potential(raw_data_np, heat_pot_data, cool_pot_data, ev_config, charger_type)
    logger.info('Removal of HVAC using HVAC potentials complete | ')

    # Remove Seasonal HVAC

    if charger_type == 'l2':
        raw_data_np = remove_swh_type_hvac(raw_data_np, heat_pot_data, cool_pot_data, ev_config)
        logger.info('Removal of Seasonal HVAC using HVAC potentials complete | ')

    input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = raw_data_np[row_idx, col_idx]

    return input_data, debug


def get_days_mapping(input_data):
    """
    HVAC Utility function
    Parameters:
        input_data                  (np.ndarray)            : Input data array
    Returns:
        days                        (np.ndarray)            : Days array
    """
    row, col = input_data.shape[0], input_data.shape[1]

    # Get the days mapping

    days = []
    for i in range(row):
        days_row = np.repeat(i, col)
        days.append(days_row)
    days = np.asarray(days)
    days = days.flatten()

    return days


def get_heavy_hvac_percentages(hvac_points, min_potential_value):
    """
    Calculate the heavy hvac usage percentage
    Parameters:
        hvac_points                  (np.ndarray)            : HVAC points
        min_potential_value          (float)                 : Minimum potential value
    Returns:
        heavy_hvac_percentage        (float)                 : Heavy HVAC percentage
    """

    heavy_hvac_percentage = 0
    if len(hvac_points):
        heavy_hvac_percentage = sum(hvac_points >= min_potential_value) / len(hvac_points) * 100

    return heavy_hvac_percentage


def remove_hvac_using_potential(input_data, heat_pot_data, cool_pot_data, ev_config, charger_type):
    """
    Remove HVAC using HVAC potentials
    Parameters:
        input_data                  (np.ndarray)            : Input data
        heat_pot_data               (np.ndarray)            : Heating potential data
        cool_pot_data               (np.ndarray)            : Cooling potential data
        ev_config                   (Dict)                  : EV configurations dictionary
        charger_type                (string)                : Charger type
    Returns:
        input_data                  (np.ndarray)            : Input data
    """

    # Extract the required variables
    cooling_corr_thr = ev_config.get('hvac_removal_configs').get(str(charger_type)).get('cooling_corr_thr')
    heating_corr_thr = ev_config.get('hvac_removal_configs').get(str(charger_type)).get('heating_corr_thr')
    min_potential_value = ev_config.get('hvac_removal_configs').get(str(charger_type)).get('min_potential_value')
    heavy_cooling_percentage_thr = ev_config.get('hvac_removal_configs').get(str(charger_type)).get(
        'heavy_cooling_percentage_thr')
    heavy_heating_percentage_thr = ev_config.get('hvac_removal_configs').get(str(charger_type)).get(
        'heavy_heating_percentage_thr')

    row, col = input_data.shape[0], input_data.shape[1]
    days = get_days_mapping(input_data)

    # Preprocess the Heating and Cooling potential

    heating_pot = deepcopy(np.asarray(pd.DataFrame(heat_pot_data).ffill()).reshape((-1)))
    cooling_pot = deepcopy(np.asarray(pd.DataFrame(cool_pot_data).ffill()).reshape((-1)))

    cooling_pot[np.isnan(cooling_pot)] = 0
    heating_pot[np.isnan(heating_pot)] = 0
    cooling_pot_idx = cooling_pot > 0
    heating_pot_idx = heating_pot > 0

    cooling_consumption = deepcopy(input_data.flatten())
    heating_consumption = deepcopy(input_data.flatten())
    final_cleaned_data = deepcopy(input_data.flatten())
    final_cleaned_data = np.c_[final_cleaned_data, days]

    cooling_consumption[~cooling_pot_idx] = 0
    heating_consumption[~heating_pot_idx] = 0

    # check for high correlation

    cooling_corr = np.corrcoef(cooling_consumption, cooling_pot)
    heating_corr = np.corrcoef(heating_consumption, heating_pot)

    # check for high hvac potential

    heating_points = heating_pot[heating_pot > 0]
    cooling_points = cooling_pot[cooling_pot > 0]

    heavy_heating_percentage = get_heavy_hvac_percentages(heating_points, min_potential_value)
    heavy_cooling_percentage = get_heavy_hvac_percentages(cooling_points, min_potential_value)

    if cooling_corr[0][1] >= cooling_corr_thr or heavy_cooling_percentage >= heavy_cooling_percentage_thr:

        cooling_days = np.unique(final_cleaned_data[cooling_pot_idx, 1])

        # for each cooling day perform cleaning by removing the hvac consumption

        for i in range(len(cooling_days)):

            cooling_day = cooling_days[i]
            cooling_day_idx = final_cleaned_data[:, 1] == cooling_day
            cooling_day_cons = final_cleaned_data[cooling_day_idx, 0]
            cooling_day_cooling_pot = cooling_pot[cooling_day_idx]
            cooling_day_cooling_pot_idx = cooling_day_cooling_pot > 0
            cooling_day_cons_on_cooling_time = cooling_day_cons[cooling_day_cooling_pot_idx]
            cooling_day_cons_non_cooling_time = cooling_day_cons[~cooling_day_cooling_pot_idx]
            cooling_day_cool_pot_on_cooling_time = cooling_day_cooling_pot[cooling_day_cooling_pot_idx]
            cooling_pot_and_cons = np.c_[cooling_day_cool_pot_on_cooling_time, cooling_day_cons_on_cooling_time]

            non_hvac_cons_value = np.min(cooling_day_cons_on_cooling_time)
            if len(cooling_day_cons_non_cooling_time):
                non_hvac_cons_value = np.percentile(cooling_day_cons_non_cooling_time, q=10)

            # get the minimum consumption for each potential value

            cooling_pot_and_cons_arr = get_min_value(cooling_pot_and_cons)

            # removing the hvac
            hvac_removed_cons = cooling_pot_and_cons_arr[:, 1] - cooling_pot_and_cons_arr[:, 2]

            # adding back the baseload
            cooling_day_cons[cooling_day_cooling_pot_idx] = hvac_removed_cons + non_hvac_cons_value

            final_cleaned_data[cooling_day_idx, 0] = cooling_day_cons

    if heating_corr[0][1] >= heating_corr_thr or heavy_heating_percentage >= heavy_heating_percentage_thr:
        heating_days = np.unique(final_cleaned_data[heating_pot_idx, 1])

        # for each heating day perform cleaning by removing the hvac consumption

        for i in range(len(heating_days)):

            heating_day = heating_days[i]
            heating_day_idx = final_cleaned_data[:, 1] == heating_day
            heating_day_cons = final_cleaned_data[heating_day_idx, 0]
            heating_day_heating_pot = heating_pot[heating_day_idx]
            heating_day_heating_pot_idx = heating_day_heating_pot > 0
            heating_day_cons_on_heating_time = heating_day_cons[heating_day_heating_pot_idx]
            heating_day_cons_non_heating_time = heating_day_cons[~heating_day_heating_pot_idx]
            heating_day_heat_pot_on_heating_time = heating_day_heating_pot[heating_day_heating_pot_idx]
            heating_pot_and_cons = np.c_[heating_day_heat_pot_on_heating_time, heating_day_cons_on_heating_time]

            non_hvac_cons_value = np.min(heating_day_cons_on_heating_time)
            if len(heating_day_cons_non_heating_time):
                non_hvac_cons_value = np.percentile(heating_day_cons_non_heating_time, q=10)

            # get the minimum consumption for each potential value
            heating_pot_and_cons_arr = get_min_value(heating_pot_and_cons)

            # removing hvac
            hvac_removed_cons = heating_pot_and_cons_arr[:, 1] - heating_pot_and_cons_arr[:, 2]

            # adding back the baseload
            heating_day_cons[heating_day_heating_pot_idx] = hvac_removed_cons + non_hvac_cons_value

            final_cleaned_data[heating_day_idx, 0] = heating_day_cons

    # Remove seasonal wh type consumption

    input_data = final_cleaned_data[:, 0]
    input_data = input_data.reshape(row, col)

    return input_data


def check_1_hvac_device(delta_diff, hvac_days_cons, min_run_qualified, cosine_similarity, ev_config,
                        device):
    """
    Function to identify a HVAC device
    Parameters:
        delta_diff                      (np.ndarray)            : Consumption difference array
        hvac_days_cons                  (np.ndarray)            : HVAC days consumption
        min_run_qualified               (Boolean)               : Minimum duration qualification
        cosine_similarity               (Float)                 : Usage similarity
        ev_config                       (Dict)                  : EV configurations dictionary
        device                          (String)               : Perform heating/cooling removal
    Returns:
        hvac_device_present             (Boolean)               : HVAC device present or not
        mean_hvac_cons                  (float)                 : Identified HVAC amplitude
    """

    # Extract the required variables

    mean_hvac_cons = 0
    hvac_device_present = False
    min_season_days = ev_config.get('hvac_removal_configs').get('min_season_days')
    hvac_duration = ev_config.get('hvac_removal_configs').get(str(device)).get('hvac_duration')
    cosine_similarity_thr = ev_config.get('hvac_removal_configs').get('cosine_similarity_thr')
    total_hvac_days = ev_config.get('hvac_removal_configs').get(str(device)).get('total_hvac_days')
    min_mean_hvac_cons = ev_config.get('hvac_removal_configs').get(str(device)).get('min_mean_hvac_cons')
    min_hvac_duration = ev_config.get('hvac_removal_configs').get(str(device)).get('min_hvac_duration')

    # Identify the mean HVAC consumption along with the total HVAC days + HVAV Duration

    if np.sum(delta_diff):
        mean_hvac_cons = np.percentile(delta_diff[delta_diff > 0], q=95)
        total_hvac_days = (np.sum(hvac_days_cons >= min_mean_hvac_cons * mean_hvac_cons, axis=1) > 0).sum() / \
                          hvac_days_cons.shape[0]
        hvac_duration = np.sum(delta_diff > 0) / (len(delta_diff) / Cgbdisagg.HRS_IN_DAY)

    # Based on the below conditions qualify a user to have HVAC device

    if min_run_qualified and cosine_similarity >= cosine_similarity_thr and total_hvac_days >= min_season_days \
            and hvac_duration <= min_hvac_duration:
        hvac_device_present = True

    return hvac_device_present, mean_hvac_cons


def check_2_hvac_device(delta_hvac_with_trans_diff, hvac_days_cons, min_run_qualified, ev_config, device):
    """
    Function to check the presence of a HVAC device
    Parameters:
        delta_hvac_with_trans_diff              (np.ndarray)            : HVAC with transition presence only
        hvac_days_cons                          (np.ndarray)            : Only HVAC consumption array
        min_run_qualified                       (Boolean)               : Minimum duration qualified status
        ev_config                               (Dict)                  : EV configurations dictionary
        device                                  (String)                : Perform heating/cooling removal
    Returns:
        hvac_device_present             (Boolean)               : HVAC device present or not
        mean_hvac_cons                  (float)                 : Identified HVAC amplitude
    """

    # Extract the required variables

    mean_hvac_cons = 0
    hvac_device_present = False
    min_hvac_days = ev_config.get('hvac_removal_configs').get('min_hvac_days')
    hvac_duration = ev_config.get('hvac_removal_configs').get(str(device)).get('hvac_duration')
    total_hvac_days = ev_config.get('hvac_removal_configs').get(str(device)).get('total_hvac_days')
    min_mean_hvac_cons = ev_config.get('hvac_removal_configs').get(str(device)).get('min_mean_hvac_cons')
    min_hvac_duration = ev_config.get('hvac_removal_configs').get(str(device)).get('min_hvac_duration')

    # Extract the required variables for HVAC detection

    if np.sum(delta_hvac_with_trans_diff):
        mean_hvac_cons = np.percentile(delta_hvac_with_trans_diff[delta_hvac_with_trans_diff > 0], q=95)
        total_hvac_days = (np.sum(hvac_days_cons >= min_mean_hvac_cons * mean_hvac_cons, axis=1) > 0).sum() / \
                          hvac_days_cons.shape[0]
        hvac_duration = np.sum(delta_hvac_with_trans_diff > 0) / (len(delta_hvac_with_trans_diff) /
                                                                  Cgbdisagg.HRS_IN_DAY)

    # Based on the below conditions qualify a user to have HVAC device

    if min_run_qualified and total_hvac_days >= min_hvac_days and hvac_duration <= min_hvac_duration:
        hvac_device_present = True

    return hvac_device_present, mean_hvac_cons


def remove_detected_hvac(hvac_device_present, mean_hvac_cons, hvac_days_cons, trans_cons, input_data, hvac_pot_days,
                         ev_config, device):
    """
    Function to remove the HVAC
    Parameters:
        hvac_device_present                 (Boolean)           : HVAC device present or not
        mean_hvac_cons                      (np.ndarray)        : Mean hvac consumption
        hvac_days_cons                      (np.ndarray)        : HVAC days consumption
        trans_cons                          (np.ndarray)        : Transition days consumption
        input_data                          (np.ndarray)        : Input data array
        hvac_pot_days                       (np.ndarray)        : HVAC potential days
        ev_config                           (Dict)              : EV config dictionary
        device                              (String)               : Perform heating/cooling removal
    Returns:
        input_data                          (np.ndarray)        : Input data array
    """

    # Extract the required variables

    min_base_hvac_cons_thr = ev_config.get('hvac_removal_configs').get(str(device)).get('min_base_hvac_cons_thr')

    if hvac_device_present:
        # Remove any points that are in the range of the minimum of the mean heating consumption

        min_base_heating_cons = min_base_hvac_cons_thr * mean_hvac_cons

        # Remove the data points above the min heating consumption
        heating_boxes = (hvac_days_cons > min_base_heating_cons)
        heating_boxes = heating_boxes * mean_hvac_cons
        baseload_value = np.mean(trans_cons)
        heating_removed_cons = hvac_days_cons - heating_boxes
        heating_removed_cons[heating_removed_cons < 0] = baseload_value
        heating_days_cons = heating_removed_cons
        input_data[hvac_pot_days] = heating_days_cons

    return input_data


def remove_swh_type_hvac(input_data, heat_pot_data, cool_pot_data, ev_config):
    """
    Remove seasonal wh type FPs
    Parameters:
        input_data                      (np.ndarray)            : Input data array
        heat_pot_data                   (np.ndarray)            : Heating potential array
        cool_pot_data                   (np.ndarray)            : Cooling potential array
        ev_config                       (Dict)                  : EV configurations dict
    Returns:
        input_data                      (np.ndarray)            : Input data array
    """

    # Extract the required variables
    min_sh_amp = ev_config.get('hvac_removal_configs').get('min_sh_amp')
    min_ac_amp = ev_config.get('hvac_removal_configs').get('min_ac_amp')
    min_duration = ev_config.get('hvac_removal_configs').get('min_duration')

    heat_pot_data[np.isnan(heat_pot_data)] = 0
    cool_pot_data[np.isnan(cool_pot_data)] = 0

    heat_pot_data = heat_pot_data * 100
    cool_pot_data = cool_pot_data * 100

    # Identify the days where heating or cooling potential is present
    heating_pot_days = np.sum(heat_pot_data > 50, axis=1) > 0
    cooling_pot_days = np.sum(cool_pot_data > 50, axis=1) > 0
    transition_days = ~(heating_pot_days + cooling_pot_days > 0)

    # Get the season wise consumption
    heating_days_cons = deepcopy(input_data[heating_pot_days])
    cooling_days_cons = deepcopy(input_data[cooling_pot_days])
    transition_cons = deepcopy(input_data[transition_days])

    if len(heating_days_cons) and len(cooling_days_cons) and len(transition_cons):
        # Get 10th percentile consumption
        tenth_perc_trans_cons = np.percentile(transition_cons, q=10, axis=0)

        # Get 90th percentile consumption
        ninety_perc_heating_cons = np.percentile(heating_days_cons, q=90, axis=0)
        ninety_perc_cooling_cons = np.percentile(cooling_days_cons, q=90, axis=0)
        ninety_perc_trans_cons = np.percentile(transition_cons, q=90, axis=0)

        # Find if there is a heating device used in summer by subtracting the heating days cons with cooling days cons

        delta_heating_diff = ninety_perc_heating_cons - ninety_perc_cooling_cons
        # anything less than 2kwh diff can be neglected
        delta_heating_diff[delta_heating_diff < min_sh_amp] = 0

        # As a final step confirm if the difference is present in transition as well
        # get the difference between heating days cons with transition days cons
        delta_heating_with_trans_diff = ninety_perc_heating_cons - ninety_perc_trans_cons
        delta_heating_with_trans_diff[delta_heating_with_trans_diff < min_sh_amp] = 0

        # Check for minimum 1.5 hour usage
        min_run_qualified = np.sum(delta_heating_diff > 0) / delta_heating_diff.shape[0] * Cgbdisagg.HRS_IN_DAY >= \
                            min_duration
        min_run_qualified = min_run_qualified and \
                            (np.sum(delta_heating_with_trans_diff > 0) / delta_heating_with_trans_diff.shape[0] *
                             Cgbdisagg.HRS_IN_DAY >= min_duration)

        # check if the delta between transitions days cons difference with heating days cons & cooling days cons
        # difference with heating days cons

        cosine_similarity = np.round(np.dot(delta_heating_diff, delta_heating_with_trans_diff) /
                                     (norm(delta_heating_diff) * norm(delta_heating_with_trans_diff)), 3)

        heating_device_present, mean_heating_cons = check_1_hvac_device(delta_heating_diff, heating_days_cons,
                                                                        min_run_qualified, cosine_similarity, ev_config,
                                                                        'heating')

        input_data = remove_detected_hvac(heating_device_present, mean_heating_cons, heating_days_cons,
                                          tenth_perc_trans_cons,
                                          input_data, heating_pot_days, ev_config, 'heating')

        # Find if there is a heating device used in summer by subtracting the heating days cons with cooling days cons

        delta_cooling_diff = ninety_perc_cooling_cons - ninety_perc_heating_cons
        # anything less than 2kwh diff can be neglected
        delta_cooling_diff[delta_cooling_diff < min_ac_amp] = 0

        # As a final step confirm if the difference is present in transition as well
        # get the difference between heating days cons with transition days cons
        delta_cooling_with_trans_diff = ninety_perc_cooling_cons - ninety_perc_trans_cons
        delta_cooling_with_trans_diff[delta_cooling_with_trans_diff < 1000] = 0

        # Check for minimum 1.5 hour usage
        min_run_qualified = np.sum(delta_cooling_diff > 0) / delta_cooling_diff.shape[0] * Cgbdisagg.HRS_IN_DAY >= \
                            min_duration
        min_run_qualified = min_run_qualified and \
                            (np.sum(delta_cooling_with_trans_diff > 0) /
                             delta_cooling_with_trans_diff.shape[0] * Cgbdisagg.HRS_IN_DAY >= min_duration)

        # check if the delta between transitions days cons difference with heating days cons & cooling days cons
        # difference with heating days cons

        cosine_similarity = np.round(np.dot(delta_cooling_diff, delta_cooling_with_trans_diff) /
                                     (norm(delta_cooling_diff) * norm(delta_cooling_with_trans_diff)), 3)

        cooling_device_present, mean_cooling_cons = check_1_hvac_device(delta_cooling_diff, cooling_days_cons,
                                                                        min_run_qualified, cosine_similarity, ev_config,
                                                                        'cooling')

        input_data = remove_detected_hvac(cooling_device_present, mean_cooling_cons, cooling_days_cons,
                                          tenth_perc_trans_cons,
                                          input_data, cooling_pot_days, ev_config, 'cooling')

    # If only heating days are present and no cooling days
    if len(cooling_days_cons) == 0 and len(heating_days_cons) and len(transition_cons):
        # Get 10th percentile consumption
        tenth_perc_trans_cons = np.percentile(transition_cons, q=10, axis=0)

        # Get 90th percentile consumption
        ninety_perc_heating_cons = np.percentile(heating_days_cons, q=90, axis=0)
        ninety_perc_trans_cons = np.percentile(transition_cons, q=90, axis=0)

        # Find if there is a heating device used in summer by subtracting the heating days cons with cooling days cons

        # As a final step confirm if the difference is present in transition as well
        # get the difference between heating days cons with transition days cons
        delta_heating_with_trans_diff = ninety_perc_heating_cons - ninety_perc_trans_cons
        delta_heating_with_trans_diff[delta_heating_with_trans_diff < min_sh_amp] = 0

        # Check for minimum 1.5 hour usage
        min_run_qualified = np.sum(delta_heating_with_trans_diff > 0) / delta_heating_with_trans_diff.shape[
            0] * 24 >= min_duration
        min_run_qualified = min_run_qualified and (np.sum(delta_heating_with_trans_diff > 0) /
                                                   delta_heating_with_trans_diff.shape[0] * 24 >= min_duration)

        heating_device_present, mean_heating_cons = check_2_hvac_device(delta_heating_with_trans_diff,
                                                                        heating_days_cons,
                                                                        min_run_qualified, ev_config, 'heating')

        input_data = remove_detected_hvac(heating_device_present, mean_heating_cons, heating_days_cons,
                                          tenth_perc_trans_cons,
                                          input_data, heating_pot_days, ev_config, 'heating')

    # If only cooling days are present and no heating days
    if len(heating_days_cons) == 0 and len(cooling_days_cons) and len(transition_cons):
        # Get 10th percentile consumption
        tenth_perc_trans_cons = np.percentile(transition_cons, q=10, axis=0)

        # Get 90th percentile consumption
        ninety_perc_cooling_cons = np.percentile(cooling_days_cons, q=90, axis=0)
        ninety_perc_trans_cons = np.percentile(transition_cons, q=90, axis=0)

        # As a final step confirm if the difference is present in transition as well
        # get the difference between heating days cons with transition days cons
        delta_cooling_with_trans_diff = ninety_perc_cooling_cons - ninety_perc_trans_cons
        delta_cooling_with_trans_diff[delta_cooling_with_trans_diff < min_ac_amp] = 0

        # Check for minimum 1.5 hour usage
        min_run_qualified = np.sum(delta_cooling_with_trans_diff > 0) / delta_cooling_with_trans_diff.shape[
            0] * Cgbdisagg.HRS_IN_DAY >= min_duration
        min_run_qualified = min_run_qualified and \
                            (np.sum(delta_cooling_with_trans_diff > 0) / delta_cooling_with_trans_diff.shape[0] *
                             Cgbdisagg.HRS_IN_DAY >= min_duration)

        cooling_device_present, mean_cooling_cons = check_2_hvac_device(delta_cooling_with_trans_diff,
                                                                        cooling_days_cons,
                                                                        min_run_qualified, ev_config, 'cooling')

        input_data = remove_detected_hvac(cooling_device_present, mean_cooling_cons, cooling_days_cons,
                                          tenth_perc_trans_cons,
                                          input_data, cooling_pot_days, ev_config, 'cooling')

    return input_data


def get_min_value(hvac_pot_and_cons):
    """
    Get the minimum value for HVAC
    Parameters:
        hvac_pot_and_cons           (np.ndarray)        : HVAC consumption array
    Returns:
        hvac_pot_and_cons           (np.ndarray)        : HVAC consumption array
    """

    unique_potentials = np.unique(hvac_pot_and_cons[:, 0])
    new_hvac_cons = np.zeros(hvac_pot_and_cons.shape[0])

    # For each unique potentials identify the underlying HVAC usage

    for i in range(len(unique_potentials)):
        curr_potential = unique_potentials[i]
        curr_potential_indexes = hvac_pot_and_cons[:, 0] == curr_potential
        curr_pot_min_value = np.min(hvac_pot_and_cons[curr_potential_indexes, 1])
        new_hvac_cons[curr_potential_indexes] = curr_pot_min_value

    hvac_pot_and_cons = np.c_[hvac_pot_and_cons, new_hvac_cons]

    capping_cons = np.percentile(hvac_pot_and_cons[:, 2], q=80)
    hvac_pot_and_cons[hvac_pot_and_cons[:, 2] >= capping_cons, 2] = capping_cons

    return hvac_pot_and_cons
