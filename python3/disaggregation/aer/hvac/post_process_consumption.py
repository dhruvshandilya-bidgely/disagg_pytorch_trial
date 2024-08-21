"""
Author - Abhinav
Date - 10/10/2018
Post-process hvac estimates and the binary search logic for hvac extraction from residue
"""

# Import python packages
import copy
import logging
import numpy as np
import pandas as pd

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params


def binary_search_for_cut_level(epoch_residue, hvac_contenders_in_month, month_base_residue, residual_to_meet):
    """
    Function to search for the consumption level cut. Consumption over the cut will go into hvac.

    Parameters:
        epoch_residue               (np.ndarray)   : Array containing hvac epoch residues
        hvac_contenders_in_month    (np.ndarray)   : Array containing valid ac/sh points
        month_base_residue          (float)        : The original residue for month
        residual_to_meet            (float)        : The required residue level for the month

    Returns:
        guess                       (float)        : The cut level
        consumption_more_than_guess (np.ndarray)   : Contains boolean of valid residues
    """

    static_params = hvac_static_params()

    residue_low = 0
    residue_high = np.max(epoch_residue[hvac_contenders_in_month])
    guess = (residue_low + residue_high) / 2
    consumption_more_than_guess = epoch_residue > guess

    hvac_contender_residue = epoch_residue[hvac_contenders_in_month & consumption_more_than_guess]

    gap_between_residue = (month_base_residue -
                           np.sum(np.fmax(hvac_contender_residue - guess, 0)) / Cgbdisagg.WH_IN_1_KWH) - residual_to_meet

    # Binary search for where to make the cut in consumption towers
    while abs(gap_between_residue) > static_params.get('bs_gap_between_residue'):

        if gap_between_residue > 0:
            residue_high = guess
        else:
            residue_low = guess

        guess = (residue_low + residue_high) / 2

        consumption_more_than_guess = epoch_residue > guess
        hvac_contender_residue = epoch_residue[hvac_contenders_in_month & consumption_more_than_guess]

        gap_between_residue = (month_base_residue -
                               np.sum(np.fmax(hvac_contender_residue - guess, 0)) / Cgbdisagg.WH_IN_1_KWH) - residual_to_meet

    return guess, consumption_more_than_guess


def filter_x_hour_consumption(mode_x_hour_output, x_hour_net, hvac_params_post_process, logger_base):
    """
    Function to postprocess day level hvac estimates

    Parameters:
        mode_x_hour_output          (np.ndarray)        : 2D array of day level cooling-heating estimates
        x_hour_net                  (np.ndarray)        : Day level aggregate of energy consumption (flowing through hvac)
        hvac_params_post_process    (dict)              : Dictionary containing hvac postprocess related initialized parameters
        logger_base                 (logging object)    : Writes logs during code flow

    Returns:
        mode_x_hour_output          (np.ndarray)        : 2D array of day level cooling-heating estimates based on daily kWh and %
    """

    logger_local = logger_base.get("logger").getChild("filter_daily_consumption")
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # validity check on minimum daily kwh for hvac
    idx = np.logical_or(mode_x_hour_output < hvac_params_post_process['MIN_DAILY_KWH'] * 1000,
                        mode_x_hour_output < hvac_params_post_process['MIN_DAILY_PERCENT'] * x_hour_net)
    mode_x_hour_output[idx] = 0

    logger_hvac.info('post processing daily: ensuring consumption doesnt exceed daily max % and min % energy |')

    # validity check on maximum daily percentages for hvac w.r.t. daily consumption
    idx = mode_x_hour_output > hvac_params_post_process['MAX_DAILY_PERCENT'] * x_hour_net
    mode_x_hour_output[idx] = hvac_params_post_process['MAX_DAILY_PERCENT'] * x_hour_net[idx]

    return mode_x_hour_output


def fill_x_hour_consumption(x_hour_output, hvac_input_data, epoch_filtered_data, idx_identifier, logger_base):
    """
    Function to postprocess processed day level hvac estimates to get estimate epoch level estimates

        Parameters:
            x_hour_output       (np.ndarray)      : Day level estimate of appliance (AC or SH) energy consumption
            hvac_input_data     (np.ndarray)      : 2D Array of epoch level input data frame flowing into hvac module
            epoch_filtered_data (np.ndarray)      : Epoch level energy for AC or SH - After data sanity filtering
            idx_identifier      (np.ndarray)      : Array of day identifier indexes (epochs of a day have same index)
            logger_base         (logging object)  : Writes logs during code flow

        Returns:
            hourly_output       (np.ndarray)      : 2D array of epoch level cooling-heating estimates
    """

    logger_local = logger_base.get("logger").getChild("fill_hourly_consumption")
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # getting daily hour count
    daily_hour_count = np.bincount(idx_identifier, (epoch_filtered_data > 0).astype(int))

    # getting hvac hour count
    hourly_hour_count = daily_hour_count[idx_identifier]
    hourly_hour_count[epoch_filtered_data <= 0] = 0

    epoch_filtered_data[epoch_filtered_data > 0] = hvac_input_data[
        epoch_filtered_data > 0, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # getting daily net ao
    daily_net_bl = np.bincount(idx_identifier, epoch_filtered_data)

    temp_daily_output = copy.deepcopy(x_hour_output)
    temp_daily_net_bl = copy.deepcopy(daily_net_bl)
    temp_daily_net_bl[np.isnan(temp_daily_net_bl)] = 10 ** 10
    logger_hvac.info('post processing for hvac consumption: ensuring daily output doesnt exceed daily net energy |')

    # post processing for hvac consumption: ensuring daily output doesnt exceed daily net energy
    x_hour_output = np.minimum(temp_daily_output, temp_daily_net_bl)
    epoch_hvac_output = x_hour_output[idx_identifier] / hourly_hour_count
    epoch_hvac_output[epoch_filtered_data <= 0] = 0

    return epoch_hvac_output


def get_x_hour_and_epoch_hvac(x_hour_hvac_by_mode, appliance, filter_info, hvac_input_data,
                              idx_identifier, hvac_params, logger_base):
    """
    Function to convert x-hour level hvac estimates to epoch level estimates
        Parameters:
            x_hour_hvac_by_mode     (dict)              : Dict to carry modewise Day level estimate of appliance (AC or SH) energy consumption
            appliance               (str)               : Identifier for appliance type
            filter_info             (dict)              : Epoch level energy for (AC or SH) - After data sanity filtering
            hvac_input_data         (np.ndarray)        : 2D Array of epoch level input data frame flowing into hvac module
            idx_identifier          (np.ndarray)        : Array of day identifier indexes (epochs of a day have same index)
            hvac_params             (dict)              : Dictionary containing hvac algo related initialized parameters
            logger_base             (logging object)    : Writes logs during code flow

        Returns:
            epoch_hvac_total        (np.ndarray)        : 1D array of total epoch level cooling/heating consumption
    """

    logger_local = logger_base.get("logger").getChild("post_process_consumption")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}

    net_at_identifier_level = np.bincount(idx_identifier, hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    hvac_energy_modes = x_hour_hvac_by_mode
    epoch_hvac = pd.DataFrame()

    for mode in hvac_energy_modes.keys():
        mode_x_hour_output = hvac_energy_modes[mode]['hvac_estimate_day']
        mode_x_hour_output = filter_x_hour_consumption(mode_x_hour_output, net_at_identifier_level,
                                                       hvac_params['postprocess'][appliance], logger_pass)

        mode_epoch_filtered_data = np.array(filter_info['epoch_filtered_data'][mode])
        epoch_hvac[mode] = fill_x_hour_consumption(mode_x_hour_output, hvac_input_data, mode_epoch_filtered_data,
                                                   idx_identifier, logger_pass)
    if len(hvac_energy_modes.keys()) == 0:
        epoch_hvac[0] = np.zeros(np.shape(hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])[0])

    epoch_hvac_total = epoch_hvac.sum(axis=1)

    return epoch_hvac_total
