"""
Author - Sahana M
Date - 20/05/2021
Module to remove noise in thin & fat pulse Season wise
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.fix_summer import fix_summer
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.post_processing_utils import days_in_billcycle


def get_vacation_days(debug, wh_config):
    """
    Parameters:
        debug                     (dict)            : Dictionary containing WH algorithm output
        wh_config                 (dict)            : Contains all the WH configurations

    Returns:
        vacation_days_bool        (np.ndarray)      : Boolean array marking vacation days as True
    """

    vacation_epochs = debug.get('other_output').get('vacation')
    sampling_rate = wh_config.get('sampling_rate')
    input_data = debug.get('input_data')

    # get the hours in a day

    num_pd_in_day = int(Cgbdisagg.SEC_IN_DAY / sampling_rate)
    pd_mult = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    # Prepare day timestamp matrix and get size of all matrices

    day_ts, row_idx = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)
    day_ts = np.tile(day_ts, reps=(num_pd_in_day, 1)).transpose()

    # Initialize all 2d matrices with default value of nan except the boolean ones

    vacation_data_matrix = np.full(shape=day_ts.shape, fill_value=0.0)

    # Compute hour of day based indices to use

    col_idx = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] - input_data[:, Cgbdisagg.INPUT_DAY_IDX]
    col_idx = col_idx / Cgbdisagg.SEC_IN_HOUR
    col_idx = (pd_mult * (col_idx - col_idx.astype(int) + input_data[:, Cgbdisagg.INPUT_HOD_IDX])).astype(int)

    vacation_data_matrix[row_idx, col_idx] = vacation_epochs

    vacation_days_bool = np.sum(vacation_data_matrix[:, :], axis=1) > 0

    return vacation_days_bool


def seasonal_fat_noise_removal(input_data, thin_signal, fat_signal, seasons_matrix, wh_config, debug):
    """
    This function fixes overestimation of fat pulses in each season
    Arguments:
        input_data          (np.ndarray)        : 21 column matrix containing input
        thin_signal         (np.ndarray)        : 21 column matrix containing thin pulse output
        fat_signal          (np.ndarray)        : 21 column matrix containing fat pulse output
        seasons_matrix      (np.ndarray)        : 2d array containing water heater information for each season
        wh_config           (dict)              : Dictionary containing all the WH configurations
        debug               (dict)              : Debug object

    Returns:
        thin_signal         (np.ndarray)        : 21 column matrix containing thin pulse output
        fat_signal          (np.ndarray)        : 21 column matrix containing fat pulse output
        seasons             (np.ndarray)        : 2d array containing water heater information for each season
    """

    # Extract the necessary data

    fat_output = deepcopy(fat_signal)
    seasons = deepcopy(seasons_matrix)
    thin_output = deepcopy(thin_signal)
    percentage_removal = wh_config['thermostat_wh']['estimation']['percentage_removal']
    avg_consumption_thr = wh_config['thermostat_wh']['estimation']['avg_consumption_thr']

    # Extract vacation days

    vacation_days = debug.get('other_output').get('vacation')
    vacation_days_bool = np.sum(vacation_days[:, :], axis=1) > 0
    factor = int(Cgbdisagg.HRS_IN_DAY*(Cgbdisagg.SEC_IN_HOUR/wh_config.get('sampling_rate')))

    # Get the number of seasons identified

    identified_seasons = np.unique(seasons[:, 2])

    # Get the bill cycle time stamp and their index

    bill_cycle_ts = debug['bill_cycle_ts']
    bill_cycle_idx = debug['bill_cycle_idx']

    # Aggregate the thin and fat consumption at bill cycle level

    thin_monthly = np.bincount(bill_cycle_idx, thin_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    fat_monthly = np.bincount(bill_cycle_idx, fat_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    vacation_monthly = (np.bincount(bill_cycle_idx, vacation_days_bool)/factor).astype(int)

    input_monthly = np.bincount(bill_cycle_idx, input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # Get number of days in each bc

    days_per_bill_cycle = days_in_billcycle(input_data, bill_cycle_ts)

    # Stack monthly thin and fat consumption to seasons table

    seasons = np.hstack((seasons,
                         thin_monthly.reshape(-1, 1),
                         fat_monthly.reshape(-1, 1),
                         input_monthly.reshape(-1, 1),
                         vacation_monthly.reshape(-1,1),
                         days_per_bill_cycle.reshape(-1,1)))

    # If more than 1 season present then perform comparative fixing of fat pulse & thin pulse noise

    perform_fixing = False
    if len(identified_seasons) > 1:
        perform_fixing = True

    if perform_fixing:
        for season_number in identified_seasons:

            # Calculate the average consumption for the current and other bill cycles

            season_bill_cycles = seasons[seasons[:, 2] == season_number, 0]
            avg_consumption = np.mean(seasons[seasons[:, 2] == season_number, 4])
            other_seasons_avg_consumption = np.mean(seasons[seasons[:, 2] != season_number, 4])

            # Get the indexes of bill cycle mapping to each epoch for the current season

            season_bill_cycle_idx = np.where(np.in1d(bill_cycle_ts, season_bill_cycles))[0]
            season_bill_cycle_idx = np.in1d(bill_cycle_idx, season_bill_cycle_idx)

            # Step - 1 : Check if the current seasons avg consumption is greater than 2 times the average of other seasons

            if avg_consumption > 0 and avg_consumption >  avg_consumption_thr * other_seasons_avg_consumption:

                # Extract the fat pulse consumption of the current season

                fat_pulses = fat_output[season_bill_cycle_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX]

                # percentage removal - Identify the % difference (surplus) in consumption between current & other seasons w.r.t
                # the current season and then divide the % by 2 (a value for amplitude adjustment)

                # Adjust the fat pulses with the percentage_removal value

                fat_pulses  = fat_pulses - percentage_removal*fat_pulses
                fat_pulses[fat_pulses < 0] = 0


                fat_output[season_bill_cycle_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = fat_pulses

            # Step - 2 : Make sure that the Summer month consumption < Winter month ideally

            # Check if winter is present or not

            winter_present = np.sum(np.where(identified_seasons == 1, 1, 0)) > 0
            non_vacation_days = 0

            if winter_present:
                # Get the number of non vacation days in winter
                winter_bill_cycles = seasons[seasons[:, 2] == 1, :]
                non_vacation_days = np.min(1 - (winter_bill_cycles[:, 6] / winter_bill_cycles[:, 7]))

            # Perform fixing for summer and intermediate months

            if winter_present and season_number != 1 and non_vacation_days >= 0.3:

                fat_output, thin_output = fix_summer(seasons, bill_cycle_ts, season_bill_cycle_idx, bill_cycle_idx,
                                                     fat_output, thin_output, wh_config)

    fat_signal = fat_output
    thin_signal = thin_output

    return thin_signal, fat_signal
