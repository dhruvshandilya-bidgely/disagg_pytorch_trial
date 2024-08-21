"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
This module contains important checks on the estimation to compensate for missed pulses
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.functions.get_seasonal_segments import season_columns
from python3.disaggregation.aer.waterheater.functions.get_seasonal_segments import get_seasonal_segments
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.missing_pulse_filler import thin_pulse_filler
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.combine_seasons_output import fat_noise_removal
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.seasonal_noise_removal import seasonal_fat_noise_removal
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.seasonal_thin_pulse_filler import seasonal_thin_pulse_filler
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.bill_cycle_consumption_filter import filter_noise_fat_usages
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.bill_cycle_consumption_filter import filter_low_consumption_bill_cycle


def post_process(debug, wh_config, logger_base):
    """
    Parameters:
        debug               (dict)          : Algorithm intermediate steps output
        wh_config           (dict)          : Config params
        logger_base         (dict)          : Logger object

    Returns:
        debug               (dict)          : Algorithm intermediate steps output
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('post_process')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking a deepcopy of input, thin and fat data to keep local instances

    input_data = deepcopy(debug['input_data'])
    thin_signal = deepcopy(debug['final_thin_output'])
    fat_signal = deepcopy(debug['final_fat_output'])

    # Logging thin and fat consumption values before post processing

    thin_cons_before_pp = np.sum(thin_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    fat_cons_before_pp = np.sum(fat_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    logger.info('Thin pulse estimation before post process | {}'.format(thin_cons_before_pp))
    logger.info('Fat pulse estimation before post process | {}'.format(fat_cons_before_pp))
    logger.info('Total estimation before post process | {}'.format(thin_cons_before_pp + fat_cons_before_pp))

    # Check for missing thin & fat pulses for a whole season

    thin_signal, fat_signal = seasonal_thin_pulse_filler(debug, fat_signal, wh_config)

    # Remove Fat pulse over estimation

    fat_signal = fat_noise_removal(fat_signal, thin_signal, wh_config)

    # Thin pulse filter check

    debug = thin_pulse_filler(debug, thin_signal, wh_config, logger)

    # Filter short duration fat usage with outlier energy values

    fat_signal = filter_noise_fat_usages(fat_signal, wh_config)

    # Filter bill cycles with very low consumption, and save to debug

    fat_signal = filter_low_consumption_bill_cycle(fat_signal, wh_config, logger_pass)

    debug['final_fat_output'] = deepcopy(fat_signal)

    # Check if hsm is present and to be used

    if debug['use_hsm']:
        # If MTD run mode

        # Get season info for the given data

        seasons = get_seasonal_segments(input_data, None, debug, logger_pass, wh_config,
                                        monthly=False, return_data=False)

        # Retrieve the bill cycle timestamps and indices

        bill_cycle_ts = debug['bill_cycle_ts']
        bill_cycle_idx = debug['bill_cycle_idx']

        # Aggregate the thin and fat consumption at bill cycle level

        thin_monthly = np.bincount(bill_cycle_idx, thin_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        fat_monthly = np.bincount(bill_cycle_idx, fat_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

        input_monthly = np.bincount(bill_cycle_idx, input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

        # Get number of days in each bc

        days_per_bill_cycle = np.array([])

        # Calculate the number of days in each bill cycle

        for bc in bill_cycle_ts:
            days_current_bc = len(np.unique(input_data[input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == bc,
                                                       Cgbdisagg.INPUT_DAY_IDX]))
            days_per_bill_cycle = np.r_[days_per_bill_cycle, days_current_bc]

        # Stack monthly thin and fat consumption to seasons table

        seasons = np.hstack(
            (seasons, thin_monthly.reshape(-1, 1), fat_monthly.reshape(-1, 1), input_monthly.reshape(-1, 1)))

    else:
        seasons = get_seasonal_segments(input_data, None, debug, logger_pass, wh_config,
                                        monthly=False, return_data=False)

        # Retrieve the bill cycle timestamps and indices

        bill_cycle_ts = debug['bill_cycle_ts']
        bill_cycle_idx = debug['bill_cycle_idx']

        # Fix seasons with fat pulse over estimation

        thin_signal, fat_signal = seasonal_fat_noise_removal(input_data, thin_signal, fat_signal, seasons, wh_config, debug)

        debug['final_fat_output'] = fat_signal
        debug['final_thin_output'] = thin_signal
        debug['final_wh_signal'][:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = fat_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] + \
                                                                       thin_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

        # Aggregate the thin and fat consumption at bill cycle level

        thin_monthly = np.bincount(bill_cycle_idx, thin_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        fat_monthly = np.bincount(bill_cycle_idx, fat_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

        wh_total_monthly = thin_monthly + fat_monthly
        input_monthly = np.bincount(bill_cycle_idx, input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

        # Get number of days in each bc

        days_per_bill_cycle = np.array([])

        for bc in bill_cycle_ts:
            days_current_bc = len(np.unique(input_data[input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == bc,
                                                       Cgbdisagg.INPUT_DAY_IDX]))
            days_per_bill_cycle = np.r_[days_per_bill_cycle, days_current_bc]

        # Stack monthly thin and fat consumption to seasons table

        seasons = np.hstack(
            (seasons, thin_monthly.reshape(-1, 1), fat_monthly.reshape(-1, 1), input_monthly.reshape(-1, 1)))

        debug['final_estimate_before_pp'] = deepcopy(seasons)

        # Convert fractions to percentage (thin of raw) and (fat of raw)

        thin_of_total = 100 * seasons[:, season_columns['thin_monthly']] / seasons[:, season_columns['raw_monthly']]
        fat_of_total = 100 * seasons[:, season_columns['fat_monthly']] / seasons[:, season_columns['raw_monthly']]

        thin_of_wh = 100 * thin_monthly / wh_total_monthly
        fat_of_wh = 100 * fat_monthly / wh_total_monthly

        # Stack percentages to season info

        seasons = np.hstack((seasons, thin_of_total.reshape(-1, 1), thin_of_wh.reshape(-1, 1),
                             fat_of_total.reshape(-1, 1), fat_of_wh.reshape(-1, 1),
                             days_per_bill_cycle.reshape(-1, 1)))


    # Save the final season info (with monthly estimates)

    debug['final_estimate'] = seasons

    if np.sum(fat_monthly) > 0:
        debug['thin_fat_ratio'] = np.sum(thin_monthly)/ np.sum(fat_monthly)
    else:
        debug['thin_fat_ratio'] = 0

    debug['thin_scale_factor'] = wh_config['thermostat_wh']['estimation']['default_scale_factor']
    debug['fat_scale_factor'] = wh_config['thermostat_wh']['estimation']['default_scale_factor']

    # Final water heater monthly consumption is sum of thin and fat monthly

    final_consumption = np.sum(seasons[:, [season_columns['thin_monthly'], season_columns['fat_monthly']]], axis=1)
    debug['wh_estimate'] = np.c_[seasons[:, season_columns['bill_cycle_ts']], final_consumption]

    # Logging values after post processing

    thin_cons_after_pp = np.sum(seasons[:, season_columns['thin_monthly']])
    fat_cons_after_pp = np.sum(seasons[:, season_columns['fat_monthly']])

    # Calculate change in consumption from post processing

    post_process_change = 100 * (fat_cons_after_pp - fat_cons_before_pp) / fat_cons_before_pp

    logger.info('Thin pulse estimation after post process | {}'.format(thin_cons_after_pp))
    logger.info('Fat pulse estimation after post process | {}'.format(fat_cons_after_pp))
    logger.info('Total estimation after post process | {}'.format(thin_cons_after_pp + fat_cons_after_pp))

    logger.info('Estimation change from post process (%) | {}'.format(post_process_change))

    return debug
