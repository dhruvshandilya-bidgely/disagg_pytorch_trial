"""
Author - Sahana M
Date - 07/12/2021
This is a fallback function used to calculate weather data features incase the weather analytics module fails
"""

# Import python packages
import logging
import numpy as np

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.waterheater.weather_data_analytics.math_utils import find_seq
from python3.disaggregation.aer.waterheater.functions.timed_japan.timed_wh_config import avg_data_indexes


def get_start_end_idx(boolean_arr, end_idx_exclusive=False):
    """This function get the starting and ending index of all the True value boxes in a boolean array
    Parameters:
        boolean_arr         (np.ndarray)        : Boolean array
        end_idx_exclusive   (np.ndarray)        : Ending indexes to be included or not
    Returns:
        box_start_idx       (List)              : Boxes start index
        box_end_idx         (List)              : Boxes end index
    """

    # Get the start and end indexes as 1 and -1

    box_energy_idx_diff = np.diff(np.r_[0, boolean_arr.astype(int), 0])
    box_start_idx = np.where(box_energy_idx_diff[:-1] > 0)[0]

    # If end idx exclusive then get the end idx directly

    if end_idx_exclusive:
        box_end_idx = np.where(box_energy_idx_diff[1:] < 0)[0]
    else:
        box_end_idx = np.where(box_energy_idx_diff[1:] < 0)[0] + 1

    return box_start_idx, box_end_idx


def mark_s_label(avg_data, winter_temp_thr, summer_temp_thr, rows):
    """
    This function is used to mark each day with its season label
    Parameters:
        avg_data                (np.ndarray)     : Contains weather info of every 30 day division
        winter_temp_thr         (float)          : Winter temperature threshold
        summer_temp_thr         (float)          : Summer temperature threshold
        rows                    (int)            : Number of days present in the data

    Returns:
        s_label                 (np.ndarray)     : Contains season label for each day
        avg_data                (np.ndarray)     : Contains weather info of every 30 day division
        valid_season_bool       (np.ndarray)     : Status on detection of 3 seasons (1. Winter, 2.Summer, 3.Transition)
    """

    # Initialise variables

    valid_season_bool = np.full(shape=3, fill_value=False)
    s_label = np.full(shape=len(np.unique(rows)), fill_value=0)

    for i in range(len(avg_data)):
        start_day = int(avg_data[i, avg_data_indexes['start_day']])
        end_day = int(avg_data[i, avg_data_indexes['end_day']])

        # See if there is winter present

        if avg_data[i, avg_data_indexes['temperature']] <= winter_temp_thr:
            avg_data[i, avg_data_indexes['s_label']] = -1
            valid_season_bool[0] = True

        # See if there is summer present

        elif avg_data[i, avg_data_indexes['temperature']] >= summer_temp_thr:
            avg_data[i, avg_data_indexes['s_label']] = 1
            valid_season_bool[1] = True

        # See if there is transition present

        else:
            avg_data[i, avg_data_indexes['s_label']] = 0
            valid_season_bool[2] = True

        s_label[start_day:end_day] = avg_data[i, avg_data_indexes['s_label']]

    return s_label, avg_data, valid_season_bool


def find_temperatures(avg_data, coldest_month_avg_temp):
    """
    Finding the max winter and transition temperatures
    Parameters:
        avg_data                    (np.ndarray)     : Contains weather info of every 30 day division
        coldest_month_avg_temp      (float)          : Coldest month avg temperature derived from avg_data

    Returns:
        max_winter_temp             (float)          : Max winter temperature
        max_tr_temp                 (float)          : max transition temperature
    """

    # Get the max winter temperature

    temp = avg_data[avg_data[:, avg_data_indexes['s_label']] == -1]
    if len(temp):
        max_winter_temp = np.max(temp[:, avg_data_indexes['temperature']])
    else:
        max_winter_temp = coldest_month_avg_temp

    # Get the max transition temperature

    temp = avg_data[avg_data[:, avg_data_indexes['s_label']] == 0]
    if len(temp):
        max_tr_temp = np.max(temp[:, avg_data_indexes['temperature']])
    else:
        max_tr_temp = np.median(avg_data[:, avg_data_indexes['temperature']])

    max_winter_temp = np.round(max_winter_temp, 2)
    max_tr_temp = np.round(max_tr_temp, 2)

    return max_winter_temp, max_tr_temp


def get_bill_cycle_mapping(total_input_days, days, rows, mean_daily_data):
    """
    This function is used to get the bill cycle mapping to the consumption index
    Parameters:
        total_input_days            (int)       : Total input days in the data
        days                        (int)       : Days in the data
        rows                        (np.ndarray): Row indexes
        mean_daily_data             (np.ndarray): Mean temperature for every day
    Returns:
         bc_mapping                 (np.ndarray): Bill cycle mapped indexes
         bill_cycles                (np.ndarray): Bill cycles
         avg_temp                   (list)      : Average temperature for each day
         start_day                  (list)      : Month starting index
         end_day                    (list)      : Month ending index
         days_count                 (list)      : Number of days in the bill cycle
    """

    bill_cycles = int(total_input_days / days) if (total_input_days % days) == 0 else int(
        (total_input_days / days) + 1)

    # Map bill cycle to epoch indexes

    bc_mapping = []
    j = bc_index = 0
    for i in np.unique(rows):
        if j == days:
            j = 0
            bc_index += 1
        bc_mapping.append([i, bc_index])
        j += 1
    bc_mapping = np.array(bc_mapping)

    # For each bill cycle get the average temperature, days, start and end day

    avg_temp = []
    days_count = []
    start_day = []
    end_day = []
    for month_idx in range(bill_cycles):
        month_bool = bc_mapping[:, 1] == month_idx
        avg_temp.append(np.nanmean(mean_daily_data[month_bool]))
        start_idx, end_idx = get_start_end_idx(month_bool)
        start_day.append(start_idx)
        end_day.append(end_idx)
        days_count.append(int(np.sum(month_bool)))

    return avg_temp, start_day, end_day, days_count


def alternate_weather_data_output(input_data, debug, wh_config, logger_base):
    """
    Alternate weather data analytics module
    Parameters:
        input_data          (np.ndarray)        : Input 21 column array
        debug               (dict)              : Algorithm outputs
        wh_config           (dict)              : WH configurations dictionary
        logger_base         (logger)            : Logger passed
    Returns:
        weather_data_output (dict)              : Weather analytics output
        exit_swh            (Bool)              : Status of weather analytics
    """

    # Initialise logger for this module

    logger_local = logger_base.get('logger').getChild('alternate_weather_data_output')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    exit_swh = False

    try:
        # Extract all the required variables

        rows = debug.get('row_idx')
        cols = debug.get('col_idx')
        days = wh_config.get('days')
        config = wh_config.get('weather_data_configs')
        total_input_days = len(np.unique(rows))
        temperature_data = input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]

        # Forward fill the nans
        mask = np.isnan(temperature_data)
        idx = np.where(~mask, np.arange(mask.shape[0]), 0)
        temperature_data = temperature_data[idx]

        day_wise_temp_data = np.full_like(debug['original_data_matrix'], fill_value=np.nan)

        day_wise_temp_data[rows, cols] = temperature_data
        mean_daily_data = np.nanmean(day_wise_temp_data, axis=1)

        # Get the bill cycles and their mapping

        avg_temp, start_day, end_day, days_count = get_bill_cycle_mapping(total_input_days, days, rows, mean_daily_data)

        class_name = None

        # Initialize thresholds for koppen classification. Adjusted for global warming

        type_a_temp_thr = config.get('type_a_temp_thr')
        type_b_hot_thr = config.get('type_b_hot_thr')
        type_c_cold_temp_min = config.get('type_c_cold_temp_min')
        type_c_cold_temp_max = config.get('type_c_cold_temp_max')
        type_c_hot_temp_min = config.get('type_c_hot_temp_min')
        type_d_cold_temp_max = config.get('type_d_cold_temp_max')
        type_d_hot_temp_min = config.get('type_d_hot_temp_min')
        type_e_max_temp = config.get('type_e_max_temp')

        # Attempt Koppen-Geiger classification class A

        if np.sum(mean_daily_data >= type_a_temp_thr) >= 0.9 * len(mean_daily_data):
            class_name = 'A'

        # Attempt Koppen-Geiger classification class C

        avg_data = np.c_[avg_temp, start_day, end_day, days_count]

        coldest_month_avg_temp = np.min(avg_data[:, avg_data_indexes['temperature']])
        hottest_month_avg_temp = np.max(avg_data[:, avg_data_indexes['temperature']])

        if (type_c_cold_temp_max >= coldest_month_avg_temp >= type_c_cold_temp_min) and (
                hottest_month_avg_temp >= type_c_hot_temp_min) and class_name is None:
            class_name = 'C'

        # Sub classify class C

        if class_name == 'C':

            c_sub_temp_thr = config.get('c_sub_temp_thr')
            perc_thr = config.get('perc_thr')
            num_days = len(mean_daily_data)

            perc_above_thr = float(np.sum(mean_daily_data > c_sub_temp_thr)) / num_days
            perc_below_thr = float(np.sum(mean_daily_data <= c_sub_temp_thr)) / num_days

            if perc_above_thr > perc_thr:
                class_name = 'Ch'
            elif perc_below_thr > perc_thr:
                class_name = 'Ck'

        # Attempt Koppen-Geiger classification class D

        if (coldest_month_avg_temp <= type_d_cold_temp_max) and (
                hottest_month_avg_temp >= type_d_hot_temp_min) and class_name is None:
            class_name = 'D'

        # Attempt Koppen-Geiger classification class E

        if np.sum(avg_data[:, avg_data_indexes['temperature']] <= type_e_max_temp) >= 0.9 * len(
                avg_data) and class_name is None:
            class_name = 'E'

        # Attempt Koppen-Geiger classification class B

        if class_name is None:
            annual_temp_avg = np.nanmean(mean_daily_data)
            class_name = 'Bk'
            if annual_temp_avg >= type_b_hot_thr:
                class_name = 'Bh'

        # Identify the seasons

        winter_set_point = config.get('winter_set_point')
        summer_set_point = config.get('summer_set_point')

        class_offset = {
            'A': 0,
            'B': 0,
            'Bk': 0,
            'Bh': 2,
            'Ch': -3,
            'C': -5,
            'Ck': -3,
            'D': -5,
            'E': -7,
        }

        # Handling different geographies temperature by customizing the wh_usage_threshold with an offset

        winter_temp_thr = min(winter_set_point + class_offset.get(class_name), winter_set_point)
        summer_temp_thr = max(summer_set_point + class_offset.get(class_name), summer_set_point)

        avg_data = np.c_[avg_data, np.zeros(avg_data.shape[0])]

        # Mark each day with a season label

        s_label, avg_data, valid_season_bool = mark_s_label(avg_data, winter_temp_thr, summer_temp_thr, rows)

        # Identify the max_winter_temp & max_tr_temp

        seq_arr = find_seq(s_label, min_seq_length=5)
        max_winter_temp, max_tr_temp = find_temperatures(avg_data, coldest_month_avg_temp)

        # Get Feels like temperature converted to a 24 column matrix

        factor = int(debug['factor'])
        day_wise_data = np.full(shape=(day_wise_temp_data.shape[0], Cgbdisagg.HRS_IN_DAY), fill_value=0.0)
        for i in range(0, day_wise_temp_data.shape[1], factor):
            j = int(i / factor)
            day_wise_data[:, j] = np.nanmean(day_wise_temp_data[:, i:i + factor], axis=1)

        # Get cooling potential

        cooling_potential = np.full_like(day_wise_temp_data, fill_value=0.0)
        if valid_season_bool[1]:
            cooling_days = mean_daily_data >= summer_temp_thr
            cooling_potential[cooling_days] = 1
            cooling_min_thr = summer_temp_thr
        else:
            cooling_min_thr = 'NA'

        # Store all the info

        weather_data_output = dict()
        weather_data_output['weather'] = {
            'day_wise_data': {
                'fl': day_wise_data,
            },
            'season_detection_dict': {
                's_label': s_label,
                'seq_arr': seq_arr,
                'max_winter_temp': max_winter_temp,
                'max_tr_temp': max_tr_temp,
                'class_name': class_name,
                'model_info_dict': {
                    'valid_season_bool': valid_season_bool
                }
            },
            'hvac_potential_dict': {
                'cooling_pot': cooling_potential,
                'cooling_min_thr': cooling_min_thr
            }
        }

        logger.info('Koppen class for the user is | {}'.format(class_name))
        logger.info('Max winter temperature | {}'.format(max_winter_temp))
        logger.info('Max transition temperature is | {}'.format(max_tr_temp))

    except Exception:

        exit_swh = True
        logger.warning('Exiting Alternate Weather analytics module | ')
        weather_data_output = {}

    return weather_data_output, exit_swh
