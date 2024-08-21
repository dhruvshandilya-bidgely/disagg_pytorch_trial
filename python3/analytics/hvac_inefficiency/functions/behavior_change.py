"""
Author - Anand Kumar Singh
Date - 26th June 2021
HVAC inefficiency HSM related operations
"""

# Import python packages
import copy
import logging
import numpy as np

# Import functions from within the project

from python3.utils.find_runs import find_runs
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params
from python3.analytics.hvac_inefficiency.utils.weighted_kl_div import get_divergence_score
from python3.analytics.hvac_inefficiency.utils.aggregate import get_temperature_dc_relationship


def detect_hvac_behavior_change(input_hvac_inefficiency_object, output_hvac_inefficiency_object, logger_pass, device):

    """
        This function estimates hvac behavior change
        Parameters:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            output_hvac_inefficiency_object     (dict)          dictionary containing all output information
            logger_pass                         (object)        logger object
            device                              (str)           string indicating device, either AC or SH
        Returns:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            output_hvac_inefficiency_object     (dict)          dictionary containing all output information
    """

    static_params = hvac_static_params()

    logger_local = logger_pass.get("logger").getChild("detect_hvac_behavior_change")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    logger.info('Starting HVAC behavior change | {}'.format(device))

    temperature = input_hvac_inefficiency_object.get('temperature')
    day_start = input_hvac_inefficiency_object.get('raw_input_data')[:, Cgbdisagg.INPUT_DAY_IDX]
    timestamp = input_hvac_inefficiency_object.get('raw_input_data')[:, Cgbdisagg.INPUT_EPOCH_IDX]

    raw_input_data = input_hvac_inefficiency_object.get('raw_input_values')
    demand_hvac_col_idx = input_hvac_inefficiency_object.get('raw_input_idx').get('{}_demand'.format(device))
    ao_hvac_col_idx = input_hvac_inefficiency_object.get('raw_input_idx').get('{}_ao'.format(device))

    hvac_consumption =\
        copy.deepcopy(raw_input_data[:, demand_hvac_col_idx] + raw_input_data[:, ao_hvac_col_idx]).reshape(-1, 1)

    cycling_debug_dictionary = output_hvac_inefficiency_object.get(device, {}).get('cycling_debug_dictionary', {})
    fcc = cycling_debug_dictionary.get('full_cycle_consumption')

    sampling_rate = input_hvac_inefficiency_object.get('sampling_rate')

    return_dictionary = {'temperature': temperature, 'compressor': hvac_consumption, 'fcc': fcc}

    if (temperature is None) | (hvac_consumption is None) | (fcc is None):
        logger.info('Both temperature and HVAC consumption not present, skipping behavior change |')
        output_hvac_inefficiency_object[device]['behavior_change'] = return_dictionary
        return input_hvac_inefficiency_object, output_hvac_inefficiency_object

    day_start = np.unique(day_start)
    day_start = np.sort(day_start)

    if sampling_rate == Cgbdisagg.SEC_IN_30_MIN:
        count_allowed = static_params.get('ineff').get('count_allowed_30')
    elif sampling_rate == Cgbdisagg.SEC_IN_15_MIN:
        count_allowed = static_params.get('ineff').get('count_allowed_15')
    else:
        count_allowed = static_params.get('ineff').get('count_allowed_other')

    # Divergence configs

    date_col = 0
    support_col = 2
    divergence_col = 1
    support_threshold = static_params.get('ineff').get('support_threshold')
    minimum_dc_limit = static_params.get('ineff').get('minimum_dc_limit')
    run_length_threshold = static_params.get('ineff').get('run_length_threshold')
    upper_divergence_threshold = static_params.get('ineff').get('upper_divergence_threshold')
    lower_divergence_threshold = static_params.get('ineff').get('lower_divergence_threshold')

    slide_size = static_params.get('ineff').get('slide_size')
    window_size = static_params.get('ineff').get('window_size')
    end = day_start.shape[0] - window_size

    # Scaling HVAC consumption by FCC

    hvac_consumption = hvac_consumption / fcc

    divergence_array = []

    for idx in range(window_size, end, slide_size):

        count_col = 2
        duty_cycle_col = 1

        # Prepare indices

        start_idx = idx - window_size
        end_idx = idx + window_size
        start_timestamp = day_start[start_idx]
        curr_timestamp = day_start[idx]
        end_timestamp = day_start[end_idx]

        # Prepare old behavior array

        valid_old_behavior = (timestamp > start_timestamp) & (timestamp <= curr_timestamp)
        valid_old_temp = temperature[valid_old_behavior]
        valid_old_hvac = hvac_consumption[valid_old_behavior]

        # Prepare new behavior array

        valid_new_behavior = (timestamp > curr_timestamp) & (timestamp <= end_timestamp)
        valid_new_temp = temperature[valid_new_behavior]
        valid_new_hvac = hvac_consumption[valid_new_behavior]

        if (np.nansum(valid_old_hvac) == 0) | (np.nansum(valid_new_hvac) == 0):
            divergence = np.nan
            length = np.nan
            divergence_array.append([curr_timestamp, divergence, length])
            continue

        # Getting duty cycle relationship

        old_behavior = get_temperature_dc_relationship(valid_old_temp, valid_old_hvac)
        new_behavior = get_temperature_dc_relationship(valid_new_temp, valid_new_hvac)

        # filtering temperature with less values

        valid_count_idx = old_behavior[:, count_col] > count_allowed
        old_behavior = old_behavior[valid_count_idx, :]
        valid_count_idx = new_behavior[:, count_col] > count_allowed
        new_behavior = new_behavior[valid_count_idx, :]

        # Filtering temperatures with less than 0.05 duty cycle

        valid_count_idx = old_behavior[:, duty_cycle_col] > minimum_dc_limit
        old_behavior = old_behavior[valid_count_idx, :]
        valid_count_idx = new_behavior[:, duty_cycle_col] > minimum_dc_limit
        new_behavior = new_behavior[valid_count_idx, :]

        divergence, length = get_divergence_score(new_behavior, old_behavior, weight=0.)
        divergence_array.append([curr_timestamp, divergence, length])

    divergence_array = np.array(divergence_array, dtype=np.float)

    if divergence_array.shape[0] == 0:

        return_dictionary.update({'reason_high': 'not enough hvac days'})

    else:

        valid_idx = (divergence_array[:, divergence_col] > upper_divergence_threshold) &\
                    (divergence_array[:, support_col] > support_threshold)

        run_values, run_starts, run_lengths = find_runs(valid_idx)
        valid_run_length = (run_lengths >= run_length_threshold) & run_values

        if np.nansum(valid_run_length) == 0:
            return_dictionary.update({'reason_high': 'not enough high run length'})
        else:
            date_idx = run_starts[valid_run_length]
            dates_with_change = divergence_array[date_idx, date_col]
            return_dictionary.update({'change_date_high': dates_with_change})

        valid_idx = (divergence_array[:, divergence_col] < lower_divergence_threshold) &\
                    (divergence_array[:, support_col] > support_threshold)

        run_values, run_starts, run_lengths = find_runs(valid_idx)
        valid_run_length = (run_lengths >= run_length_threshold) & run_values

        if np.nansum(valid_run_length) == 0:
            return_dictionary.update({'reason_low': 'not enough high run length'})
        else:
            date_idx = run_starts[valid_run_length]
            dates_with_change = divergence_array[date_idx, date_col]
            return_dictionary.update({'change_date_low': dates_with_change})

    return_dictionary.update({'divergence_array': divergence_array, 'upper_threshold': upper_divergence_threshold,
                              'lower_threshold': lower_divergence_threshold})

    output_hvac_inefficiency_object[device]['behavior_change'] = return_dictionary

    return input_hvac_inefficiency_object, output_hvac_inefficiency_object
