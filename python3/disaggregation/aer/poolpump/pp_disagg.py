"""
Author - Mayank Sharan
Date - 10th Jan 2019
This module runs the pool pump disaggregation and returns consumption value
"""

# Import python packages

import copy
import logging
import numpy as np
from scipy import stats
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.disaggregation.aer.poolpump.functions.clean_edges import clean_edges
from python3.disaggregation.aer.poolpump.functions.smart_union import smart_union
from python3.disaggregation.aer.poolpump.functions.filter_data import filter_data
from python3.disaggregation.aer.poolpump.functions.get_pp_model import get_pp_model
from python3.disaggregation.aer.poolpump.functions.get_gradient import get_gradient
from python3.disaggregation.aer.poolpump.functions.get_day_data import get_day_data
from python3.disaggregation.aer.poolpump.functions.remove_baseload import remove_baseload
from python3.disaggregation.aer.poolpump.functions.get_monthly_estimate import get_monthly_estimate
from python3.disaggregation.aer.poolpump.functions.get_seasonal_segments import get_seasonal_segments
from python3.disaggregation.aer.poolpump.functions.non_maximal_suppression import non_maximal_suppression


def run_type_code(run_type):

    """
    Encoding for the PP run_type

    Parameters:
        run_type          (str)         : Pool pump run type

    Returns:
        run_type          (int)         : Encoded Pool pump run type
    """

    if run_type == 'NoRun':
        run_type = 0
    elif run_type == 'Single':
        run_type = 1
    elif run_type == 'Multiple':
        run_type = 2
    elif run_type == 'Variable':
        run_type = 3

    return run_type


def get_tou_bc_level(month_ts, schedules):

    """
    Function to get BC level TOU

    Parameters:
        month_ts                            (np.ndarray)   : Day wise array containing bill cycle timestamps
        schedules                           (np.ndarray)   : 2 D containing information of pool pump run schedules

    Returns:
        bc_level_schedules_list             (list)         : Bill cycle level pool pump run schedules
    """

    per_day_month_col = stats.mode(month_ts, axis=1)[0]
    ts, _, idx = np.unique(per_day_month_col, return_index=True, return_inverse=True)

    bc_level_schedules_list = list()

    for month_timestamp in ts:
        schedule_list = list()
        schedule_list.append(month_timestamp)
        month_timestamp_end = month_timestamp + 30 * Cgbdisagg.SEC_IN_DAY
        schedules_in_month_idx = np.where((schedules[:, 0] <= month_timestamp_end) & (schedules[:, 3] >= month_timestamp))

        if len(schedules_in_month_idx) == 0:
            schedule_list.append(0)
            bc_level_schedules_list.append(schedule_list)
            continue

        schedule_list.append(schedules[schedules_in_month_idx[0]])
        bc_level_schedules_list.append(schedule_list)

    return bc_level_schedules_list


def pp_disagg(input_data, user_profile_object, pp_config, logger_pass):

    """
    Parameters:
        input_data          (np.ndarray)        : 21 column matrix containing the data
        user_profile_object (dict)              : Dictionary to store Poolpump attributes for AIAAS customer profiling
        pp_config           (dict)              : Dictionary containing all configuration variables for pool pump
        logger_pass         (dict)              : Dictionary containing logging related variables

    Returns:
        monthly_pp          (np.ndarray)        : Billing cycle level Poolpump kWh estimate
        epoch_ts            (np.ndarray)        : ndarray containing epoch timestamps
        data_pp_cons        (np.ndarray)        : epoch level Poolpump consumption
        hsm                 (dict)              : Parameters building house-specific-model
        user_profile_object (dict)              : Dictionary with Poolpump attributes for AIAAS customer profiling
    """

    logger_base = logger_pass.get('logger_base').getChild('pp_disagg')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    t0 = datetime.now()

    sampling_rate = pp_config.get('sampling_rate')

    # Step 1 : Convert input data to to day wise matrix and remove baseload

    day_seasons = get_seasonal_segments(input_data, pp_config)
    month_ts, day_ts, epoch_ts, day_data = get_day_data(input_data, sampling_rate)
    data_bl_removed = remove_baseload(day_data, pp_config)

    t1 = datetime.now()

    logger.debug('Conversion to 2d data and baseload removal took | %.3f s', get_time_diff(t0, t1))

    # Step 2 : Apply min max filter on the data

    data_min_max = filter_data(data_bl_removed, pp_config.get('filtering_pad_rows'), pp_config)
    data_min_max_2 = filter_data(data_bl_removed, pp_config.get('filtering_2_pad_rows'), pp_config)

    num_rows_to_replace = pp_config.get('num_days_from_min_max_2')
    data_min_max[-num_rows_to_replace:, :] = data_min_max_2[-num_rows_to_replace:, :]

    t2 = datetime.now()

    logger.debug('MinMax Filtering with 2 filters took | %.3f s', get_time_diff(t1, t2))

    # Step 3 : Get gradient for data and original data

    data_grad = get_gradient(data_min_max, pp_config)
    data_grad_raw = get_gradient(data_bl_removed, pp_config)

    t3 = datetime.now()

    logger.debug('Gradient of smoothed data and raw data took | %.3f s', get_time_diff(t2, t3))

    # Step 4 : Perform non maximal suppression on the gradient

    data_nms = non_maximal_suppression(data_grad, pp_config.get('zero_val_limit'), pp_config)
    data_nms_raw = non_maximal_suppression(data_grad_raw, pp_config.get('zero_val_limit_raw'), pp_config)

    t4 = datetime.now()

    logger.debug('Non-Maximal Suppression took | %.3f s', get_time_diff(t3, t4))

    # Step 5 : Smart Union to be performed on the non maximal suppressed gradients

    data_smart_union = smart_union(data_nms, data_nms_raw, pp_config)

    t5 = datetime.now()

    logger.debug('Smart Union took | %.3f s', get_time_diff(t4, t5))

    # Step 6 : Merge edges together based on the cleaning procedure

    data_clean_edges = clean_edges(data_smart_union, pp_config)

    t6 = datetime.now()

    logger.debug('Cleaning of Smart Union took | %.3f s', get_time_diff(t5, t6))

    # Step 7 : Get PP model using inputs accumulated so far

    pp_config["input_data"] = data_bl_removed

    data_pp_cons, data_pp_steps, runs_data, structures = get_pp_model(data_clean_edges, data_nms, data_bl_removed,
                                                                      day_seasons, pp_config)

    logger.info('Details about the pool pump run are | %s', str(runs_data).replace('\n', ' '))

    t7 = datetime.now()

    logger.debug('Detection and Estimation of pool pump by pp_model took | %.3f s', get_time_diff(t6, t7))

    confidence_val = runs_data[-1]

    if confidence_val[-1] == 'W':
        logger.info('Pool Pump ran only in Winter, hence blocked | ')
        confidence_value = 0
        data_pp_cons[:] = 0
        data_pp_steps[3][:] = 0
    elif float(confidence_val) < pp_config.get('probability_threshold'):
        logger.info('Pool Pump blocked because of low confidence value | ')
        confidence_value = float(confidence_val)
        data_pp_cons[:] = 0
        data_pp_steps[3][:] = 0
    else:
        confidence_value = float(confidence_val)

    monthly_pp = get_monthly_estimate(month_ts, data_pp_cons)

    t_final = datetime.now()

    logger.debug('Monthly estimate calculation took | %.3f s', get_time_diff(t7, t_final))

    # CODE BEYOND HERE IS FOR NON DEV QA ONLY OPEN TO A LOT OF REFACTORING AND CLEANING BEFORE PRODUCTION

    # Populate HSM variables

    structures = structures.astype(int)
    structures_copy = copy.deepcopy(structures)

    if structures.size > 0:

        num_rows_structure, num_cols_structure = structures.shape

        # Initialising column indices for structure matrix

        indices = np.arange(num_cols_structure)
        idx_value = ['start_day', 'end_day', 'low_speed_start_time_div', 'high_speed_start_time_div',
                     'high_speed_end_time_div', 'low_speed_end_time_div', 'low_speed_start_edge_amp',
                     'high_speed_start_edge_amp', 'high_speed_end_edge_amp', 'low_speed_end_edge_amp',
                     'VSPP_low_speed_start_cons_threshold', 'VSPP_high_speed_cons_threshold',
                     'VSPP_low_speed_end_cons_threshold']
        idx_representation = dict(zip(idx_value, indices))

        for row in range(num_rows_structure):

            structures_copy[row, idx_representation['low_speed_start_time_div']] = \
                epoch_ts[structures[row, idx_representation['start_day']],
                         structures[row, idx_representation['low_speed_start_time_div']]]

            structures_copy[row, idx_representation['high_speed_start_time_div']] = 0
            if structures[row, idx_representation['high_speed_start_time_div']] != -1:
                structures_copy[row, idx_representation['high_speed_start_time_div']] = \
                    epoch_ts[structures[row, idx_representation['start_day']],
                             structures[row, idx_representation['high_speed_start_time_div']]]

            structures_copy[row, idx_representation['high_speed_end_time_div']] = 0
            if structures[row, idx_representation['high_speed_end_time_div']] != -1:
                structures_copy[row, idx_representation['high_speed_end_time_div']] = \
                    epoch_ts[structures[row, idx_representation['end_day']] - 1,
                             structures[row, idx_representation['high_speed_end_time_div']]]

            structures_copy[row, idx_representation['low_speed_end_time_div']] = \
                epoch_ts[structures[row, idx_representation['end_day']] - 1,
                         structures[row, idx_representation['low_speed_end_time_div']]]

        structures_copy = structures_copy[:, 2:]
        num_rows_structure, num_cols_structure = structures_copy.shape
    else:
        num_rows_structure = 0
        num_cols_structure = 0

    structures_1d = np.reshape(structures_copy, newshape=(num_rows_structure * num_cols_structure,))

    run_type = run_type_code(runs_data[0])

    num_of_runs = runs_data[1]

    if pp_config.get('hybrid_conf_val') is None:
        pp_config['hybrid_conf_val'] = int(float(confidence_value) * 100)

    hsm = {
        'timestamp': int(input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX]),
        'confidence': int(float(confidence_value) * 100),
        'hybrid_confidence': int(pp_config['hybrid_conf_val'] * 100),
        'attributes': {
            'start_timestamp': list([int(epoch_ts[0, 0])]),
            'end_timestamp': list([int(input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX])]),
            'schedules': structures_1d,
            'num_schedules': list([num_rows_structure]),
            'num_schedule_params': list([num_cols_structure]),
            'run_type_code': list([run_type]),
            'num_of_runs': list([num_of_runs]),
            'num_samples_per_hr': list([int(Cgbdisagg.SEC_IN_HOUR / pp_config.get('sampling_rate'))]),
            'confidence': int(float(confidence_value) * 100),
            'hybrid_confidence': int(pp_config['hybrid_conf_val'] * 100)
        }
    }

    if hsm.get('attributes').get('run_type_code')[0] == 0 and hsm.get('attributes') is not None:
        hsm['attributes']['schedules'] = [0]

    user_profile_object['pp'] = dict()

    return monthly_pp, epoch_ts, data_pp_cons, hsm, user_profile_object
