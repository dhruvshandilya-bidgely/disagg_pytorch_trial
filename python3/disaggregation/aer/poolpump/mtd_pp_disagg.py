"""
Author - Mayank Sharan
Date - 10th Jan 19
This module runs the pool pump disaggregation and returns consumption value
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# Import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff

from python3.disaggregation.aer.poolpump.functions.clean_edges import clean_edges
from python3.disaggregation.aer.poolpump.functions.filter_data import filter_data
from python3.disaggregation.aer.poolpump.functions.get_gradient import get_gradient
from python3.disaggregation.aer.poolpump.functions.get_day_data import get_day_data
from python3.disaggregation.aer.poolpump.functions.mtd_smart_union import smart_union
from python3.disaggregation.aer.poolpump.functions.mtd_get_pp_model import get_pp_model
from python3.disaggregation.aer.poolpump.functions.remove_baseload import remove_baseload
from python3.disaggregation.aer.poolpump.functions.get_monthly_estimate import get_monthly_estimate
from python3.disaggregation.aer.poolpump.functions.get_seasonal_segments import get_seasonal_segments
from python3.disaggregation.aer.poolpump.functions.non_maximal_suppression import non_maximal_suppression


def mtd_pp_disagg(input_data, user_profile_object, pp_config, hsm_in, logger_pass):

    """
    Parameters:
        input_data          (np.ndarray)        : 21 column matrix containing the data
        pp_config           (dict)              : Dictionary containing all configuration variables for pool pump
        hsm_in              (dict)              : Dictionary containing hsm
        logger_pass         (dict)              : Dictionary containing logging related variables
    Returns:

    """

    logger_base = logger_pass.get('logger_base').getChild('pp_disagg')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    t0 = datetime.now()

    sampling_rate = pp_config.get('sampling_rate')

    # Step 1 : Convert input data to to day wise matrix and remove baseload

    day_seasons = get_seasonal_segments(input_data, pp_config)
    month_ts, day_ts, epoch_ts, day_data = get_day_data(input_data, sampling_rate)
    data_bl_removed = remove_baseload(day_data, pp_config)

    # Form schedule_matrix and extract attributes from hsm

    hsm_asmatrix = np.reshape(hsm_in['attributes']['schedules'],
                              newshape=(int(hsm_in['attributes']['num_schedules'][0]),
                                        int(hsm_in['attributes']['num_schedule_params'][0]))).astype(float)

    num_days = np.ceil(
        (hsm_in['attributes']['end_timestamp'][0] - hsm_in['attributes']['start_timestamp'][0]) / 86400).astype(int)
    num_cols = int(24 * hsm_in['attributes']['num_samples_per_hr'][0])
    num_rows_structure = int(hsm_in['attributes']['num_schedules'][0])
    num_cols_structure = int(hsm_in['attributes']['num_schedule_params'][0])
    run_type = int(hsm_in['attributes']['run_type_code'][0])
    num_of_runs = int(hsm_in['attributes']['num_of_runs'][0])

    # Identify 70 days data and separate into old and new billing cycles

    day_ts_diff = epoch_ts - hsm_in['attributes']['end_timestamp'][0]
    non_overlap_days = np.where(day_ts_diff > 0)[0]
    non_overlap_day_idx = 0
    if len(non_overlap_days) > 0:
        non_overlap_day_idx = non_overlap_days[0]

    overlap_arr = np.zeros(shape=(num_days + (len(day_ts_diff) - non_overlap_day_idx)))
    overlap_arr[(num_days - non_overlap_day_idx):] = 1
    overlap_arr[num_days:] = 2

    # Convert epoch timestamps to numerical day and time_division

    hsm_asmatrix[:, :4] -= hsm_in['attributes']['start_timestamp'][0]
    hsm_asmatrix[hsm_asmatrix < 0] = 0
    hsm_asmatrix[:, :4] = np.divide(hsm_asmatrix[:, :4].astype(float), 86400)

    # Form matrix of schedules stored in hsm

    pairs_matrix = np.zeros(shape=(num_rows_structure, num_cols_structure + 2))
    pairs_matrix[:, -7:] = hsm_asmatrix[:, 4:]

    pairs_matrix[:, 0] = hsm_asmatrix[:, 0].astype(int)
    pairs_matrix[:, 1] = (hsm_asmatrix[:, 3] + 1).astype(int)
    pairs_matrix[:, 2:4] = num_cols * np.modf(hsm_asmatrix[:, 0:2])[0]
    pairs_matrix[:, 4:6] = num_cols * np.modf(hsm_asmatrix[:, 2:4])[0]
    pairs_matrix = np.round(pairs_matrix, 0).astype(int)

    # Create raw_data and edges_matrix from hsm by filling pairs with consumption value

    hsm_matrix = np.zeros(shape=(num_days, num_cols))
    hsm_raw_matrix = np.zeros(shape=(num_days, num_cols))
    num_samples_per_hr = int(3600 / sampling_rate)

    for pair in pairs_matrix:
        # Conditional statements make matrix formation robust to type of PP
        hsm_matrix[pair[0]:pair[1], pair[2]] = pair[6]
        hsm_matrix[pair[0]:pair[1], pair[5]] = -pair[9]
        if pair[7] != 0:
            hsm_matrix[pair[0]:pair[1], pair[3]] = pair[7]
        if pair[8] != 0:
            hsm_matrix[pair[0]:pair[1], pair[4]] = -pair[8]

        if pair[10] != 0 and pair[12] != 0:
            amp_1 = max(np.mean([pair[6], pair[7]]), pair[10] / num_samples_per_hr)
            time_div_range_1 = np.arange(pair[2], pair[2] + (pair[3] - pair[2]) % num_cols + 1) % num_cols
            amp_2 = max(np.mean([pair[7], pair[8]]), pair[11] / num_samples_per_hr)
            time_div_range_2 = np.arange(pair[3], pair[3] + (pair[4] - pair[3]) % num_cols + 1) % num_cols
            amp_3 = max(np.mean([pair[8], pair[9]]), pair[12] / num_samples_per_hr)
            time_div_range_3 = np.arange(pair[4], pair[4] + (pair[5] - pair[4]) % num_cols + 1) % num_cols
            hsm_raw_matrix[pair[0]:pair[1], time_div_range_1] = amp_1
            hsm_raw_matrix[pair[0]:pair[1], time_div_range_2] = amp_2
            hsm_raw_matrix[pair[0]:pair[1], time_div_range_3] = amp_3
        elif pair[10] != 0:
            amp_1 = max(np.mean([pair[6], pair[7]]), pair[10] / num_samples_per_hr)
            time_div_range_1 = np.arange(pair[2], pair[2] + (pair[3] - pair[2]) % num_cols + 1) % num_cols
            amp_2 = max(np.mean([pair[7], pair[9]]), pair[11] / num_samples_per_hr)
            time_div_range_2 = np.arange(pair[3], pair[3] + (pair[5] - pair[3]) % num_cols + 1) % num_cols
            hsm_raw_matrix[pair[0]:pair[1], time_div_range_1] = amp_1
            hsm_raw_matrix[pair[0]:pair[1], time_div_range_2] = amp_2
        elif pair[12] != 0:
            amp_1 = max(np.mean([pair[6], pair[8]]), pair[11] / num_samples_per_hr)
            time_div_range_1 = np.arange(pair[2], pair[2] + (pair[4] - pair[2]) % num_cols + 1) % num_cols
            amp_2 = max(np.mean([pair[8], pair[9]]), pair[12] / num_samples_per_hr)
            time_div_range_2 = np.arange(pair[4], pair[4] + (pair[5] - pair[4]) % num_cols + 1) % num_cols
            hsm_raw_matrix[pair[0]:pair[1], time_div_range_1] = amp_1
            hsm_raw_matrix[pair[0]:pair[1], time_div_range_2] = amp_2
        else:
            amp_1 = max(np.mean([pair[6], pair[9]]), pair[11] / num_samples_per_hr)
            time_div_range_1 = np.arange(pair[2], pair[2] + (pair[5] - pair[2]) % num_cols + 1) % num_cols
            hsm_raw_matrix[pair[0]:pair[1], time_div_range_1] = amp_1

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

    hsm_matrix_copy = copy.deepcopy(hsm_matrix)
    hsm_non_overlap_idx = np.where(overlap_arr == 0)[0]
    hsm_smooth_nms = np.vstack((hsm_matrix_copy[hsm_non_overlap_idx], data_nms))
    hsm_raw_nms = np.vstack((hsm_matrix_copy[hsm_non_overlap_idx], data_nms_raw))

    data_smart_union = smart_union(hsm_smooth_nms, hsm_raw_nms, pp_config, overlap_arr)

    t5 = datetime.now()

    logger.debug('Smart Union took | %.3f s', get_time_diff(t4, t5))

    # Step 6 : Merge edges together based on the cleaning procedure

    data_clean_edges = clean_edges(data_smart_union, pp_config)

    t6 = datetime.now()

    logger.debug('Cleaning of Smart Union took | %.3f s', get_time_diff(t5, t6))

    # Step 7 : Get PP model using inputs accumulated so far
    cons_threshold_arr = hsm_asmatrix[:, -3:]
    hsm_input_dict = dict()
    hsm_input_dict['hsm_matrix'] = hsm_matrix
    hsm_input_dict['hsm_raw_matrix'] = hsm_raw_matrix
    hsm_input_dict['overlap_arr'] = overlap_arr
    hsm_input_dict['cons_threshold_arr'] = cons_threshold_arr
    data_pp_cons, data_pp_steps, runs_data, structures = get_pp_model(hsm_input_dict, data_clean_edges, data_nms,
                                                                      data_bl_removed, day_seasons, run_type,
                                                                      num_of_runs, pp_config)
    logger.info('Details about the pool pump run are | %s', str(runs_data).replace('\n', ' '))

    t7 = datetime.now()

    logger.debug('Detection and Estimation of pool pump by pp_model took | %.3f s', get_time_diff(t6, t7))

    confidence_val = runs_data[-1]

    if confidence_val[-1] == 'W' or float(confidence_val) < 0.45:
        data_pp_cons[:] = 0
        data_pp_steps[3][:] = 0

    monthly_pp = get_monthly_estimate(month_ts, data_pp_cons)

    t_final = datetime.now()

    logger.debug('Monthly estimate calculation took | %.3f s', get_time_diff(t7, t_final))

    user_profile_object['pp'] = dict()

    return monthly_pp, epoch_ts, data_pp_cons, hsm_in, user_profile_object
