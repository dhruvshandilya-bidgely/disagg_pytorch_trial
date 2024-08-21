"""
Author - Mayank Sharan
Date - 13/01/19
Fetches all data for the user using cache and returns in a single dictionary
"""

# Import python packages

import os
import copy
import pickle
import logging
import numpy as np
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.path_constants import PathConstants
from python3.utils.time.get_time_diff import get_time_diff
from python3.utils.weather_utils.weather_data_prep_utils import combine_raw_data_and_weather_data
from python3.initialisation.object_initialisations.init_pipeline_input_objects import init_pipeline_input_objects


def get_disagg_mode_list(data_t0, data_t1, t_start, t_end, bc_arr):

    """
    Utility to get the list disagg_modes to run these timestamps in
    """

    disagg_info = []

    bc_end_with_data = bc_arr[:, 1] <= (data_t1 + Cgbdisagg.SEC_IN_HOUR)
    bc_start_with_data = bc_arr[:, 1] >= data_t0

    complete_bc_bool = np.logical_and(bc_start_with_data, bc_end_with_data)
    complete_bc_arr = bc_arr[complete_bc_bool, :]

    is_mtd_candidate = np.logical_and(t_end < bc_arr[:, 1], t_end > bc_arr[:, 0])

    if np.sum(is_mtd_candidate) == 1:

        if data_t1 > bc_arr[is_mtd_candidate, 0]:

            # The mtd mode is confirmed
            mtd_ev_t1 = min(data_t1, t_end)
            mtd_ev_t0 = bc_arr[is_mtd_candidate, 0]

            disagg_info.append(['MTD', mtd_ev_t0, mtd_ev_t1])

        else:

            t_end = np.max(bc_arr[np.logical_and(bc_end_with_data, t_end > bc_arr[:, 1]), 1])

    if t_start >= t_end:
        return disagg_info

    bc_in_event = np.logical_and(t_start < complete_bc_arr[:, 1], t_end > complete_bc_arr[:, 0])

    if np.sum(bc_in_event) == 1:

        # Confirmed incremental mode

        disagg_info.append(['COMPLETE_BILLING_CYCLE', complete_bc_arr[bc_in_event, 0], complete_bc_arr[bc_in_event, 1]])

    elif np.sum(bc_in_event) > 1:

        # Confirmed historical mode

        disagg_info.append(['HISTORICAL', 0, np.max(complete_bc_arr[bc_in_event, 1])])

    return disagg_info


def compute_prod_pipeline_events(data_t0, data_t1, t_start, t_end, bc_arr, pipeline_events, out_bill_cycles):

    """
    Utility to compute and return pipeline events when running in prod mode
    """

    pipeline_modes_list = get_disagg_mode_list(data_t0, data_t1, t_start, t_end, bc_arr)

    t_data_start = 1600000000
    t_data_end = 0

    if len(pipeline_modes_list) > 0:

        # Create pipeline events and process input data

        for idx in range(len(pipeline_modes_list)):

            pipeline_mode_info = pipeline_modes_list[idx]

            if pipeline_mode_info[0] == 'MTD':
                t_data_start = min(t_data_start, pipeline_mode_info[2] - 70 * Cgbdisagg.SEC_IN_DAY)
                t_data_end = max(t_data_end, pipeline_mode_info[2])

                pipeline_events.append({
                    'start': int(pipeline_mode_info[1]),
                    'end': int(pipeline_mode_info[2]),
                    'rawDataDurationInDays': 70,
                    'disaggMode': pipeline_mode_info[0],
                    'mode': 'DAY',
                })

                out_bill_cycles = \
                    np.r_[out_bill_cycles,
                          np.array([pipeline_mode_info[1], bc_arr[bc_arr[:, 0] == pipeline_mode_info[1], 1]])]

            elif pipeline_mode_info[0] in ['COMPLETE_BILLING_CYCLE', 'HISTORICAL']:

                temp_t0 = pipeline_mode_info[2] - 395 * Cgbdisagg.SEC_IN_DAY
                temp_t0 = bc_arr[np.where(np.logical_and(temp_t0 < bc_arr[:, 1], temp_t0 >= bc_arr[:, 0]))[0], 0]

                if len(temp_t0) == 0:
                    temp_t0 = bc_arr[0, 0]
                elif len(temp_t0) > 1:
                    temp_t0 = np.array([np.min(temp_t0)])

                if pipeline_mode_info[0] == 'HISTORICAL':
                    pipeline_mode_info[1] = temp_t0

                t_data_start = min(t_data_start, temp_t0)
                t_data_end = max(t_data_end, pipeline_mode_info[2])

                pipeline_events.append({
                    'start': int(pipeline_mode_info[1]),
                    'end': int(pipeline_mode_info[2]),
                    'rawDataDurationInDays': (t_data_end - t_data_start) / Cgbdisagg.SEC_IN_DAY,
                    'disaggMode': pipeline_mode_info[0],
                    'mode': 'MONTH',
                })

                out_bc_temp = bc_arr[np.logical_and(bc_arr[:, 0] >= pipeline_mode_info[1],
                                                    bc_arr[:, 1] <= pipeline_mode_info[2]), :]

                out_bill_cycles = np.r_[out_bill_cycles, out_bc_temp]

    return pipeline_events, out_bill_cycles, t_data_start, t_data_end


def prepare_pipeline_events(input_data, pipeline_run_data, fetch_params):

    """
    Parameters:
        input_data          (np.ndarray)        : 21 column input data matrix
        pipeline_run_data     (dict)              : Contains info like bill cycles pilot id etc
        fetch_params        (dict)              : Contains uuid, t_start, t_end and pipeline_mode

    Returns:
        process_str         (str)               : Processing status of pipeline event
        input_data          (np.ndarray)        : 21 column input data matrix
        pipeline_run_data     (dict)              : Contains info like bill cycles pilot id etc
    """

    # Copy the variables to be processed

    proc_input_data = copy.deepcopy(input_data)
    proc_pipeline_run_data = copy.deepcopy(pipeline_run_data)

    # Get the t0 and t1 requested

    t_start = fetch_params.get('t_start')
    t_end = fetch_params.get('t_end')

    if t_end <= t_start:
        return 'Invalid timestamps', np.array([]), {}

    if len(input_data) == 0:
        return 'Empty data', np.array([]), {}

    # Get the span of data we have

    data_t0 = input_data[0, Cgbdisagg.INPUT_EPOCH_IDX]
    data_t1 = input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX]

    # Get bill cycles for the user

    known_bc = pipeline_run_data.get('outputBillCycles')
    bc_arr = []

    for dict_idx in range(len(known_bc)):
        bc_arr.append([known_bc[dict_idx].get('key'), known_bc[dict_idx].get('value')])

    bc_arr = np.array(bc_arr)

    # If run type is prod we create event and modify data as in prod

    pipeline_events = []
    output_bill_cycles = np.zeros(shape=(0, 2))

    t_data_start = 0
    t_data_end = 1600000000

    if fetch_params.get('run_mode') == 'prod':

        pipeline_events, output_bill_cycles, t_data_start, t_data_end = \
            compute_prod_pipeline_events(data_t0, data_t1, t_start, t_end, bc_arr, pipeline_events, output_bill_cycles)

    elif fetch_params.get('run_mode') == 'custom':

        pipeline_mode = 'HISTORICAL'

        if fetch_params.get('disagg_mode') is not None:
            if fetch_params.get('disagg_mode') == 'incremental':
                pipeline_mode = 'COMPLETE_BILLING_CYCLE'
            else:
                pipeline_mode = fetch_params.get('disagg_mode')

        valid_bc_bool = np.logical_and(bc_arr[:, 0] < t_end, bc_arr[:, 1] > t_start)
        output_bill_cycles = bc_arr[valid_bc_bool, :]

        t_data_start = t_start
        t_data_end = t_end

        pipeline_events.append({
            'start': int(t_start),
            'end': int(t_end),
            'rawDataDurationInDays': (t_end - t_start) / Cgbdisagg.SEC_IN_DAY,
            'disagg_Mode': pipeline_mode,
            'mode': 'MONTH',
        })

    # Prepare everything

    proc_input_data = proc_input_data[np.logical_and(proc_input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] >= t_data_start,
                                                     proc_input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] <= t_data_end), :]

    output_bill_cycles = np.sort(output_bill_cycles, axis=1)

    out_bc_arr = []

    for idx in range(output_bill_cycles.shape[0]):

        out_bc_arr.append({
            'key': int(output_bill_cycles[idx, 0]),
            'value': int(output_bill_cycles[idx, 1]),
        })

    proc_pipeline_run_data['gbDisaggEvents'] = pipeline_events
    proc_pipeline_run_data['outputBillCycles'] = out_bc_arr

    if len(pipeline_events) > 0:
        process_str = 'Pipeline event creation successful'
    else:
        process_str = 'Failed in Pipeline event creation'

    return process_str, proc_input_data, proc_pipeline_run_data


def fetch_data_cache(fetch_params, config_params, root_logger):

    """
    Parameters:
        fetch_params        (dict)              : Contains uuid, t_start, t_end and disagg_mode
        config_params       (dict)              : Dictionary with all custom run parameters provided
        root_logger         (logger)            : The root logger from which to get the child logger

    Returns:
        disagg_input_object (dict)              : Dictionary containing all inputs to run the pipeline
    """

    # Initialise logger to be used

    logger_base = root_logger.get('logger').getChild('fetch_data_cache')
    logging_dict = root_logger.get('logging_dict')
    logger_pass = {
        'logger_base': logger_base,
        'logging_dict': logging_dict
    }

    logger = logging.LoggerAdapter(logger_base, logging_dict)

    # Initialise variables for the data check

    uuid = fetch_params.get('uuid')

    # Check for presence of cache

    cache_path = PathConstants.CACHE_DIR + 'data_' + uuid + '.pb'

    if not os.path.exists(cache_path) or fetch_params.get('cache_update'):
        logger.info('Cache load exiting | cache exists - %s, cache_update - %s', str(os.path.exists(cache_path)),
                    str(fetch_params.get('cache_update')))
        return []

    # Load cached data

    t_before_load = datetime.now()

    with open(cache_path, 'rb') as handle:
        data_dict = pickle.load(handle)

    t_after_load = datetime.now()

    logger.info('Data loaded in | %.3f s', get_time_diff(t_before_load, t_after_load))

    # Extract different parts of the cached data

    input_data = data_dict.get('input_data')
    pipeline_run_data = data_dict.get('disagg_run_data')
    home_meta_data = data_dict.get('home_meta_data')
    app_profile = data_dict.get('app_profile')
    hsm_appliances = data_dict.get('hsm_appliances')
    weather_analytics_data = data_dict.get('weather_analytics_data')

    # Combine raw input data and weather data

    input_data = combine_raw_data_and_weather_data(input_data, weather_analytics_data, logger_pass)

    # Prepare input pipeline run data as needed

    process_str, input_data, pipeline_run_data = prepare_pipeline_events(input_data, pipeline_run_data, fetch_params)

    if not(process_str == 'Pipeline event creation successful'):
        logger.info('Pipeline event could not be generated due to | %s', process_str)
        return []

    # Log all pipeline events received to get a better understanding of how the code is going to run

    num_pipeline_events = len(pipeline_run_data.get('gbDisaggEvents'))
    logger.info('Number of pipeline events created | %d', num_pipeline_events)

    for idx in range(num_pipeline_events):
        pipeline_event = pipeline_run_data.get('gbDisaggEvents')[idx]
        logger.info('Pipeline event | %d : %s', idx + 1, str(pipeline_event).replace('\n', ' '))

    # Package inputs together to avoid excessive parameters

    pipeline_object_params = {
        'input_data': input_data,
        'disagg_run_data': pipeline_run_data,
        'home_meta_data': home_meta_data,
        'app_profile': app_profile,
        'hsm_appliances': hsm_appliances,
        'logging_dict': logging_dict,
        'weather_analytics_data': weather_analytics_data,
    }

    # Initialize pipeline input objects

    pipeline_input_objects = init_pipeline_input_objects(fetch_params, pipeline_object_params, config_params, logger_pass)

    return pipeline_input_objects
