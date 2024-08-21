"""
Author - Abhinav Srivastava
Date - 22nd Oct 2018
Call the ao smb disagg wrapper and get smb ao results
"""

# Import python packages

import copy
import logging
import scipy
import numpy as np
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import init_hvac_params

from python3.utils.write_estimate import write_estimate
from python3.utils.time.get_time_diff import get_time_diff

from python3.disaggregation.aes.smb_ao.extract_ao_hvac import m2m_stability_on_baseload, apply_ao_seasonality_smb
from python3.disaggregation.aes.smb_ao.compute_baseload_smb import compute_baseload_smb
from python3.disaggregation.aes.smb_ao.compute_baseload_daily_smb import compute_day_level_ao_smb
from python3.utils.validate_output_schema import validate_appliance_profile_schema_for_billcycle


def extract_hsm(disagg_input_object, global_config):
    """Function to extract hsm

    Parameters:
        disagg_input_object (dict) :  Contains disagg related key inputs
        global_config       (dict) :  Contains user related key info

    Returns:
        hsm_in              (dict) : dictionary containing ao hsm
        hsm_fail            (dict) : Contains boolean of whether valid smb is found or not

    """

    # noinspection PyBroadException
    try:

        hsm_dic = disagg_input_object.get('appliances_hsm')
        hsm_in = hsm_dic.get('ao')

    except KeyError:

        hsm_in = None

    hsm_fail = (hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0) and \
               (global_config.get("disagg_mode") == "mtd")

    return hsm_in, hsm_fail


def ao_smb_disagg_wrapper(disagg_input_object, disagg_output_object):
    """
    Function to estimate always on consumption at epoch level

    Parameters:

        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs

    Returns:

        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # Initiate logger for the ao module
    logger_ao_base = disagg_input_object.get('logger').getChild('ao_disagg_wrapper')
    logger_ao = logging.LoggerAdapter(logger_ao_base, disagg_input_object.get('logging_dict'))
    logger_ao_pass = {"logger": logger_ao_base, "logging_dict": disagg_input_object.get("logging_dict")}

    t_ao_start = datetime.now()

    error_list = []

    global_config = disagg_input_object.get('config')

    if global_config is None:
        error_list.append('Key Error: config does not exist')

    input_data = copy.deepcopy(disagg_input_object.get('input_data'))
    if input_data is None:
        error_list.append('Key Error: input data does not exist')

    exit_status = {
        'exit_code': 1,
        'error_list': error_list,
    }

    # to replicate matlab like results

    is_nan_cons = disagg_input_object.get('data_quality_metrics').get('is_nan_cons')
    input_data[is_nan_cons, Cgbdisagg.INPUT_CONSUMPTION_IDX] = np.nan

    # Extract HSM from disagg input object

    hsm_in, hsm_fail = extract_hsm(disagg_input_object, global_config)

    # Extract sampling rate to send as parameter

    sampling_rate = global_config.get('sampling_rate')

    month_epoch, _, month_idx = scipy.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_index=True,
                                             return_inverse=True)

    disagg_output_object['analytics'] = {'required': True}
    disagg_output_object['analytics']['values'] = {}

    disagg_output_object['ao_seasonality'] = {}

    disagg_output_object['ao_seasonality']['cooling'] = np.zeros(len(month_epoch))
    disagg_output_object['ao_seasonality']['heating'] = np.zeros(len(month_epoch))
    disagg_output_object['ao_seasonality']['baseload'] = np.zeros(len(month_epoch))
    disagg_output_object['ao_seasonality']['grey'] = np.zeros(len(month_epoch))

    disagg_output_object['ao_seasonality']['epoch_cooling'] = np.zeros(len(month_idx))
    disagg_output_object['ao_seasonality']['epoch_heating'] = np.zeros(len(month_idx))
    disagg_output_object['ao_seasonality']['epoch_baseload'] = np.zeros(len(month_idx))
    disagg_output_object['ao_seasonality']['epoch_grey'] = np.zeros(len(month_idx))

    if (global_config.get('run_mode') == 'prod' or global_config.get('run_mode') == 'custom') and (not hsm_fail):

        if global_config.get('disagg_mode') == 'historical':

            disagg_output_object['created_hsm']['ao'] = {}

            logger_ao.info(' ------------------- AO : Month Baseload ------------------------ |')
            month_algo_baseload, epoch_algo_baseload, _, exit_status = compute_baseload_smb(input_data, sampling_rate,
                                                                                            logger_ao_pass,
                                                                                            disagg_input_object)

            logger_ao.info(' ------------------- AO : Month Stability ------------------------ |')
            epoch_m2m_baseload, last_baseload, min_baseload = \
                m2m_stability_on_baseload(epoch_algo_baseload[:, 1],
                                          init_hvac_params(sampling_rate, disagg_input_object, logger_ao), logger_ao)

            logger_ao.info(' ------------------- AO : DAY LEVEL ------------------------ |')
            month_algo_ao, epoch_algo_ao, epoch_raw_minus_ao, exit_status = compute_day_level_ao_smb(input_data,
                                                                                                     logger_ao_pass,
                                                                                                     global_config,
                                                                                                     disagg_input_object)

            hsm_update = dict({'timestamp': input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX]})
            hsm_update['attributes'] = {}
            hsm_update['attributes']['last_baseload'] = last_baseload
            hsm_update['attributes']['min_baseload'] = min_baseload
            disagg_output_object['created_hsm']['ao'] = hsm_update

        elif global_config.get('disagg_mode') == 'incremental':

            disagg_output_object['created_hsm']['ao'] = {}

            logger_ao.info(' ------------------- AO incr : Month Baseload ------------------------ |')
            month_algo_baseload, epoch_algo_baseload, _, exit_status = compute_baseload_smb(input_data, sampling_rate,
                                                                                            logger_ao_pass,
                                                                                            disagg_input_object)

            logger_ao.info(' ------------------- AO incr : Month Stability ------------------------ |')
            epoch_m2m_baseload, last_baseload, min_baseload = \
                m2m_stability_on_baseload(epoch_algo_baseload[:, 1],
                                          init_hvac_params(sampling_rate, disagg_input_object, logger_ao), logger_ao)

            logger_ao.info(' ------------------- AO incr : DAY LEVEL ------------------------ |')
            month_algo_ao, epoch_algo_ao, epoch_raw_minus_ao, exit_status = compute_day_level_ao_smb(input_data,
                                                                                                     logger_ao_pass,
                                                                                                     global_config,
                                                                                                     disagg_input_object)

            hsm_update = dict({'timestamp': input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX]})
            hsm_update['attributes'] = {}
            hsm_update['attributes']['last_baseload'] = last_baseload
            hsm_update['attributes']['min_baseload'] = min_baseload
            disagg_output_object['created_hsm']['ao'] = hsm_update

        elif global_config.get('disagg_mode') == 'mtd':

            ao_hsm = disagg_input_object['appliances_hsm']['ao']['attributes']
            last_baseload = ao_hsm['last_baseload']
            min_baseload = ao_hsm['min_baseload']

            logger_ao.info(' ------------------- AO mtd : Month Baseload ------------------------ |')
            month_algo_baseload, epoch_algo_baseload, _, exit_status = compute_baseload_smb(input_data, sampling_rate,
                                                                                            logger_ao_pass,
                                                                                            disagg_input_object)

            logger_ao.info(' ------------------- AO mtd : Month Stability ------------------------ |')
            epoch_m2m_baseload, last_baseload, min_baseload = \
                m2m_stability_on_baseload(epoch_algo_baseload[:, 1],
                                          init_hvac_params(sampling_rate, disagg_input_object, logger_ao), logger_ao,
                                          last_baseload, min_baseload)

            logger_ao.info(' ------------------- AO mtd : DAY LEVEL ------------------------ |')
            month_algo_ao, epoch_algo_ao, epoch_raw_minus_ao, exit_status = compute_day_level_ao_smb(input_data,
                                                                                                     logger_ao_pass,
                                                                                                     global_config,
                                                                                                     disagg_input_object)

        else:

            logger_ao.error('Unrecognized disagg mode %s |', global_config.get('disagg_mode'))

    if not hsm_fail:

        # Code to write results to disagg output object and Column identifier of ao estimates in epoch_algo_ao

        ao_out_idx = disagg_output_object.get('output_write_idx_map').get('ao_smb')
        read_col_idx = 1

        disagg_output_object = write_estimate(disagg_output_object, epoch_algo_ao, read_col_idx, ao_out_idx, 'epoch')
        disagg_output_object = write_estimate(disagg_output_object, month_algo_ao, read_col_idx, ao_out_idx,
                                              'bill_cycle')

        # Writing the monthly output to log
        monthly_output_log = [(datetime.utcfromtimestamp(month_algo_ao[i, 0]).strftime('%b-%Y'),
                               month_algo_ao[i, read_col_idx]) for i in range(month_algo_ao.shape[0])]

        logger_ao.info("The monthly always on consumption (in Wh) is : | %s",
                       str(monthly_output_log).replace('\n', ' '))

        logger_ao.info(' ------------------- AO : Seasonality ------------------------ |')

        apply_ao_seasonality_smb(disagg_input_object, disagg_output_object, epoch_m2m_baseload, global_config,
                                 logger_ao)

    else:
        logger_ao.warning('AO did not run since %s mode required HSM and HSM was missing |',
                          global_config.get('disagg_mode'))

    t_ao_end = datetime.now()
    logger_ao.info('AO Estimation took | %.3f s', get_time_diff(t_ao_start, t_ao_end))

    # Write exit status time taken etc.
    ao_metrics = {
        'time': get_time_diff(t_ao_start, t_ao_end),
        'confidence': 1.0,
        'exit_status': exit_status,
    }

    disagg_output_object['disagg_metrics']['ao'] = ao_metrics

    # Schema Validation for filled appliance profile
    out_bill_cycles = disagg_input_object.get('out_bill_cycles')

    for billcycle_start, _ in out_bill_cycles:
        # TODO(Abhinav): Write your code for filling appliance profile for this bill cycle here
        validate_appliance_profile_schema_for_billcycle(disagg_output_object, billcycle_start, logger_ao_pass)

    return disagg_input_object, disagg_output_object
