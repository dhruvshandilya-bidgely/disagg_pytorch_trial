"""
Author: Neelabh Goyal
Date:   14 June 2023
Call the work hour disaggregation module and get results
"""

# Import python packages

import copy
import logging
import numpy as np
import pandas as pd
from scipy.stats import mode
from datetime import datetime
from scipy.stats import spearmanr

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff
from python3.disaggregation.aes.hvac.get_smb_params import get_smb_params
from python3.disaggregation.aes.work_hours.smb_work_hours import get_work_hours
from python3.disaggregation.aes.work_hours.get_work_hour_params import get_work_hour_params
from python3.disaggregation.aes.lighting.hourglass_lighting_estimation import external_lighting_estimation

from python3.disaggregation.aes.work_hours.work_hour_utils import update_hsm
from python3.disaggregation.aes.work_hours.work_hour_utils import get_input_data
from python3.disaggregation.aes.work_hours.work_hour_utils import remove_external_lighting


def extract_hsm(disagg_input_object, global_config):
    """Function to extract hsm

    Parameters:
        disagg_input_object (dict) :  Contains disagg related key inputs
        global_config       (dict) :  Contains user related key info

    Returns:
        hsm_in              (dict) : dictionary containing ao hsm
        hsm_fail            (dict) : Contains boolean of whether valid smb is found or not

    """

    try:

        hsm_dic = disagg_input_object.get('appliances_hsm')
        hsm_in = hsm_dic.get('ao')
        # Check if the new AO HSM with work hour attributes is present for the user
        hsm_work_hour = hsm_in.get('attributes').get('user_work_hour_arr')

    except (KeyError, TypeError, AttributeError):

        hsm_in = None
        hsm_work_hour = None

    # Check if the hsm exists especially when the run modes are incremental and mtd
    hsm_fail = (hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0 or hsm_work_hour is None) and \
               (global_config.get("disagg_mode") in ["incremental", "mtd"])

    # Mark the hsm_fail true whenever the run mode is historical
    hsm_fail = hsm_fail or global_config.get("disagg_mode") == "historical"

    return hsm_in, hsm_fail


def write_work_hour_day_info(input_df, open_close, time_array, disagg_output_object):
    """
    Function to create, populate and store day level work hour info ibject
    Args:

        input_df             (pd.DataFrame)  : Dataframe with raw data
        open_close           (np.array)      : 2D numpy array with boolean work hours
        time_array:          (np.array)      : 1D array of strings, containing human readable time
        disagg_output_object (dict)          : Dictionary with pipeline level output data

    Returns:
        disagg_output_object (dict)          : Dictionary with pipeline level output data
    """

    smb_day_info = {}
    billing_cycle_by_day = input_df.pivot_table(index='date', columns='time', values='month').values[:, 0]

    unique_billing_cycles = np.unique(input_df.iloc[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX])

    for billing_cycle in unique_billing_cycles:

        billing_cycle_days = billing_cycle_by_day == billing_cycle
        # getting work hour density
        work_hour_density = np.sum(open_close[billing_cycle_days], axis=0)

        smb_info = {'open': time_array[0],
                    'close': time_array[0],
                    'time_array': time_array,
                    'open_close_table': open_close[billing_cycle_days]}

        if not np.sum(work_hour_density) == 0:
            # getting back shift and taking diff of values
            shift = np.append(np.array(work_hour_density[1:]), 0)
            diff = work_hour_density - shift
            diff[-1] = 0

            # getting general opening and closing time
            smb_info['open'] = time_array[np.where(diff == np.min(diff))[0][0] + 1]
            smb_info['close'] = time_array[np.where(diff == np.max(diff))[0][0] + 1]

        smb_day_info[billing_cycle] = smb_info

    disagg_output_object['special_outputs']['smb'] = smb_day_info

    return disagg_output_object


def post_process_for_external_light(input_data_raw, open_close, disagg_output_object, sampling, disagg_input_object,
                                    static_params, logger_work_hour_pass):
    """
    Function to check for incorrect detection of external light as overnight work hours
    Parameters:
        input_data_raw          (pd.DataFrame)  : Dataframe with raw data
        open_close              (np.array)      : 2D numpy array with boolean work hours
        disagg_output_object    (dict)          : Dictionary with pipeline level output data
        sampling                (int)           : Integer denoting sampling rate of the user's consumption data
        disagg_input_object     (dict)          : Dictionary with pipeline level input data
        static_params           (dict)          : Dictionary containing work hour specific parameters
        logger_work_hour_pass   (logging object): Logging object to log steps in the function

    Returns:
        input_data_raw          (pd.DataFrame)  : Dataframe with post processed raw data
    """
    logger_local = logger_work_hour_pass.get("logger").getChild("work_hour_external_li_post_process")
    logger_pass = {"logger": logger_local, "logging_dict": logger_work_hour_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_work_hour_pass.get("logging_dict"))

    post_process = False
    func_params = static_params.get('post_process_for_ext_light')

    min_night_hours = func_params.get('min_night_work_hours') * sampling
    # Check if work hour has been detected in night hours b/w 6 pm and 6 am
    if (np.mean(np.sum(open_close[:, :func_params.get('morning_hour') * sampling], axis=1)) > min_night_hours or
            np.mean(np.sum(open_close[:, func_params.get('evening_hour') * sampling:], axis=1)) > min_night_hours):

        ext_li_idx = disagg_output_object.get('output_write_idx_map').get('li_smb')

        max_morning_hour = func_params.get('morning_hour_end') * sampling
        min_eve_hour = func_params.get('evening_hour_start') * sampling

        start_indices = np.argmax((open_close[:, :-1] - open_close[:, 1:])[:, :max_morning_hour], axis=1)
        end_indices = np.argmin((open_close[:, :-1] - open_close[:, 1:])[:, min_eve_hour:], axis=1) + min_eve_hour
        valid_days = np.logical_and(start_indices >= 0, end_indices > 15 * sampling)

        # Condition to check if count of days with overnight SMB is at least 10 or 60% of total valid days
        if np.nansum(valid_days) > np.maximum(func_params.get('min_valid_days'),
                                              valid_days.size * func_params.get('min_valid_days_frac')):

            start_indices[valid_days] = pd.DataFrame(start_indices[valid_days]).ewm(span=20).mean().values[:, 0]
            end_indices[valid_days] = pd.DataFrame(end_indices[valid_days]).ewm(span=20).mean().values[:, 0]

            sunrise = disagg_output_object['hourglass_data']['sunrise']
            sunset = disagg_output_object['hourglass_data']['sunset']

            corr1, _ = spearmanr(start_indices[valid_days], sunrise[valid_days])
            corr1 = np.round(corr1, 2)
            corr2, _ = spearmanr(end_indices[valid_days], sunset[valid_days])
            corr2 = np.round(corr2, 2)

            start_indices_rounded = np.round(start_indices[valid_days], 0)
            end_indices_rounded = np.round(end_indices[valid_days], 0)

            straight_edges = mode(start_indices_rounded)[1][0] > len(start_indices_rounded) * 0.5 \
                             and mode(end_indices_rounded)[1][0] > len(end_indices_rounded) * 0.5

            if np.maximum(corr1, corr2) >= func_params.get('corr_thresh') and not straight_edges:
                post_process = True
                start_indices = np.nan_to_num(start_indices).astype(float)
                end_indices = np.nan_to_num(end_indices).astype(float)

                start_indices[valid_days] = pd.DataFrame(start_indices[valid_days]).ewm(span=30).mean().values[:, 0]
                end_indices[valid_days] = pd.DataFrame(end_indices[valid_days]).ewm(span=30).mean().values[:, 0]

                start_indices[~(start_indices > 0)] = np.NaN

                hourglass_data = {'morning_hours': start_indices,
                                  'evening_hours': end_indices,
                                  'sampling': sampling}

                logger.info(' Identifying AO HVAC |')
                ao_hvac = disagg_output_object['ao_seasonality']['epoch_cooling'] + \
                          disagg_output_object['ao_seasonality']['epoch_heating']

                external_light = external_lighting_estimation(input_data_raw, hourglass_data, ao_hvac, False, logger_pass)

                external_light = np.nan_to_num(external_light).reshape(-1, 1)[:, 0]
                input_df = disagg_input_object['input_df']
                epoch_df = input_df.pivot_table(index='date', columns=['time'], values='epoch', aggfunc=np.min)
                epochs = epoch_df.values.flatten()
                external_light = external_light.flatten()
                _, idx_mem_1, idx_mem_2 = np.intersect1d(epochs, input_df['epoch'], return_indices=True)

                disagg_output_object['epoch_estimate'][idx_mem_2, ext_li_idx] = external_light[idx_mem_1]

                input_data_raw['raw-ao'] = input_data_raw['consumption'] - disagg_output_object['ao_seasonality'][
                    'epoch_baseload'] - disagg_output_object['epoch_estimate'][:, ext_li_idx]

                # Condition to identify epochs where hourglass is estimated
                hourglass_condition = disagg_output_object['epoch_estimate'][:, ext_li_idx] > 0

                # Condition to identify epochs where hourglass is estimated and the residue is less than 200Wh
                hourglass_condition = np.logical_and(hourglass_condition, input_data_raw['raw-ao'] < static_params.get('min_hourglass_residue') / sampling)

                # Suppress data points where hourglass was detected and the residue is <200Wh
                input_data_raw['raw-ao'][hourglass_condition] = 0

    return post_process, input_data_raw


def identify_cons_level(input_data_raw, smb_params, sampling, disagg_output_object):
    """

    Parameters:
        input_data_raw          (pd.DataFrame)  : Dataframe with raw data
        smb_params              (dict)          : Dictionary containing SMB specific constants
        sampling                (int)           : Integer to denote count of samples per hour
        disagg_output_object    (dict)          : Dictionary with pipeline level output data

    Returns:
        cons_level              (str)           : String to denote the consumption level of the user

    """
    # getting raw minus always on component of consumption
    input_data_raw['raw-ao'] = input_data_raw['consumption'] - disagg_output_object['ao_seasonality']['epoch_baseload']

    consumption_metric = np.median(input_data_raw['raw-ao']) + \
                         (smb_params.get('month_info').get('low_energy_std_arm') * np.std(input_data_raw['raw-ao']))

    low_consumption_smb = consumption_metric < (smb_params.get('month_info').get('low_energy_smb_limit') / sampling)

    if low_consumption_smb:
        cons_level = 'low'

        if consumption_metric < smb_params.get('month_info').get('low_energy_smb_limit') / 2:
            cons_level = 'v_low'

    elif consumption_metric > smb_params.get('month_info').get('high_energy_smb_limit'):
        cons_level = 'high'

    else:
        cons_level = 'mid'

    return cons_level, input_data_raw


def work_hour_smb_disagg_wrapper(disagg_input_object, disagg_output_object):
    """
    Function to detect work hours

    Parameters:
        disagg_input_object         (dict)              : Dictionary containing all inputs
        disagg_output_object        (dict)              : Dictionary containing all outputs

    Returns:
        disagg_input_object         (dict)              : Dictionary containing all inputs
    """

    # initializing logging object

    logger_work_hour_base = disagg_input_object.get('logger').getChild('work_hour_smb_disagg_wrapper')
    logger_work_hour = logging.LoggerAdapter(logger_work_hour_base, disagg_input_object.get('logging_dict'))
    logger_work_hour_pass = {"logger": logger_work_hour_base, "logging_dict": disagg_input_object.get("logging_dict")}

    t_work_hour_start = datetime.now()

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

    # Extract HSM from disagg input object
    hsm_in, hsm_fail = extract_hsm(disagg_input_object, global_config)

    smb_params = get_smb_params()

    smb_type = str(disagg_input_object.get('home_meta_data').get('smb_type')).upper()
    survey_work_hours = disagg_input_object.get('home_meta_data').get('businessTimings').get('businessDays')
    input_data_raw = get_input_data(disagg_input_object)
    sampling_rate = disagg_input_object.get('config').get('sampling_rate')
    sampling = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    cons_level, input_data_raw = identify_cons_level(input_data_raw, smb_params, sampling, disagg_output_object)
    logger_work_hour.info(" User's consumption level for work hour has been calculated as | {}".format(cons_level))

    input_data_raw, external_lighting = remove_external_lighting(input_data_raw, sampling, disagg_output_object)
    logger_work_hour.info(' Removed External Lighting estimates from raw data | ')

    # initializing switch carrier object specific to smb
    logger_work_hour.info("Initializing switch object for smb work hour |")
    disagg_input_object['switch'] = dict()
    disagg_input_object['switch']['smb'] = dict()
    disagg_input_object['switch']['smb']['open_close_table'] = np.array([])

    # SMB V2.0 Improvement ##

    input_df = disagg_input_object['input_df']
    epoch_df = input_df.pivot_table(index='date', columns='time', values='epoch', aggfunc=np.min)
    open_close = np.zeros_like(epoch_df)
    time_array = np.unique(input_data_raw['time'])
    disagg_output_object['alt_open_close_table'] = np.zeros_like(epoch_df)
    static_params = get_work_hour_params()

    if global_config.get("disagg_mode") in ["historical", "incremental"]:
        open_close, valid_days, hsm_out = get_work_hours(input_data_raw, disagg_output_object, cons_level, smb_type,
                                                         static_params, survey_work_hours, logger_work_hour_pass,
                                                         hsm_in, hsm_fail)

        if not np.sum(external_lighting) > 1:

            post_process, input_data_raw = post_process_for_external_light(input_data_raw, open_close,
                                                                           disagg_output_object, sampling,
                                                                           disagg_input_object, static_params,
                                                                           logger_work_hour_pass)
            if post_process:
                open_close, valid_days, hsm_out = get_work_hours(input_data_raw, disagg_output_object, cons_level,
                                                                 smb_type, static_params, survey_work_hours,
                                                                 logger_work_hour_pass, hsm_in, hsm_fail)

        user_type = get_user_type(open_close)

        # Marking days with zero consumption as 0 work hour days
        open_close[~ valid_days] = 0
        logger_work_hour.info(' Work hour detection done | User detected as a {} user'.format(user_type))

        # Check if valid HSM is present
        if np.nansum(valid_days) > 120 and not np.isnan(hsm_out.get('last_timestamp')):
            update_hsm(disagg_output_object, hsm_out)
            logger_work_hour.info(' Work hour detection done | Added work hour HSM information to AO HSM')

        else:
            logger_work_hour.info(' Work hour detection done | Not enough valid days, work hour HSM not added to AO HSM')

    elif global_config.get("disagg_mode") == "mtd" and not hsm_fail:
        open_close = np.tile(hsm_in.get('attributes').get('user_work_hour_arr'), (open_close.shape[0], 1))
        user_type = get_user_type(open_close)
        logger_work_hour.info(' Work hour detection done | User detected as a {} user'.format(user_type))

    elif global_config.get("disagg_mode") == "mtd":
        logger_work_hour.warning('Work Hours did not run since MTD mode requires HSM and HSM is missing |')

    else:
        logger_work_hour.error('Unrecognized disagg mode {} |'.format(global_config.get('disagg_mode')))

    disagg_input_object['switch']['smb']['open_close_table'] = open_close

    disagg_output_object = write_work_hour_day_info(input_df, open_close, time_array, disagg_output_object)

    t_work_hour_end = datetime.now()

    work_hour_metrics = {
        'time': get_time_diff(t_work_hour_start, t_work_hour_end),
        'confidence': 1.0,
        'exit_status': exit_status,
    }

    disagg_output_object['disagg_metrics']['work_hour'] = work_hour_metrics

    logger_work_hour.info(' Work hour module took | {} s'.format(work_hour_metrics.get('time')))

    return disagg_input_object, disagg_output_object


def get_user_type(open_close):
    """
    Function to identify type of user
    Parameters:
        open_close  (np.array): 2D array with

    Returns:
        user_type   (str)     : String identifying the user type
    """
    if np.nansum(open_close) == 0:
        user_type = '0x7'

    elif np.nansum(open_close) == open_close.size:
        user_type = '24x7'

    else:
        user_type = 'defined work hour'

    return user_type
