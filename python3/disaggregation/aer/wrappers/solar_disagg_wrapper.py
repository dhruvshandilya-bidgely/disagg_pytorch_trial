"""
Author - Anand Kumar Singh / Paras Tehria
Date - 19th Feb 2020
Call the Solar disaggregation module and get output
"""

# Import python packages

import copy
import logging
import traceback
import numpy as np
from copy import deepcopy
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.write_estimate import write_estimate
from python3.config.pilot_constants import PilotConstants
from python3.utils.time.get_time_diff import get_time_diff
from python3.disaggregation.aer.solar.solar_disagg import run_solar_detection
from python3.disaggregation.aer.solar.solar_disagg import run_solar_estimation
from python3.disaggregation.aer.solar.functions.get_solar_user_profile import get_solar_user_profile
from python3.disaggregation.aer.solar.functions.initialize_solar_params import init_solar_detection_config
from python3.disaggregation.aer.solar.functions.initialize_solar_params import init_solar_estimation_config
from python3.disaggregation.aer.solar.functions.solar_run_mode_manager import update_hsm_in_solar_config
from python3.disaggregation.aer.solar.functions.detect_solar_presence import get_detection_metrics

from python3.master_pipeline.preprocessing.modify_high_consumption import modify_high_consumption
from python3.disaggregation.pre_disagg_ops.reconstruct_rounded_signal import reconstruct_rounded_signal

def get_run_solar_bool(solar_detection_config, input_data):
    """
    Function to check whether sufficient data is available for running solar detection

    Parameters:
        solar_detection_config               (dict)              : Dictionary containing solar configurations
        input_data              (np.ndarray)        : Input thirteen column matrix

    Returns:
        run_solar_det           (bool)              : boolean indicating whether to run solar detection or not
    """

    num_days = len(np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX]))

    if num_days < solar_detection_config.get('so_min_data_req'):
        run_solar_det = False
    else:
        run_solar_det = True

    return run_solar_det


def get_solar_app_profile(disagg_input_object):
    """
    Utility to extract app profile for solar

    Parameters:
        disagg_input_object        (dict)              : Disagg input object

    Returns:
        solar_present                 (str)               : string showing whether user said yes to solar panel at home
    """

    solar_app_profile = disagg_input_object.get("app_profile").get("solar")

    solar_present = "no_input"
    if solar_app_profile is not None:
        solar_number = solar_app_profile.get("number")
        if solar_number is not None:
            if solar_number == 0:
                solar_present = "no"
            else:
                solar_present = "yes"

    return solar_present


def solar_disagg_wrapper(disagg_input_object, disagg_output_object):
    """
    Parameters:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs

    Returns:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    logger_solar_base = disagg_input_object.get("logger").getChild("solar_disagg_wrapper")
    logger_solar_pass = {"logger": logger_solar_base,
                         "logging_dict": disagg_input_object.get("logging_dict")}

    # Calling solar detection wrapper
    disagg_input_object, disagg_output_object = solar_detection_wrapper(disagg_input_object, disagg_output_object)

    # Calling solar estimation wrapper
    disagg_input_object, disagg_output_object = solar_estimation_wrapper(disagg_input_object, disagg_output_object)

    disagg_mode = disagg_input_object.get('config').get('disagg_mode')

    if not (disagg_mode == 'mtd'):

        disagg_output_object = \
            get_solar_user_profile(disagg_input_object, disagg_output_object, logger_solar_pass)

    return disagg_input_object, disagg_output_object


def solar_estimation_wrapper(disagg_input_object, disagg_output_object):
    """
    This function estimates solar generation for a solar user
    Parameters:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs

    Returns:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # Initiate logger for the Solar disaggregation module module
    logger_solar_base = disagg_input_object.get("logger").getChild("solar_disagg_wrapper")
    logger_solar_pass = {"logger": logger_solar_base,
                         "base_logger": logger_solar_base,
                         "logging_dict": disagg_input_object.get("logging_dict")}
    logger_solar = logging.LoggerAdapter(logger_solar_base, disagg_input_object.get("logging_dict"))

    # Starting the algorithm time counter
    time_solar_start = datetime.now()

    # List to store the error points throughout the algo run
    error_list = []

    # Reading global configuration from disagg_input_object
    global_config = disagg_input_object.get("config")

    # noinspection PyBroadException
    try:
        # Load the configuration required for Solar algo
        solar_estimation_config = init_solar_estimation_config(global_config)
        solar_estimation_config['timezone'] = disagg_input_object.get('home_meta_data').get('timezone')

        solar_estimation_config['longitude'] = disagg_input_object.get('home_meta_data').get('longitude')

        solar_estimation_config['latitude'] = disagg_input_object.get('home_meta_data').get('latitude')

        solar_estimation_config['disagg_mode'] = disagg_input_object.get('config').get('disagg_mode')

        solar_estimation_config['estimation_model'] = \
            disagg_input_object.get('loaded_files', {}).get('solar_files', {}).get('estimation_model')

        solar_estimation_config['out_bill_cycles'] = disagg_input_object.get('out_bill_cycles')

        logger_solar.info("Solar disaggregation parameters initialized successfully |")

        is_missing_information = False
        missing_value = ''

        if not solar_estimation_config.get('latitude'):
            is_missing_information = True
            missing_value = missing_value + ' ' + 'Latitude'

        if not solar_estimation_config.get('longitude'):
            is_missing_information = True
            missing_value = missing_value + ' ' + 'Longitude'

        if not solar_estimation_config.get('timezone'):
            is_missing_information = True
            missing_value = missing_value + ' ' + 'Timezone'

        if is_missing_information:
            logger_solar.info('{} missing, skipping solar estimation |'.format(missing_value))
            return disagg_input_object, disagg_output_object

        logger_solar.info("Location features latitude = {:.2f}, longitude = {:.2f}, timezone = {} |".format(
            solar_estimation_config.get('latitude'),
            solar_estimation_config.get('longitude'),
            solar_estimation_config.get('timezone')))

    except KeyError:
        # If loading parameters failed, don't run the algorithm
        error_str = (traceback.format_exc()).replace('\n', ' ')
        error_list.append("Solar disaggregation parameter initialization failed")
        logger_solar.error("Solar disaggregation parameter initialization failed | %s", error_str)

        return disagg_input_object, disagg_output_object

    # Start detection hsm
    solar_detection_time = datetime.now()
    input_data = copy.deepcopy(disagg_input_object.get('input_data_with_neg_and_nan'))

    detection_hsm = disagg_output_object.get("created_hsm", {}).get("solar", {})

    logger_solar.info('Solar detection took | %.3f s ', get_time_diff(solar_detection_time, datetime.now()))

    if detection_hsm is not None:
        # Reading solar flag from detection hsm
        solar_detection_flag = detection_hsm.get('attributes', {}).get('solar_present')
    else:
        solar_detection_flag = None

    # Updating solar config based on run mode
    solar_estimation_config = update_hsm_in_solar_config(disagg_input_object, solar_estimation_config, logger_solar)

    # Estimate solar generation
    if solar_estimation_config.get('estimation_model') is None:
        logger_solar.warning("Solar estimation models not found, skipping estimation module | ")

    elif not solar_detection_flag:
        logger_solar.info("Solar not detected for the user, skipping estimation module | ")

    elif global_config.get('pilot_id') not in PilotConstants.SOLAR_ESTIMATION_ENABLED_PILOTS:
        logger_solar.warning("Solar estimation will not run for this pilot: | {}".format(global_config.get('pilot_id')))

    elif solar_detection_flag is None:
        logger_solar.info("Solar detection did not run, skipping estimation module | ")

    else:
        irradiance = disagg_output_object.get('special_outputs', {}).get('solar', {}).get('attributes', {}).get('irradiance', {})
        input_data, monthly_output, estimation_hsm =\
            run_solar_estimation(input_data, solar_estimation_config, irradiance, logger_solar_pass, solar_detection_flag, detection_hsm)

        # Combining detection and estimation HSMs

        estimation_hsm['detection_confidence'] = detection_hsm.get('attributes').get('confidence')
        estimation_hsm['solar_present'] = detection_hsm.get('attributes').get('solar_present')
        estimation_hsm['instance_probabilities'] = detection_hsm.get('attributes').get('instance_probabilities')
        estimation_hsm['chunk_start'] = detection_hsm.get('attributes').get('chunk_start')
        estimation_hsm['chunk_end'] = detection_hsm.get('attributes').get('chunk_end')
        estimation_hsm['kind'] = detection_hsm.get('attributes').get('kind')

        solar_hsm = {'timestamp': detection_hsm.get('timestamp'),
                     'attributes': estimation_hsm}
        disagg_output_object['created_hsm']['solar'] = solar_hsm

        # Writing results to disagg output object
        read_col_idx = 1
        solar_out_idx = disagg_output_object.get("output_write_idx_map").get("solar")
        disagg_output_object =\
            write_estimate(disagg_output_object, monthly_output, read_col_idx, solar_out_idx, 'bill_cycle')

        solar_generation_column = solar_estimation_config.get('solar_generation_column')

        if input_data.shape[1] >= solar_generation_column:
            epoch_output_copy = copy.deepcopy(input_data[:, [Cgbdisagg.INPUT_EPOCH_IDX, solar_generation_column]])
        else:
            epoch_output_copy = np.c_[input_data[:, Cgbdisagg.INPUT_EPOCH_IDX], np.zeros(len(input_data),)]

        disagg_output_object =\
            write_estimate(disagg_output_object, epoch_output_copy, read_col_idx, solar_out_idx, 'epoch')

        # Post processing to maintain sanity of the results
        input_data_raw = copy.deepcopy(disagg_input_object.get('input_data_with_neg_and_nan'))

        input_data_raw[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] =\
            input_data_raw[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] + epoch_output_copy[:, 1]

        neg_idx = input_data_raw[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] < 0
        nan_idx = np.isnan(input_data_raw[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

        # Setting NaN and negatives as zero

        input_data_raw[nan_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0
        input_data_raw[neg_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

        sampling_rate = disagg_input_object.get("config").get("sampling_rate")
        disagg_input_object['input_data_without_outlier_removal'] = copy.deepcopy(input_data_raw)

        input_data_raw = modify_high_consumption(input_data_raw, sampling_rate, logger_solar_pass)

        # Updating input data after solar adjustment and storing previous data
        disagg_input_object['input_data_without_solar'] = copy.deepcopy(disagg_input_object.get('input_data'))
        disagg_input_object['input_data'][:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = input_data_raw[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

        # adding this to update the raw data used for residual calculation
        disagg_input_object['input_data_without_rounding_correction'] = copy.deepcopy(disagg_input_object.get('input_data'))
        disagg_input_object['input_data_without_outlier_removal'] = copy.deepcopy(disagg_input_object.get('input_data'))

        disagg_input_object = reconstruct_rounded_signal(disagg_input_object, logger_solar)

    time_solar_end = datetime.now()
    solar_time = get_time_diff(time_solar_start, time_solar_end)
    logger_solar.info("Solar disaggregation Estimation took | %0.3f s", solar_time)

    # Write time taken etc.

    disagg_metrics_detection = disagg_output_object.get('disagg_metrics', {}).get('solar', {})

    disagg_metrics_dict = {
        'time': disagg_metrics_detection.get('time', 0) + get_time_diff(time_solar_start, datetime.now()),
        'exit_status': disagg_metrics_detection.get('exit_status', {})
    }

    # Write code runtime, confidence level to the disagg_output_object

    disagg_output_object["disagg_metrics"]["solar"] = disagg_metrics_dict

    logger_solar.info('Timing: Total time taken to run solar module is %0.3f',
                      get_time_diff(time_solar_start, datetime.now()))

    disagg_output_object['disagg_metrics']['solar'] = disagg_metrics_dict

    return disagg_input_object, disagg_output_object


def solar_detection_wrapper(disagg_input_object, disagg_output_object):
    """
    Parameters:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs

    Returns:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # Initiate logger for the solar module

    logger_so_base = disagg_input_object.get('logger').getChild('solar_detection_wrapper')
    logger_so = logging.LoggerAdapter(logger_so_base, disagg_input_object.get('logging_dict'))

    logger_pass = {
        'logger': logger_so_base,
        'logging_dict': disagg_input_object.get('logging_dict'),
    }

    t_so_start = datetime.now()

    # Fetching information from the config file

    global_config = copy.deepcopy(disagg_input_object.get('config'))
    input_data = copy.deepcopy(disagg_input_object.get('input_data_with_neg_and_nan'))

    # Fetching solar hsm

    try:
        hsm_dic = disagg_input_object.get('appliances_hsm')
        hsm_in = hsm_dic.get('solar')
    except KeyError:
        hsm_in = None
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger_so.error("HSM fetching failed in solar detection | {}".format(error_str))

    # hsm fail when hsm is not available and disagg mode is incremental or mtd
    hsm_not_available = hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0

    hsm_fail = hsm_not_available and (global_config.get("disagg_mode") == "mtd")

    solar_app_profile = get_solar_app_profile(disagg_input_object)
    solar_detection_config = init_solar_detection_config(global_config, disagg_input_object)

    #Initialise latitude/longitude
    solar_detection_config['latitude'] = disagg_input_object.get('home_meta_data').get('latitude')
    solar_detection_config['longitude'] = disagg_input_object.get('home_meta_data').get('longitude')

    #Load Detection Models
    detection_models = disagg_input_object.get('loaded_files', {}).get('solar_files', {}).get('detection_model')

    # get_run_solar_bool checks whether sufficient data is present for solar detection or not

    run_solar = get_run_solar_bool(solar_detection_config, input_data)

    # initialising exit_code and error_list

    exit_code = 1
    error_list = []

    write_hsm = True

    max_instances = solar_detection_config.get('max_instances')

    hsm = {}
    if hsm_fail:
        logger_so.warning("Solar detection did not run because {} mode needs HSM and it is missing | ".format(
            global_config.get("disagg_mode")))
        exit_code = 0
        error_list.append("no_hsm_available")

    elif detection_models is None:
        logger_so.warning("Solar detection model files are missing, skipping detection | ")
        exit_code = 0
        error_list.append("detection_model_not_found")

    elif global_config.get("disagg_mode") == "mtd":
        hsm = hsm_in
        sun_presence = hsm.get('attributes', {}).get('solar_present', {})[0]

        # Solar detection metrics
        irradiance, start_date, end_date, kind = get_detection_metrics(solar_detection_config, input_data, sun_presence, logger_pass)

        # Updating hsm attributes
        hsm['attributes']['irradiance'] = irradiance
        hsm['attributes']['start_date'] = start_date
        hsm['attributes']['end_date'] = end_date
        hsm['attributes']['kind'] = kind
        hsm['attributes']['solar_present'] = sun_presence

        logger_so.info('solar detection is blocked in mtd mode, writing last incremental/historical run\'s output | ')

    elif not run_solar:
        logger_so.warning("Solar detection did not run because less than required number of data available | ")
        write_hsm = False
        hsm = {}
        exit_code = 0
        error_list.append("less_num_days")

    elif (not solar_detection_config['latitude']) or not (solar_detection_config['longitude']):
        logger_so.warning("Solar detection did not run because latitude/longitude not available | ")
        write_hsm = False
        hsm = {}
        exit_code = 0
        error_list.append("latitude/longitude missing")

    # Setting detection to 0 if user has said no in app profile
    elif solar_app_profile == "no":
        logger_so.warning("User said no to solar panel presence in the app profile | ")

        hsm = {
            'timestamp': int(input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX]),
            'attributes': {
                'confidence': 0.00,
                'solar_present': 0,
                'instance_probabilities': [0] * max_instances,
                'start_date': None,
                'end_date': None,
                'kind': None
            }
        }
        logger_so.info(
            'Solar instance probabilities | {}'.format(hsm.get('attributes').get('instance_probabilities')))
        logger_so.info(
            'Solar Presence:, Confidence: | {}, {}'.format(hsm.get('attributes').get('solr_present'),
                                                           hsm.get('attributes').get('confidence')))

    # Setting detection to 1 if user has said yes in app profile
    elif solar_app_profile == "yes":
        logger_so.warning("User said yes to solar panel presence in the app profile | ")

        solar_present = 1
        irradiance, start_date, end_date, kind = get_detection_metrics(solar_detection_config, input_data, solar_present, logger_pass)

        hsm = {
            'timestamp': int(input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX]),
            'attributes': {
                'confidence': 1.00,
                'solar_present': solar_present,
                'instance_probabilities': [1.0] * max_instances,
                'irradiance': irradiance,
                'start_date': start_date,
                'end_date': end_date,
                'kind': kind
            }
        }
        logger_so.info(
            'Solar instance probabilities | {}'.format(hsm.get('attributes').get('instance_probabilities')))
        logger_so.info(
            'Solar Presence:, Confidence: | {}, {}'.format(hsm.get('attributes').get('solar_present'),
                                                           hsm.get('attributes').get('confidence')))

    # if all checks pass and disagg_mode is historical
    elif global_config.get("disagg_mode") == "historical" or hsm_not_available:
        # Calling the main solar disagg algo
        global_config['disagg_mode'] = "historical"
        hsm, write_hsm = run_solar_detection(global_config, input_data, disagg_input_object, solar_detection_config, logger_pass)
        logger_so.info(
            'Solar instance probabilities | {}'.format(hsm.get('attributes').get('instance_probabilities')))

        logger_so.info(
            'Solar Presence:, Confidence: | {}, {}'.format(hsm.get('attributes').get('solar_present'),
                                                           hsm.get('attributes').get('confidence')))

    # if all checks pass and disagg_mode is incremental
    elif global_config.get("disagg_mode") == "incremental":
        # Calling the main solar disagg algo for incremental run
        hsm, write_hsm = run_solar_detection(global_config, input_data, disagg_input_object, solar_detection_config, logger_pass)

        # new probability is the output from recent run, old probability is from hsm

        final_confidence = hsm.get('attributes').get('confidence')
        old_probabilities = hsm_in.get('attributes', {}).get('instance_probabilities', [])
        # combining the results from old hsm and new run

        # removing -1 from the list as it means detection module failure and adding current run probability to the list
        final_probabilities = [x for x in old_probabilities if x != -1]

        # limiting the size of detection instances
        final_probabilities = final_probabilities[-max_instances:]

        solar_present = int(final_confidence > solar_detection_config.get('solar_disagg').get('lgbm_threshold'))

        logger_so.info('Solar instance probabilities | {}'.format(final_probabilities))

        logger_so.info(
            'Solar Presence:, Confidence: | {}, {}'.format(solar_present, final_confidence))

        # Solar detection metrics
        irradiance, start_date, end_date, kind = get_detection_metrics(solar_detection_config, input_data, solar_present, logger_pass)

        # Updating hsm attributes
        hsm['attributes'] = {
            'confidence': final_confidence,
            'solar_present': solar_present,
            'instance_probabilities': final_probabilities,
            'irradiance': irradiance,
            'start_date': start_date,
            'end_date': end_date,
            'kind': kind
        }

    if write_hsm:
        # Saving the new HSM to the disagg_output_object
        disagg_output_object["created_hsm"]["solar"] = hsm
        disagg_output_object['special_outputs']['solar'] = hsm

    # Final exit status dict

    exit_status = {
        'exit_code': exit_code,
        'error_list': error_list,
    }

    t_so_end = datetime.now()

    logger_so.info('Created solar HSM is | %s',
                   str(disagg_output_object.get("created_hsm").get("solar") ).replace('\n', ' '))
    logger_so.info('Solar detection took | %.3f s ', get_time_diff(t_so_start, t_so_end))

    # Write exit status time taken etc.

    disagg_metrics_dict = {
        'time': get_time_diff(t_so_start, t_so_end),
        'exit_status': exit_status,
    }

    disagg_output_object['disagg_metrics']['solar'] = disagg_metrics_dict

    return disagg_input_object, disagg_output_object
