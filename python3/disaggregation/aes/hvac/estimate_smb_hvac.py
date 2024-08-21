"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to estimate smb HVAC consumption
"""

# Import python packages
import logging
import numpy as np

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aes.hvac.filter_smb_consumption import filter_by_mode
from python3.disaggregation.aes.hvac.populate_hvac import populate_cooling_estimates
from python3.disaggregation.aes.hvac.populate_hvac import populate_heating_estimates
from python3.disaggregation.aes.hvac.plot_utils_smb_hvac import plot_regression_clusters_smb
from python3.disaggregation.aes.hvac.init_hourly_smb_hvac_params import hvac_static_params
from python3.disaggregation.aes.hvac.estimate_smb_setpoint import setpoint_by_mode_at_x_hour
from python3.disaggregation.aes.hvac.postprocess_hvac_smb import get_x_hour_and_epoch_hvac


def get_min_threshold(detection_debug, logger_hvac, hvac_params):
    """
    Function to return minimum hvac threshold at epoch level

    Parameters:
        detection_debug (dict)              : Dictionary containing hvac detection related key attributes
        logger_hvac     (logging object)    : To keep log of algo progress
        hvac_params     (dict)              : Dictionary containing hvac related key parameters

    Returns:
        min_threshold   (float)             : The minimum threshold for consumption
    """

    # Finding minimum threshold based on hvac amplitude
    if np.isnan(detection_debug.get('mu')) or np.isnan(detection_debug.get('sigma')):

        logger_hvac.info('appliance mu or sigma is null. taking minimum amplitude from params |')
        # appliance mu or sigma is null. taking minimum amplitude from params
        min_threshold = hvac_params['MIN_AMPLITUDE']
        logger_hvac.debug(' minimum amplitude is {}/epoch for filtering |'.format(min_threshold))

    else:

        # Finding minimum threshold based on hvac amplitude within failsafe
        logger_hvac.info('evaluating minimum amplitude for filtering |')
        # evaluating minimum amplitude for filtering
        min_threshold = detection_debug['amplitude_cluster_info']['cluster_limits'][0][0]
        logger_hvac.debug('minimum amplitude is {}/epoch for filtering |'.format(np.around(min_threshold, 2)))

    return min_threshold


def add_setpoint_analytics(disagg_output_object, estimation_debug):
    """
    Function to populate attributes for second order analytics, from setpoint perspective only

    Parameters:

        disagg_output_object    (dict)                  : Dictionary containing all outputs
        estimation_debug        (dict)                  : Dictionary containing hvac estimation related key information

    Returns:
        None
    """

    # Finding analytics related setpoint attributes for hvac
    disagg_output_object['analytics']['values']['cooling']['setpoint'] = estimation_debug['cdd']
    disagg_output_object['analytics']['values']['heating']['setpoint'] = estimation_debug['hdd']


def add_estimation_analytics(disagg_output_object, x_hour_hvac_mode):
    """
    Function to populate attributes for second order analytics, from estimation perspective

    Parameters:

        disagg_output_object    (dict)                  : Dictionary containing all outputs
        x_hour_hvac_mode        (dict)                  : Dictionary containing hvac mode related key information

    Returns:
        None
    """
    # Finding analytics related estimation attributes for hvac
    disagg_output_object['analytics']['values']['cooling']['estimation'] = x_hour_hvac_mode['cooling']
    disagg_output_object['analytics']['values']['heating']['estimation'] = x_hour_hvac_mode['heating']


def estimate_smb_hvac(hvac_input_data, invalid_idx, hvac_debug, hvac_params, global_config,
                      logger_base, hvac_exit_status, disagg_output_object):
    """
    Function to estimate hvac appliance epoch level consumption after filtering out non-hvac data

    Parameters:

        hvac_input_data     (np.ndarray)            : 2D Array of epoch level input data frame flowing into hvac module
        invalid_idx         (np.ndarray)            : Array of invalid epochs based on consumption and temperature
        hvac_debug          (dict)                  : Dictionary containing hvac detection stage attributes
        hvac_params         (dict)                  : Dictionary containing hvac algo related initialized parameters
        global_config       (dict)                  : Dictionary containing user profile related information
        logger_base         (logging object)        : Writes logs during code flow
        hvac_exit_status    (dict)                  : Dictionary containing hvac exit code and list of handled errors
        disagg_output_object(dict)                  : Dictionary containing all outputs

    Returns:

        daily_output       (np.ndarray)           : 2D array of day epoch and daily cooling-heating estimates
        hourly_output      (np.ndarray)           : 2D array of epoch time-stamp and epoch cooling-heating estimates
        estimation_debug   (dict)                 : Dictionary containing estimation stage related debugging information
        regression_debug   (dict)                 : Dictionary containing degree-day vs daily-energy regression params
        hvac_exit_status   (dict)                 : Dictionary containing hvac exit code and list of handled errors
    """

    static_params = hvac_static_params()

    # initializing logger object
    logger_local = logger_base.get("logger").getChild("estimate_hvac")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # getting detection debug for key attributes related to hvac detection
    detection_debug = hvac_debug.get('detection')

    # getting unique days and unique day indexes at epoch level. Followed by unique month indexes
    _, day_idx = np.unique(hvac_input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_index=False, return_inverse=True, return_counts=False)
    _, month_idx = np.unique(hvac_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_inverse=True)

    # Initializing estimation debug dictionary
    estimation_debug = {}

    # estimating hvac setpoints for not mtd disagg modes
    if (global_config.get('disagg_mode') == 'historical') or (global_config.get('disagg_mode') == 'incremental'):

        # creating ket attributes dictionary objects for hvac module
        common_objects = {'hvac_input_data': hvac_input_data,
                          'invalid_idx': invalid_idx,
                          'global_config': global_config,
                          'hvac_exit_status': hvac_exit_status,
                          'disagg_output_object': disagg_output_object}

        logger_hvac.info(' ------- 1 : Detecting setpoint for AC ------- |')

        # estimating user ac-setpoint for modes at x-hour level aggregate (config based)
        estimation_debug['cdd'], hvac_exit_status = setpoint_by_mode_at_x_hour(detection_debug['cdd'], day_idx,
                                                                               hvac_params['setpoint']['AC'], 0,
                                                                               logger_pass, common_objects)

        # estimating user sh-setpoint for modes at x-hour level aggregate (config based)
        logger_hvac.info(' ------- 1 : Detecting setpoint for SH  ------- |')
        estimation_debug['hdd'], hvac_exit_status = setpoint_by_mode_at_x_hour(detection_debug['hdd'], day_idx,
                                                                               hvac_params['setpoint']['SH'], 0,
                                                                               logger_pass, common_objects)

        logger_hvac.info(' >> Success in detecting setpoints, Cooling : {}, Heating : {} |'.format(
            estimation_debug['cdd']['setpoint'], estimation_debug['hdd']['setpoint']))

        # adding attributes found to analytics dictionary
        if disagg_output_object['analytics']['required']:
            add_setpoint_analytics(disagg_output_object, estimation_debug)

    elif global_config.get('disagg_mode') == 'mtd':

        # reading detection and estimation related attributes for mtd run
        detection_debug = hvac_debug.get('detection')
        estimation_debug = hvac_debug.get('estimation')

        logger_hvac.info(' Successfully read hvac estimation hsm |')

    logger_hvac.info(' ------- 2 : Filtering epoch level consumption for AC ------- |')

    # filtering epoch level ac-consumption based on amplitudes
    ac_filter_info = filter_by_mode(hvac_input_data, invalid_idx, detection_debug.get('cdd'), logger_pass,
                                    estimation_debug.get('cdd'),
                                    day_idx, month_idx, hvac_params.get('estimation').get('AC'))

    logger_hvac.info(' ------- 2 : Filtering epoch level consumption for SH ------- |')

    # filtering epoch level sh-consumption based on amplitudes
    sh_filter_info = filter_by_mode(hvac_input_data, invalid_idx, detection_debug.get('hdd'), logger_pass,
                                    estimation_debug.get('hdd'),
                                    day_idx, month_idx, hvac_params.get('estimation').get('SH'))

    # assigning important attributes to estimation_debug dictionary
    estimation_debug['cdd']['cdd_daily'] = ac_filter_info.get('degree_day')
    estimation_debug['hdd']['hdd_daily'] = sh_filter_info.get('degree_day')
    estimation_debug['cdd']['hoursSelected'] = ac_filter_info.get('hours_selected')
    estimation_debug['hdd']['hoursSelected'] = sh_filter_info.get('hours_selected')
    estimation_debug['cdd']['netBL'] = ac_filter_info.get('epoch_filtered_data')
    estimation_debug['hdd']['netBL'] = sh_filter_info.get('epoch_filtered_data')

    # accessing user level utility attributes from config
    epoch_array = np.arange(len(hvac_input_data))
    epochs_per_hour = Cgbdisagg.SEC_IN_HOUR / global_config['sampling_rate']
    hours_aggregation = disagg_output_object.get('switch').get('hvac').get('hour_aggregate_level')
    aggregation_factor = hours_aggregation * epochs_per_hour
    aggregate_identifier = ((epoch_array + static_params.get('inequality_handler')) // aggregation_factor).astype(int)
    logger_hvac.info(' Running regression at {} hours aggregation |'.format(hours_aggregation))

    # initializing dictionary to capture mode level key attributes
    x_hour_hvac_by_mode = {'cooling': {}, 'heating': {}}

    # initializing dictionary to capture cluster level key attributes
    estimation_debug['cdd']['cluster_info'] = {}
    estimation_debug['hdd']['cluster_info'] = {}
    estimation_debug['cdd']['cluster_info']['hvac'] = {}
    estimation_debug['hdd']['cluster_info']['hvac'] = {}
    estimation_debug['cdd']['cluster_info']['hvac'][0] = {'regression_kind': 'linear', 'validity': False}
    estimation_debug['hdd']['cluster_info']['hvac'][0] = {'regression_kind': 'linear', 'validity': False}

    logger_hvac.info(' ------- 3 : Regression at x - hours for AC ------- |')

    # populating cooling estimates based on regression
    populate_cooling_estimates(ac_filter_info, aggregate_identifier, x_hour_hvac_by_mode, global_config, logger_pass,
                               hvac_exit_status, estimation_debug)

    logger_hvac.info(' ------- 3 : Regression at x - hours for SH ------- |')

    # populating heating estimates based on regression
    populate_heating_estimates(sh_filter_info, aggregate_identifier, x_hour_hvac_by_mode, global_config, logger_pass,
                               hvac_exit_status, estimation_debug)

    # adding estimation related key attributes to analytics dictionary
    if disagg_output_object.get('analytics').get('required') and global_config.get('disagg_mode', '') != 'mtd':
        add_estimation_analytics(disagg_output_object, x_hour_hvac_by_mode)

    # generating regression plots for debugging
    general_plot_condition = (('hvac' in global_config.get('generate_plots')) or ('all' in global_config.get('generate_plots')))
    generate_regression_plot = general_plot_condition and global_config.get('disagg_mode', '') != 'mtd'

    if generate_regression_plot and (disagg_output_object.get('switch').get('plot_level') >= 2):
        plot_regression_clusters_smb(hvac_input_data, x_hour_hvac_by_mode, estimation_debug, global_config)

    logger_hvac.info(' ------- 4 : Spreading HVAC at epoch level ------- |')

    # spreading x hour estimates to hour level
    x_hour_hvac_by_mode, epoch_hvac = get_x_hour_and_epoch_hvac(x_hour_hvac_by_mode, ac_filter_info, sh_filter_info,
                                                                hvac_input_data,
                                                                aggregate_identifier, hvac_params, logger_pass)

    # ensuring hvac estimate doesn't exceed total, at any epoch --->
    suppress_ac_epochs = epoch_hvac[:, 0] >= hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    suppress_sh_epochs = epoch_hvac[:, 1] >= hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    epoch_hvac[suppress_ac_epochs, 0] = hvac_input_data[suppress_ac_epochs, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    epoch_hvac[suppress_sh_epochs, 1] = hvac_input_data[suppress_sh_epochs, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    return x_hour_hvac_by_mode, epoch_hvac, estimation_debug, hvac_exit_status
