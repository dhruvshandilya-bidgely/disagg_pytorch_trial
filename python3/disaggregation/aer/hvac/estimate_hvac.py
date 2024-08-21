"""
Author - Abhinav Srivastava / Mirambika Sikdar
Date - 06/12/2023
Call for Estimating HVAC
"""

# Import python packages
import logging
import numpy as np

# Import functions from within the project
import pandas as pd

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.dbscan import get_hvac_clusters
from python3.disaggregation.aer.hvac.dbscan import get_hvac_clusters_mtd
from python3.disaggregation.aer.hvac.filter_consumption import filter_by_mode
from python3.disaggregation.aer.hvac.plot_regression import plot_regression_clusters
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params
from python3.disaggregation.aer.hvac.estimate_setpoint import setpoint_by_mode_at_x_hour
from python3.disaggregation.aer.hvac.post_process_consumption import get_x_hour_and_epoch_hvac
from python3.disaggregation.aer.hvac.estimate_from_regression_df import estimate_from_regression_df


def get_min_threshold(detection_debug, logger_hvac, hvac_params):
    """
    Function to return minimum hvac threshold at epoch level

    Parameters:
        detection_debug     (dict)              : Dictionary containing hvac detection related key attributes
        logger_hvac         (logging object)    : To keep log of algo progress
        hvac_params         (dict)              : Dictionary containing hvac related key parameters

    Returns:
        min_threshold       (float)             : The minimum threshold for consumption
    """

    if np.isnan(detection_debug['mu']) or np.isnan(detection_debug['sigma']):

        logger_hvac.info('appliance mu or sigma is null. taking minimum amplitude from params |')
        # appliance mu or sigma is null. taking minimum amplitude from params
        min_threshold = hvac_params['MIN_AMPLITUDE']
        logger_hvac.debug(' minimum amplitude is {}/epoch for filtering |'.format(min_threshold))

    else:

        logger_hvac.info('evaluating minimum amplitude for filtering |')
        # evaluating minimum amplitude for filtering
        min_threshold = detection_debug['amplitude_cluster_info']['cluster_limits'][0][0]
        logger_hvac.debug('minimum amplitude is {}/epoch for filtering |'.format(np.around(min_threshold, 2)))

    return min_threshold


def select_best_model(fit_model_linear, fit_model_sqrt):
    """
    Function to select better model between linear and root

    Parameters:
        fit_model_linear    (pd.DataFrame)   : Linear model fit on setpoint and consumption
        fit_model_sqrt      (pd.DataFrame)   : root model fit on setpoint and consumption

    Return:
        fit_model           (object)         : Best model selected between linear and root
    """
    # Check which model has higher r square
    if fit_model_linear['Rsquared']['Ordinary'] > fit_model_sqrt['Rsquared']['Ordinary']:
        fit_model = fit_model_linear
    else:
        fit_model = fit_model_sqrt

    return fit_model


def mtd_estimate(cluster_info, filtered_data_df, filter_day, identifier, logger_hvac, hvac_estimate_x_hour):
    """
    Function to estimate hvac appliance in mtd mode

    Parameters:

        cluster_info            (dict)               : Dictionary containing validity info of clusters and regression type
        filtered_data_df        (pd.Dataframe)       : Dataframe containing cluster id
        filter day              (np.ndarray)         : Array containing boolean of valid consumption and cdd/hdd
        identifier              (dict)               : Dictionary containing attributes of ac/sh
        logger_hvac             (logging object)     : Writes logs during code flow
        hvac_estimate_x_hour    (np.ndarray)         : Array containing hvac estimates at x-hour level

    Returns:
        hvac_estimate_x_hour    (np.ndarray)         : Array containing hvac estimates at x-hour level
    """

    # For each cluster
    for cluster in cluster_info.keys():

        # If the cluster is valid, get the estimated from stored regression coefficient
        regression_df = \
            filtered_data_df[
                (filtered_data_df[filter_day] == 1) & (filtered_data_df[identifier['cluster_id']] == cluster)][
                list(identifier.values())]

        if cluster_info[cluster]['validity']:

            logger_hvac.info(' >> Estimating from hsm coefficients |')
            hvac_coefficient = cluster_info[cluster]['coefficient']

            if cluster_info[cluster]['regression_kind'] == 'linear':

                hvac_estimate = list(np.array(regression_df[identifier['degree_day']]) * hvac_coefficient[0])
                hvac_estimate = [0 if cluster_flag != cluster else hvac_estimate.pop(0)
                                 for cluster_flag in filtered_data_df[identifier['cluster_id']]]
                hvac_estimate_x_hour = np.sum([hvac_estimate_x_hour, hvac_estimate], axis=0)

            elif cluster_info[cluster]['regression_kind'] == 'root':

                hvac_estimate = list(np.sqrt(np.array(regression_df[identifier['degree_day']])) * hvac_coefficient[0])
                hvac_estimate = [0 if cluster_flag != cluster else hvac_estimate.pop(0)
                                 for cluster_flag in filtered_data_df[identifier['cluster_id']]]
                hvac_estimate_x_hour = np.sum([hvac_estimate_x_hour, hvac_estimate], axis=0)

    return hvac_estimate_x_hour


def estimate_cluster_energy_at_hour(global_config, logger_base, hvac_exit_status, filtered_data_df, cluster_info, mode):
    """
    Function to estimate hvac appliance epoch level consumption

    Parameters:

        global_config           (dict)               : Dictionary containing user profile related information
        logger_base             (logging object)     : Writes logs during code flow
        hvac_exit_status        (dict)               : Dictionary containing hvac exit code and list of handled errors
        filtered_data_df        (pd.DataFrame)       : Dataframe containing cluster id
        cluster_info            (dict)               : Dictionary containing validity info of clusters and regression type
        mode                    (int)                : Mode identifier in loop

    Returns:
        estimation_debug        (dict)                : Dictionary containing estimation stage related key information
    """

    logger_local = logger_base.get("logger").getChild("estimate_consumption")
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    cluster_info_master = cluster_info

    appliance = 'hvac'
    filter_day = 'filter_day'
    cluster_info = cluster_info['hvac']
    identifier = {'degree_day': 'degree_day',
                  'day_consumption': 'filter_cons',
                  'day_validity': 'filter_day',
                  'cluster_id': 'day_hvac_cluster'}

    # Estimation of coefficients only happens in historical or incremental mode
    if (global_config.get('disagg_mode') == 'historical') or (global_config.get('disagg_mode') == 'incremental'):

        master_regression_df = filtered_data_df[(filtered_data_df[filter_day] == 1)][list(identifier.values())]
        hvac_estimate_x_hour = list(np.zeros(len(filtered_data_df)))
        points_for_regression = np.array(filtered_data_df[filter_day])

        # If non-zero points present for regression for all clusters, estimate coefficients
        if any(points_for_regression):

            regression_carrier_dict = {}
            regression_carrier_dict['cluster_info'] = cluster_info
            regression_carrier_dict['filtered_data_df'] = filtered_data_df
            regression_carrier_dict['filter_day'] = filter_day
            regression_carrier_dict['identifier'] = identifier
            regression_carrier_dict['cluster_info_master'] = cluster_info_master
            regression_carrier_dict['appliance'] = appliance
            regression_carrier_dict['hvac_estimate_day'] = hvac_estimate_x_hour

            hvac_estimate_x_hour = estimate_from_regression_df(regression_carrier_dict, logger_hvac)

        else:

            logger_hvac.info(' No points for regression in mode {} |'.format(mode))
            hvac_estimate_x_hour = np.array(hvac_estimate_x_hour)

    # For MTD mode, user stored hvac coefficients to estimate hvac if non-zero points for regression are present
    elif global_config.get('disagg_mode') == 'mtd':

        master_regression_df = filtered_data_df[(filtered_data_df[filter_day] == 1)][list(identifier.values())]
        hvac_estimate_x_hour = np.array(list(np.zeros(len(filtered_data_df))))
        points_for_regression = np.array(filtered_data_df[filter_day])

        if any(points_for_regression):

            hvac_estimate_x_hour = mtd_estimate(cluster_info, filtered_data_df, filter_day, identifier, logger_hvac,
                                                hvac_estimate_x_hour)

        else:

            logger_hvac.info(' No points for regression in mode {} |'.format(mode))
            hvac_estimate_x_hour = np.array(hvac_estimate_x_hour)

    estimation_debug = {
        'cluster_info': cluster_info,
        'regression_df': master_regression_df,
        'hvac_estimate_day': np.array(hvac_estimate_x_hour),
        'exit_status': hvac_exit_status
    }

    return estimation_debug


def add_setpoint_analytics(disagg_output_object, estimation_debug):
    """
    Function to populate attributes for second order analytics, from setpoint perspective only

    Parameters:

        disagg_output_object    (dict)   : Dictionary containing all outputs
        estimation_debug        (dict)   : Dictionary containing hvac estimation related key information

    Returns:
        None
    """
    # Update values in disagg_output_object
    disagg_output_object['analytics']['values']['cooling']['setpoint'] = estimation_debug['cdd']
    disagg_output_object['analytics']['values']['heating']['setpoint'] = estimation_debug['hdd']


def add_estimation_analytics(disagg_output_object, x_hour_hvac_mode):
    """
    Function to populate attributes for second order analytics, from estimation perspective

    Parameters:

        disagg_output_object    (dict)    : Dictionary containing all outputs
        x_hour_hvac_mode        (dict)    : Dictionary containing hvac mode related key information

    Returns:
        None
    """
    # Update values in disagg_output_object
    disagg_output_object['analytics']['values']['cooling']['estimation'] = x_hour_hvac_mode['cooling']
    disagg_output_object['analytics']['values']['heating']['estimation'] = x_hour_hvac_mode['heating']


def include_relevant_points(filtered_data_df, cluster_info, logger_hvac):
    """
    Function to include left over but relevant points after DBSCAN that goes into regression

    Parameters:

        filtered_data_df    (pd.DataFrame)       : Dataframe containing filtered data with tentative cluster ids
        cluster_info        (dict)               : cluster info with key information updated
        logger_hvac         (logging object)     : Writes logs during code flow

    Returns:

        main_df             (pd.DataFrame)       : Dataframe containing filtered data with updated cluster ids
        info                (dict)               : cluster info with key information updated
    """
    # Function to include left over but relevant points after DBSCAN that goes into regression
    static_params = hvac_static_params()

    main_df = filtered_data_df
    info = cluster_info

    valid_clusters = []
    for cluster in info['hvac'].keys():
        if info['hvac'][cluster]['validity']:
            valid_clusters.append(cluster)

    df_placeholder = main_df.copy()

    if len(valid_clusters) == 1:
        df_valid = main_df[main_df['day_hvac_cluster'] == valid_clusters[0]]
        degree_limit_trend = np.percentile(np.array(df_valid['degree_day']),
                                           static_params['include_extra_points_upper'])
        cons_limit_trend = np.percentile(np.array(df_valid['filter_cons']), static_params['include_extra_points_upper'])

        degree_limit_off = np.percentile(np.array(df_valid['degree_day']), static_params['include_extra_points_lower'])
        cons_limit_off = np.percentile(np.array(df_valid['filter_cons']), static_params['include_extra_points_lower'])

        # adding points along the trend line and adding high consumption points, off-trend line
        df_placeholder['include'] = False
        df_placeholder['include'][(df_placeholder['degree_day'] > degree_limit_trend) & (df_placeholder['filter_cons'] > cons_limit_trend)] = True
        df_placeholder['include'][(df_placeholder['degree_day'] > degree_limit_off) & (df_placeholder['filter_cons'] > cons_limit_off)] = True

        logger_hvac.info(' Total extra valid points included : {} |'.format(np.sum(df_placeholder['include'])))
        df_placeholder['day_hvac_cluster'][df_placeholder['include']] = valid_clusters[0]

    main_df['day_hvac_cluster'] = df_placeholder['day_hvac_cluster']

    return main_df, info


def populate_cooling_estimates(ac_filter_info, estimation_debug, aggregate_identifier, x_hour_ac_mode, global_config,
                               logger_base, hvac_exit_status):
    """
    Function to populate hvac estimates at epoch level

    Parameters:

        ac_filter_info          (dict)             : Dictionary containing key information post filtering stage
        estimation_debug        (dict)             : Dictionary containing all key information related to AC estimation
        aggregate_identifier    (np.ndarray)       : Integer Index array identifying at what hour level aggregation has to be done
        x_hour_ac_mode          (dict)             : Dictionary initialized to carry mode wise AC estimates
        global_config           (dict)             : Dictionary containing user profile related information
        logger_base             (logging object)   : Writes logs during code flow
        hvac_exit_status        (dict)             : Dictionary containing HVAC exit code and list of handled errors

    Returns:
        None
    """

    logger_local = logger_base.get("logger").getChild("populate_cooling_estimates")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    cooling_modes = ac_filter_info['degree_day'].columns.values

    # For each mode in hvac
    for mode in cooling_modes:

        # Calculate aggregated cooling discomfort degree from epoch degree
        ac_epoch_degree = np.array(ac_filter_info['epoch_degree'][mode])
        ac_degree_x_hours = np.bincount(aggregate_identifier, ac_epoch_degree)
        ac_degree_x_hours = ac_degree_x_hours.reshape(len(ac_degree_x_hours), 1)

        # Calculate aggregated cooling consumption from epoch consumption
        epoch_ac_filtered_data = np.array(ac_filter_info['epoch_filtered_data'][mode])
        x_hours_ac_filtered_data = np.bincount(aggregate_identifier, epoch_ac_filtered_data)
        x_hours_ac_filtered_data = x_hours_ac_filtered_data.reshape(len(x_hours_ac_filtered_data), 1)

        # Select valid ac usage hours
        ac_hours_selected = np.logical_and(ac_degree_x_hours > 0, x_hours_ac_filtered_data > 0)

        logger_hvac.info(' Mode : {} , Number of data points selected at aggregation level : {}  |'.format(mode, np.sum(
            ac_hours_selected)))

        # Combine the input arrays
        filtered_data = np.c_[ac_degree_x_hours, x_hours_ac_filtered_data, ac_hours_selected]

        # Default values
        filtered_data_df = pd.DataFrame()
        cluster_info = {}
        # Estimation of coefficients is only valid in historical / incremental mode
        if (global_config.get('disagg_mode') == 'historical') or (global_config.get('disagg_mode') == 'incremental'):
            # Get clusters from dbscan
            filtered_data_df, cluster_info = get_hvac_clusters(filtered_data)

            # Include high consumption epochs in case missed in clustering
            try:
                filtered_data_df, cluster_info = include_relevant_points(filtered_data_df, cluster_info, logger_hvac)
                logger_hvac.info(' >> Included other relevant cooling points for regression |')

            except (ValueError, IndexError, KeyError):
                logger_hvac.info(' Unable to include relevant extra points from other invalid clusters |')

            estimation_debug['cluster_info'] = cluster_info

        elif global_config.get('disagg_mode') == 'mtd':

            cluster_info = estimation_debug['cluster_info']
            filtered_data_df = get_hvac_clusters_mtd(filtered_data, cluster_info)

        # Estimate aggregated hourly ac estimate for each mode
        x_hour_ac_mode[mode] = estimate_cluster_energy_at_hour(global_config, logger_pass, hvac_exit_status,
                                                               filtered_data_df, cluster_info, mode)


def populate_heating_estimates(sh_filter_info, estimation_debug, aggregate_identifier, x_hour_sh_mode, global_config,
                               logger_base, hvac_exit_status):
    """
    Function to populate hvac estimates at epoch level

    Parameters:
        sh_filter_info          (dict)             : Dictionary containing key information post filtering stage
        estimation_debug        (dict)             : Dictionary containing all key information related to SH estimation
        aggregate_identifier    (np.ndarray)       : Integer Index array identifying at what hour level aggregation has to be done
        x_hour_sh_mode          (dict)             : Dictionary initialized to carry mode wise SH estimates
        global_config           (dict)             : Dictionary containing user profile related information
        logger_base             (logging object)   : Writes logs during code flow
        hvac_exit_status        (dict)             : Dictionary containing HVAC exit code and list of handled errors

    Returns:
        None
    """

    logger_local = logger_base.get("logger").getChild("populate_heating_estimates")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    heating_modes = sh_filter_info['degree_day'].columns.values

    # For each mode in hvac
    for mode in heating_modes:
        # Calculate aggregated cooling discomfort degree from epoch degree
        sh_epoch_degree = np.array(sh_filter_info['epoch_degree'][mode])
        sh_degree_for_hours = np.bincount(aggregate_identifier, sh_epoch_degree)
        sh_degree_for_hours = sh_degree_for_hours.reshape(len(sh_degree_for_hours), 1)

        # Calculate aggregated cooling consumption from epoch consumption
        epoch_sh_filtered_data = np.array(sh_filter_info['epoch_filtered_data'][mode])
        hours_sh_filtered_data = np.bincount(aggregate_identifier, epoch_sh_filtered_data)
        hours_sh_filtered_data = hours_sh_filtered_data.reshape(len(hours_sh_filtered_data), 1)
        sh_hours_selected = np.logical_and(sh_degree_for_hours > 0, hours_sh_filtered_data > 0)

        # Combine the input arrays
        filtered_data = np.c_[sh_degree_for_hours, hours_sh_filtered_data, sh_hours_selected]

        # Estimation of coefficients is only valid in historical / incremental mode
        if (global_config.get('disagg_mode') == 'historical') or (global_config.get('disagg_mode') == 'incremental'):
            # Get clusters from dbscan
            filtered_data_df, cluster_info = get_hvac_clusters(filtered_data)

            # Include high consumption epochs in case missed in clustering
            try:
                filtered_data_df, cluster_info = include_relevant_points(filtered_data_df, cluster_info, logger_hvac)
                logger_hvac.info(' >> Included other relevant heating points for regression |')

            except (ValueError, IndexError, KeyError):
                logger_hvac.info(' Unable to include relevant extra points from other invalid clusters |')

            estimation_debug['cluster_info'] = cluster_info

        elif global_config.get('disagg_mode') == 'mtd':

            cluster_info = estimation_debug['cluster_info']
            filtered_data_df = get_hvac_clusters_mtd(filtered_data, cluster_info)

        # Estimate aggregated hourly sh estimate for each mode
        x_hour_sh_mode[mode] = estimate_cluster_energy_at_hour(global_config, logger_pass, hvac_exit_status,
                                                               filtered_data_df, cluster_info, mode)


def estimate_hvac(hvac_input_data, invalid_idx, hvac_debug, hvac_params, disagg_output_object,
                  global_config, logger_base, hvac_exit_status):
    """
    Wrapper Function to estimate AC and SH appliance epoch level consumption

    Parameters:
        hvac_input_data         (np.ndarray)         : 2D Array of epoch level input data frame flowing into hvac module
        invalid_idx             (np.ndarray)         : Array of invalid epochs based on consumption and temperature
        hvac_debug              (dict)               : Dictionary containing hvac detection stage attributes
        hvac_params             (dict)               : Dictionary containing hvac algo related initialized parameters
        disagg_output_object    (dict)               : Dictionary containing all outputs
        global_config           (dict)               : Dictionary containing user profile related information
        logger_base             (logging object)     : Writes logs during code flow
        hvac_exit_status        (dict)               : Dictionary containing hvac exit code and list of handled errors

    Returns:
        x_hour_hvac_by_mode     (dict)               : Dictionary with hourly aggregated mode-wise ac and sh consumption
        epoch_hvac              (np.ndarray)         : 2D array of epoch time-stamp and epoch cooling-heating estimates
        estimation_debug        (dict)               : Dictionary containing overall estimation stage related debugging information
        hvac_exit_status        (dict)               : Dictionary containing hvac exit code and list of handled errors
    """

    logger_local = logger_base.get("logger").getChild("estimate_hvac")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # Initialise objects handling for different disagg modes and nonetypes
    if global_config.get('disagg_mode') == 'mtd':
        detection_debug = hvac_debug['detection']
        estimation_debug = hvac_debug['estimation']
        hours_aggregation_ac = estimation_debug.get('cdd', {}).get('aggregation_factor')
        hours_aggregation_sh = estimation_debug.get('hdd', {}).get('aggregation_factor')
        pre_pipeline_params = None
        logger_hvac.info(' Successfully read hvac estimation hsm |')

    else:
        detection_debug = hvac_debug['detection']
        estimation_debug = {'cdd': {}, 'hdd': {}}
        hours_aggregation_ac = disagg_output_object['switch']['hvac']['hour_aggregate_level_ac']
        hours_aggregation_sh = disagg_output_object['switch']['hvac']['hour_aggregate_level_sh']
        pre_pipeline_params = hvac_debug.get('pre_pipeline')

    # Get unique days and unique day indexes at epoch level. Followed by unique month indexes
    _, day_idx = np.unique(hvac_input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_index=False, return_inverse=True,
                           return_counts=False)
    _, month_idx = np.unique(hvac_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_inverse=True)
    all_indices = {
        'day_idx': day_idx,
        'month_idx': month_idx,
        'invalid_idx': invalid_idx
    }

    # Store important input parameters in one dictionary to be passed in the functions
    common_objects = {'hvac_input_data': hvac_input_data,
                      'all_indices': all_indices,
                      'hvac_params': hvac_params,
                      'global_config': global_config,
                      'hvac_exit_status': hvac_exit_status,
                      'pre_pipeline_params': pre_pipeline_params,
                      'disagg_output_object': disagg_output_object}

    # Intialise final dictionary : estimation_output
    x_hour_hvac_by_mode = {'heating': {}, 'cooling': {}}
    estimation_output = {
        'x_hour_hvac_by_mode': x_hour_hvac_by_mode,
        'epoch_hvac': np.zeros(hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX].shape),
        'estimation_debug': estimation_debug
    }

    # Estimate cooling
    logger_hvac.info(' ------- 1. Estimating AC -------- |')
    estimation_output_ac, hvac_exit_status = estimate_ac(detection_debug['cdd'], estimation_debug['cdd'],
                                                         hours_aggregation_ac, common_objects, logger_pass)
    x_hour_ac_by_mode = estimation_output_ac['x_hour_hvac_by_mode']
    epoch_ac = estimation_output_ac['epoch_hvac']
    estimation_debug_ac = estimation_output_ac['estimation_debug']

    # Estimate heating
    logger_hvac.info(' ------- 2. Estimating SH -------- |')
    common_objects['hvac_exit_status'] = hvac_exit_status
    estimation_output_sh, hvac_exit_status = estimate_sh(detection_debug['hdd'], estimation_debug['hdd'],
                                                         hours_aggregation_sh, common_objects, logger_pass)

    x_hour_sh_by_mode = estimation_output_sh['x_hour_hvac_by_mode']
    epoch_sh = estimation_output_sh['epoch_hvac']
    estimation_debug_sh = estimation_output_sh['estimation_debug']

    # Combine ac and sh estimated into one 2D array with different columns
    logger_hvac.info(' ------- 3. Combine AC and SH Estimates ------ |')
    x_hour_hvac_by_mode = {'heating': x_hour_sh_by_mode, 'cooling': x_hour_ac_by_mode}
    epoch_hvac = np.c_[epoch_ac, epoch_sh]
    estimation_debug = {'hdd': estimation_debug_sh, 'cdd': estimation_debug_ac}

    # Populate estimation_output object
    estimation_output['x_hour_hvac_by_mode'] = x_hour_hvac_by_mode
    estimation_output['epoch_hvac'] = epoch_hvac
    estimation_output['estimation_debug'] = estimation_debug

    if disagg_output_object['analytics']['required'] and (global_config.get('disagg_mode') != 'mtd'):
        add_setpoint_analytics(disagg_output_object, estimation_debug)
        add_estimation_analytics(disagg_output_object, x_hour_hvac_by_mode)

    generate_regression_plot = ('hvac' in global_config['generate_plots']) and (global_config.get('disagg_mode') != 'mtd')
    if generate_regression_plot and (disagg_output_object['switch']['plot_level'] >= 3):
        plot_regression_clusters(hvac_input_data, x_hour_hvac_by_mode, global_config)

    return x_hour_hvac_by_mode, epoch_hvac, estimation_debug, hvac_exit_status


def estimate_ac(detection_debug_ac, estimation_debug_ac, hours_aggregation, common_objects, logger_base):
    """
    Function to estimate AC setpoint and cooling on demand consumption

    Parameters:
        detection_debug_ac      (dict)            : Dictionary containing AC detection parameters
        estimation_debug_ac     (dict)            : Dictionary to be populated for AC estimation parameters
        hours_aggregation       (int)             : Aggregation factor for estimation
        common_objects          (dict)            : Important objects for estimation
        logger_base             (logging object)  : Writes logs during code flow

    Returns:
        estimation_debug        (dict)            : Dictionary containing estimation stage related debugging information
        hvac_exit_status        (dict)            : Dictionary containing hvac exit code and list of handled errors
    """

    static_params = hvac_static_params()

    logger_local = logger_base.get("logger").getChild("estimate_ac")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_ac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    appliance = 'AC'
    x_hour_ac_by_mode = {}
    hvac_input_data = common_objects['hvac_input_data']
    global_config = common_objects['global_config']
    hvac_params = common_objects['hvac_params']
    setpoint_list = hvac_params['setpoint']['AC']['SETPOINTS']
    all_indices = common_objects['all_indices']
    hvac_exit_status = common_objects['hvac_exit_status']

    # Initialise estimation_output object for cooling
    estimation_output = {
        'x_hour_hvac_by_mode': x_hour_ac_by_mode,
        'epoch_hvac': np.zeros(hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX].shape),
        'estimation_debug': estimation_debug_ac
    }

    # If non MTD mode, calculate setpoint and update in estimation_output object for cooling
    if (global_config.get('disagg_mode') == 'historical') or (global_config.get('disagg_mode') == 'incremental'):
        logger_ac.info(' -------------- A : Detecting setpoint ------- |')
        estimation_debug_ac, hvac_exit_status = setpoint_by_mode_at_x_hour(detection_debug_ac, appliance,
                                                                           setpoint_list, hours_aggregation, 0,
                                                                           logger_pass, common_objects)
        logger_ac.info(
            ' >> Success in detecting setpoints, Cooling : {} |'.format(estimation_debug_ac['setpoint']))

    # Assign each valid consumption epoch to either mode 0, mode 1 or outlier
    logger_ac.info(' ------------- B : Filtering epoch level consumption ------- |')
    ac_filter_info = filter_by_mode(hvac_input_data, all_indices, detection_debug_ac, logger_pass, estimation_debug_ac,
                                    hvac_params['estimation']['AC'])

    # Assign important attributes to estimation_debug dictionary
    estimation_debug_ac['cdd_daily'] = ac_filter_info['degree_day']
    estimation_debug_ac['hoursSelected'] = ac_filter_info['hours_selected']
    estimation_debug_ac['netBL'] = ac_filter_info['epoch_filtered_data']
    estimation_debug_ac['netBL_minhrs_day'] = ac_filter_info['epoch_filtered_data_minhrs_day']
    estimation_debug_ac['netBL_minhrs_month'] = ac_filter_info['epoch_filtered_data_minhrs_month']

    epoch_array = np.arange(len(hvac_input_data))
    epochs_per_hour = Cgbdisagg.SEC_IN_HOUR / global_config['sampling_rate']
    aggregation_factor = hours_aggregation * epochs_per_hour
    aggregate_identifier = ((epoch_array + static_params['inequality_handler']) // aggregation_factor).astype(int)
    logger_ac.info(' ------------ C. Running regression at {} hours aggregation |'.format(hours_aggregation))

    if (global_config.get('disagg_mode') != 'mtd') and ('cluster_info' not in estimation_debug_ac):
        estimation_debug_ac['cluster_info'] = {}
        estimation_debug_ac['cluster_info']['hvac'] = {}
        estimation_debug_ac['cluster_info']['hvac'][0] = {'regression_kind': 'linear', 'validity': False}

    # Populate hourly aggregated cooling estimates and update x_hour_ac_by_mode
    logger_ac.info(' ------- D : Regression at x - hours for AC ------- |')
    populate_cooling_estimates(ac_filter_info, estimation_debug_ac, aggregate_identifier, x_hour_ac_by_mode,
                               global_config, logger_pass, hvac_exit_status)

    # Spread estimate at epoch level
    logger_ac.info(' ------- E : Spreading HVAC at epoch level ------- |')
    epoch_ac = get_x_hour_and_epoch_hvac(x_hour_ac_by_mode, appliance, ac_filter_info,
                                         hvac_input_data, aggregate_identifier,
                                         hvac_params, logger_pass)

    # Ensure hvac estimate doesn't exceed total, at any epoch
    suppress_ac_epochs = epoch_ac[:] >= hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    epoch_ac[suppress_ac_epochs] = hvac_input_data[suppress_ac_epochs, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    estimation_output['x_hour_hvac_by_mode'] = x_hour_ac_by_mode
    estimation_output['epoch_hvac'] = epoch_ac
    estimation_output['estimation_debug'] = estimation_debug_ac

    return estimation_output, hvac_exit_status


def estimate_sh(detection_debug_sh, estimation_debug_sh, hours_aggregation, common_objects, logger_base):
    """
    Function to estimate SH setpoint and heating on demand consumption

    Parameters:
        detection_debug_sh          (dict)            : Dictionary containing SH detection parameters
        estimation_debug_sh         (dict)            : Dictionary to be populated for SH estimation parameters
        hours_aggregation           (int)             : Aggregation factor for estimation
        common_objects              (dict)            : Important objects for estimation
        logger_base                 (logging object)  : Writes logs during code flow

    Returns:
        estimation_debug            (dict)            : Dictionary containing estimation stage related debugging information
        hvac_exit_status            (dict)            : Dictionary containing hvac exit code and list of handled errors
    """

    static_params = hvac_static_params()

    logger_local = logger_base.get("logger").getChild("estimate_sh")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_sh = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # Initialise estimation_output object for cooling
    appliance = 'SH'
    x_hour_sh_by_mode = {}
    hvac_input_data = common_objects['hvac_input_data']
    global_config = common_objects['global_config']
    hvac_params = common_objects['hvac_params']
    setpoint_list = hvac_params['setpoint']['SH']['SETPOINTS']
    all_indices = common_objects['all_indices']
    hvac_exit_status = common_objects['hvac_exit_status']

    estimation_output = {
        'x_hour_hvac_by_mode': x_hour_sh_by_mode,
        'epoch_hvac': np.zeros(hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX].shape),
        'estimation_debug': estimation_debug_sh
    }
    # If non MTD mode, calculate setpoint and update in estimation_output object for cooling
    if (global_config.get('disagg_mode') == 'historical') or (global_config.get('disagg_mode') == 'incremental'):
        logger_sh.info(' -------------- A : Detecting setpoint ------- |')
        estimation_debug_sh, hvac_exit_status = setpoint_by_mode_at_x_hour(detection_debug_sh, appliance,
                                                                           setpoint_list, hours_aggregation, 0,
                                                                           logger_pass, common_objects)
        logger_sh.info(
            ' >> Success in detecting setpoints, Cooling : {} |'.format(estimation_debug_sh['setpoint']))

    logger_sh.info(' ------------- B : Filtering epoch level consumption ------- |')
    sh_filter_info = filter_by_mode(hvac_input_data, all_indices, detection_debug_sh, logger_pass, estimation_debug_sh,
                                    hvac_params['estimation']['SH'])

    # Assign important attributes to estimation_debug dictionary
    estimation_debug_sh['hdd_daily'] = sh_filter_info['degree_day']
    estimation_debug_sh['hoursSelected'] = sh_filter_info['hours_selected']
    estimation_debug_sh['netBL'] = sh_filter_info['epoch_filtered_data']
    estimation_debug_sh['netBL_minhrs_day'] = sh_filter_info['epoch_filtered_data_minhrs_day']
    estimation_debug_sh['netBL_minhrs_month'] = sh_filter_info['epoch_filtered_data_minhrs_month']

    epoch_array = np.arange(len(hvac_input_data))
    epochs_per_hour = Cgbdisagg.SEC_IN_HOUR / global_config['sampling_rate']

    aggregation_factor = hours_aggregation * epochs_per_hour
    aggregate_identifier = ((epoch_array + static_params['inequality_handler']) // aggregation_factor).astype(int)
    logger_sh.info(' ------------ C. Running regression at {} hours aggregation |'.format(hours_aggregation))

    if (global_config.get('disagg_mode') != 'mtd') and ('cluster_info' not in estimation_debug_sh):
        estimation_debug_sh['cluster_info'] = {}
        estimation_debug_sh['cluster_info']['hvac'] = {}
        estimation_debug_sh['cluster_info']['hvac'][0] = {'regression_kind': 'linear', 'validity': False}

    # Populate hourly aggregated cooling estimates and update x_hour_ac_by_mode
    logger_sh.info(' ------- D : Regression at x - hours for AC ------- |')
    populate_heating_estimates(sh_filter_info, estimation_debug_sh, aggregate_identifier, x_hour_sh_by_mode,
                               global_config, logger_pass, hvac_exit_status)

    # Spread estimate at epoch level
    logger_sh.info(' ------- E : Spreading HVAC at epoch level ------- |')
    epoch_sh = get_x_hour_and_epoch_hvac(x_hour_sh_by_mode, appliance, sh_filter_info,
                                         hvac_input_data, aggregate_identifier, hvac_params, logger_pass)

    # Ensure hvac estimate doesn't exceed total, at any epoch
    suppress_sh_epochs = epoch_sh[:] >= hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    epoch_sh[suppress_sh_epochs] = hvac_input_data[suppress_sh_epochs, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    estimation_output['x_hour_hvac_by_mode'] = x_hour_sh_by_mode
    estimation_output['epoch_hvac'] = epoch_sh
    estimation_output['estimation_debug'] = estimation_debug_sh

    return estimation_output, hvac_exit_status
