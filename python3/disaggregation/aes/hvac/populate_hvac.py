"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to populate the estimated hvac
"""

# Import python packages
import logging
import numpy as np

# Import functions from within the project
from python3.disaggregation.aer.hvac.dbscan import get_hvac_clusters
from python3.disaggregation.aer.hvac.dbscan import get_hvac_clusters_mtd
from python3.disaggregation.aes.hvac.init_hourly_smb_hvac_params import hvac_static_params
from python3.disaggregation.aer.hvac.estimate_from_regression_df import estimate_from_regression_df


def include_relevant_points(filtered_data_df, cluster_info, logger_hvac):

    """
    Function to include left over but relevant points after DBSCAN that goes into regression

    Parameters:

        filtered_data_df    (object)               : Dataframe containing filtered data with tentative cluster ids
        cluster_info        (dict)                 : cluster info with key information updated
        logger_hvac         (logging object)       : Writes logs during code flow

    Returns:

        main_df             (object)               : Dataframe containing filtered data with updated cluster ids
        info                (dict)                 : cluster info with key information updated
    """

    static_params = hvac_static_params()

    main_df = filtered_data_df
    info = cluster_info

    # checking cluster validity
    valid_clusters = []
    for cluster in info.get('hvac').keys():
        if info.get('hvac')[cluster].get('validity') == True:
            valid_clusters.append(cluster)

    df_placeholder = main_df.copy()

    # adding probable points into regression cluster
    if len(valid_clusters) == 1:

        df_valid = main_df[main_df['day_hvac_cluster'] == valid_clusters[0]]

        # defining limits of points to be added
        degree_limit_trend = np.percentile(np.array(df_valid['degree_day']), static_params.get('include_extra_points_upper'))
        cons_limit_trend = np.percentile(np.array(df_valid['filter_cons']), static_params.get('include_extra_points_upper'))

        degree_limit_off = np.percentile(np.array(df_valid['degree_day']), static_params.get('include_extra_points_lower'))
        cons_limit_off = np.percentile(np.array(df_valid['filter_cons']), static_params.get('include_extra_points_lower'))

        # adding points along the trend line and adding high consumption points, off-trend line
        df_placeholder['include'] = False
        df_placeholder['include'][(df_placeholder['degree_day'] > degree_limit_trend) & (df_placeholder['filter_cons'] > cons_limit_trend)] = True
        df_placeholder['include'][(df_placeholder['degree_day'] > degree_limit_off) & (df_placeholder['filter_cons'] > cons_limit_off)] = True

        logger_hvac.info(' Total extra valid points included : {} |'.format(np.sum(df_placeholder['include'])))
        df_placeholder['day_hvac_cluster'][df_placeholder['include'] == True] = valid_clusters[0]

    main_df['day_hvac_cluster'] = df_placeholder['day_hvac_cluster']

    return main_df, info


def mtd_estimate(cluster_info, filtered_data_df, filter_day, identifier, logger_hvac, hvac_estimate_x_hour):

    """
    Function to estimate hvac appliance in mtd mode

    Parameters:

        cluster_info        (dict)              : Contains validity info of clusters and regression type
        filtered_data_df    (pd.Dataframe)      : Dataframe containing cluster id
        filter day          (np.ndarray)        : Array containing boolean of valid consumption and cdd/hdd
        identifier          (dict)              : Dictionary containing attributes of ac/sh
        logger_hvac         (logging object)    : Writes logs during code flow
        hvac_estimate_x_hour(np.ndarray)        : Array containing hvac estimates at x-hour level

    Returns:

        hvac_estimate_x_hour(np.ndarray)        : Array containing hvac estimates at x-hour level
    """

    # making estimates in mtd mode using degree day, regression kind and coefficients
    for cluster in cluster_info.keys():
        # getting data for making estimates
        regression_df = \
        filtered_data_df[(filtered_data_df[filter_day] == 1) & (filtered_data_df[identifier['cluster_id']] == cluster)][
            list(identifier.values())]

        # making estimates ony for valid clusters
        if cluster_info[cluster]['validity'] == True:

            # reading regression coefficient
            logger_hvac.info(' >> Estimating from hsm coefficients |')
            hvac_coefficient = cluster_info[cluster]['coefficient']

            # estimates for linear trend
            if cluster_info[cluster]['regression_kind'] == 'linear':

                hvac_estimate = list(np.array(regression_df[identifier['degree_day']]) * hvac_coefficient[0])
                hvac_estimate = [0 if cluster_flag != cluster else hvac_estimate.pop(0)
                                 for cluster_flag in filtered_data_df[identifier['cluster_id']]]
                hvac_estimate_x_hour = np.sum([hvac_estimate_x_hour, hvac_estimate], axis=0)

            # estimates for root trend
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

        global_config       (dict)                  : Dictionary containing user profile related information
        logger_base         (logging object)        : Writes logs during code flow
        hvac_exit_status    (dict)                  : Dictionary containing hvac exit code and list of handled errors
        filtered_data_df    (pandas dataframe)      : Dataframe containing cluster id
        cluster_info        (dict)                  : Contains validity info of clusters and regression type
        mode                (int)                   : Mode identifier in loop

    Returns:

        estimation_debug    (dict)                  : Dictionary containing estimation stage related key information
    """

    # initializing logging object
    logger_local = logger_base.get("logger").getChild("estimate_consumption")
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # reading info about each hvac cluster
    cluster_info_master = cluster_info

    # making a dictionary bag to carry key info needed to make estimates
    appliance = 'hvac'
    filter_day = 'filter_day'
    cluster_info = cluster_info['hvac']
    identifier = {'degree_day': 'degree_day',
                  'day_consumption': 'filter_cons',
                  'day_validity': 'filter_day',
                  'cluster_id': 'day_hvac_cluster'}

    # estimates for historical and incremental runs only
    if (global_config.get('disagg_mode') == 'historical') or (global_config.get('disagg_mode') == 'incremental'):

        # checking if any regression points exist
        master_regression_df = filtered_data_df[(filtered_data_df[filter_day] == 1)][list(identifier.values())]
        hvac_estimate_x_hour = list(np.zeros(len(filtered_data_df)))
        points_for_regression = np.array(filtered_data_df[filter_day])

        # estimates would be made only if regression points exist
        if any(points_for_regression):

            regression_carrier_dict = {}
            regression_carrier_dict['cluster_info'] = cluster_info
            regression_carrier_dict['filtered_data_df'] = filtered_data_df
            regression_carrier_dict['filter_day'] = filter_day
            regression_carrier_dict['identifier'] = identifier
            regression_carrier_dict['cluster_info_master'] = cluster_info_master
            regression_carrier_dict['appliance'] = appliance
            regression_carrier_dict['hvac_estimate_day'] = hvac_estimate_x_hour

            # making estimates for mode
            hvac_estimate_x_hour = estimate_from_regression_df(regression_carrier_dict, logger_hvac)

        else:

            # failsafe default estimates for mode
            logger_hvac.info(' No points for regression in mode {} |'.format(mode))
            hvac_estimate_x_hour = np.array(hvac_estimate_x_hour)

    elif global_config.get('disagg_mode') == 'mtd':

        # making estimates in mtd
        master_regression_df = filtered_data_df[(filtered_data_df[filter_day] == 1)][list(identifier.values())]
        hvac_estimate_x_hour = list(np.zeros(len(filtered_data_df)))
        points_for_regression = np.array(filtered_data_df[filter_day])

        # checking if regression points exist
        if any(points_for_regression):

            # estimate with closest mode
            hvac_estimate_x_hour = mtd_estimate(cluster_info, filtered_data_df, filter_day, identifier, logger_hvac, hvac_estimate_x_hour)

        else:

            # fail-safe estimate for mode
            logger_hvac.info(' No points for regression in mode {} |'.format(mode))
            hvac_estimate_x_hour = np.array(hvac_estimate_x_hour)

    # updating estimation debug dictionary
    estimation_debug = {
        'cluster_info': cluster_info,
        'regression_df': master_regression_df,
        'hvac_estimate_day': np.array(hvac_estimate_x_hour),
        'exit_status': hvac_exit_status
    }

    return estimation_debug


def populate_cooling_estimates(ac_filter_info, aggregate_identifier, x_hour_hvac_mode, global_config, logger_base,
                               hvac_exit_status, estimation_debug):

    """
    Function to populate hvac estimates at epoch level

    Parameters:

        ac_filter_info          (dict)             : Dictionary containing key information post filtering stage
        aggregate_identifier    (int)              : Integer identifying at what hour level aggregation has to be done
        x_hour_hvac_mode        (dict)             : Dictionary initialized to carry mode wise hvac estimates
        global_config           (dict)             : Dictionary containing user profile related information
        logger_base             (logging object)   : Writes logs during code flow
        hvac_exit_status        (dict)             : Dictionary containing HVAC exit code and list of handled errors

    Returns:
        None
    """

    # initializing debug
    logger_local = logger_base.get("logger").getChild("populate_cooling_estimates")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # getting cooling modes
    cooling_modes = ac_filter_info['degree_day'].columns.values

    # populating cooling estimates for every mode
    for mode in cooling_modes:

        # reading epoch level degree measure
        ac_epoch_degree = np.array(ac_filter_info['epoch_degree'][mode])

        # aggregating degrees at x-hours
        ac_degree_x_hours = np.bincount(aggregate_identifier, ac_epoch_degree)
        ac_degree_x_hours = ac_degree_x_hours.reshape(len(ac_degree_x_hours), 1)

        # aggregating ac filtered data at x-hour level
        epoch_ac_filtered_data = np.array(ac_filter_info['epoch_filtered_data'][mode])
        x_hours_ac_filtered_data = np.bincount(aggregate_identifier, epoch_ac_filtered_data)
        x_hours_ac_filtered_data = x_hours_ac_filtered_data.reshape(len(x_hours_ac_filtered_data), 1)

        # selecting only valid consumption and degree days aggregates
        ac_hours_selected = np.logical_and(ac_degree_x_hours > 0, x_hours_ac_filtered_data > 0)

        logger_hvac.info(' Mode : {} , Number of data points selected at aggregation level : {}  |'.format(mode, np.sum(ac_hours_selected)))

        # making array of filtered valid data points
        filtered_data = np.c_[ac_degree_x_hours, x_hours_ac_filtered_data, ac_hours_selected]

        # getting cluster info for each regression mode in historical and incremental runs
        if (global_config.get('disagg_mode') == 'historical') or (global_config.get('disagg_mode') == 'incremental'):

            # getting hvac clusters
            filtered_data_df, cluster_info = get_hvac_clusters(filtered_data)

            # noinspection PyBroadException
            try:
                filtered_data_df, cluster_info = include_relevant_points(filtered_data_df, cluster_info, logger_hvac)
                logger_hvac.info(' >> Included other relevant cooling points for regression |')

            except (ValueError, IndexError, KeyError):
                logger_hvac.info(' Unable to include relevant extra points from other invalid clusters |')

            # populating estimation debug dictionary with cluster info
            estimation_debug['cdd']['cluster_info'] = cluster_info

        elif global_config.get('disagg_mode') == 'mtd':

            # making estimates in mtd mode
            cluster_info = estimation_debug['cdd']['cluster_info']

            # finding clusters for aggregate points in mtd mode
            filtered_data_df = get_hvac_clusters_mtd(filtered_data, cluster_info)

        x_hour_hvac_mode['cooling'][mode] = estimate_cluster_energy_at_hour(global_config, logger_pass, hvac_exit_status,
                                                                            filtered_data_df, cluster_info, mode)


def populate_heating_estimates(sh_filter_info, aggregate_identifier, x_hour_hvac_mode, global_config, logger_base, hvac_exit_status, estimation_debug):

    """
    Function to populate hvac estimates at epoch level

    Parameters:
        sh_filter_info      (dict)             : Dictionary containing key information post filtering stage
        aggregate_identifier(int)              : Integer identifying at what hour level aggregation has to be done
        x_hour_hvac_mode    (dict)             : Dictionary initialized to carry mode wise hvac estimates
        global_config       (dict)             : Dictionary containing user profile related information
        logger_base         (logging object)   : Writes logs during code flow
        hvac_exit_status    (dict)             : Dictionary containing HVAC exit code and list of handled errors
        estimation_debug    (dict)             : Dictionary containing all key information related to HVAC estimation

    Returns:
        None
    """

    # initializing logging object
    logger_local = logger_base.get("logger").getChild("populate_heating_estimates")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # getting heating modes
    heating_modes = sh_filter_info['degree_day'].columns.values

    # populating heating estimates for every mode
    for mode in heating_modes:

        # reading epoch level degree measure
        sh_epoch_degree = np.array(sh_filter_info['epoch_degree'][mode])

        # aggregating degrees at x-hours
        sh_degree_for_hours = np.bincount(aggregate_identifier, sh_epoch_degree)
        sh_degree_for_hours = sh_degree_for_hours.reshape(len(sh_degree_for_hours), 1)

        # aggregating sh filtered data at x-hour level
        epoch_sh_filtered_data = np.array(sh_filter_info['epoch_filtered_data'][mode])
        hours_sh_filtered_data = np.bincount(aggregate_identifier, epoch_sh_filtered_data)
        hours_sh_filtered_data = hours_sh_filtered_data.reshape(len(hours_sh_filtered_data), 1)

        # selecting only valid consumption and degree days aggregates
        sh_hours_selected = np.logical_and(sh_degree_for_hours > 0, hours_sh_filtered_data > 0)
        filtered_data = np.c_[sh_degree_for_hours, hours_sh_filtered_data, sh_hours_selected]

        # getting cluster info for each regression mode in historical and incremental runs
        if (global_config.get('disagg_mode') == 'historical') or (global_config.get('disagg_mode') == 'incremental'):

            # getting hvac clusters
            filtered_data_df, cluster_info = get_hvac_clusters(filtered_data)

            # noinspection PyBroadException
            try:
                filtered_data_df, cluster_info = include_relevant_points(filtered_data_df, cluster_info, logger_hvac)
                logger_hvac.info(' >> Included other relevant heating points for regression |')

            except (ValueError, IndexError, KeyError):
                logger_hvac.info(' Unable to include relevant extra points from other invalid clusters |')

            # populating estimation debug dictionary with cluster info
            estimation_debug['hdd']['cluster_info'] = cluster_info

        elif global_config.get('disagg_mode') == 'mtd':

            # making estimates in mtd mode
            cluster_info = estimation_debug['hdd']['cluster_info']

            # finding clusters for aggregate points in mtd mode
            filtered_data_df = get_hvac_clusters_mtd(filtered_data, cluster_info)

        x_hour_hvac_mode['heating'][mode] = estimate_cluster_energy_at_hour(global_config, logger_pass, hvac_exit_status,
                                                                            filtered_data_df, cluster_info, mode)
