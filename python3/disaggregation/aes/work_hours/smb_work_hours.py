"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to run main smb functions
"""

# Import python packages
import copy
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aes.work_hours.work_hour_utils import get_labels
from python3.disaggregation.aes.work_hours.work_hour_utils import get_alt_label
from python3.disaggregation.aes.work_hours.work_hour_utils import get_binned_data
from python3.disaggregation.aes.work_hours.work_hour_utils import check_work_hour_spread
from python3.disaggregation.aes.work_hours.work_hour_utils import apply_continuity_filter
from python3.disaggregation.aes.work_hours.work_hour_utils import align_work_hours_with_hsm
from python3.disaggregation.aes.work_hours.work_hour_utils import align_epoch_work_hours_with_hsm

from python3.disaggregation.aes.work_hours.user_lvl_work_hours import get_common_work_hours
from python3.disaggregation.aes.work_hours.user_lvl_work_hours import user_work_hour_pre_process
from python3.disaggregation.aes.work_hours.user_lvl_work_hours import user_work_hour_post_process

from python3.disaggregation.aes.work_hours.epoch_work_hours import get_epoch_work_hours
from python3.disaggregation.aes.work_hours.epoch_work_hours import epoch_work_hour_preprocess
from python3.disaggregation.aes.work_hours.epoch_work_hours import post_process_epoch_work_hours

from python3.disaggregation.aes.work_hours.operational_and_xao.get_operational_and_xao import get_operational_and_xao


def get_smb_components(global_config, month_ao_hvac_res_net, epoch_ao_hvac_true, disagg_input_object,
                       disagg_output_object, column_index, logger_base):
    """
    Function to extract special SMB components like Operational Load and Extra-AO

    Parameters:
        global_config           (dict)             : Dictionary containing user level global config parameters
        month_ao_hvac_res_net   (np.array)         : Array containing | month-ao-ac-sh-residue-net energies
        epoch_ao_hvac_true      (np.array)         : Array containing | epoch-ao-ac-sh energies
        disagg_input_object     (dict)             : Dictionary containing all input attributes
        disagg_output_object    (dict)             : Dictionary containing all output attributes
        column_index            (dict)             : Dictionary containing column identifier indices of ao-ac-sh
        logger_base             (logging object)   : Writes logs during code flow

    Returns:
        month_ao_hvac_res_net   (np.array)         : Array containing | month-ao-ac-sh-residue-net energies (Processed)
        epoch_ao_hvac_true      (np.array)         : Array containing | epoch-ao-ac-sh energies (Processed)
    """

    # getting operational load and extra ao
    if not (global_config.get('disagg_mode') == 'mtd'):

        # getting general operational and extra ao
        month_ao_hvac_res_net, epoch_ao_hvac_true = get_operational_and_xao(month_ao_hvac_res_net, epoch_ao_hvac_true,
                                                                            disagg_input_object, disagg_output_object,
                                                                            column_index, logger_base)
    else:

        # fail-safe operational load and extra ao
        columns_to_make = 6
        month_ao_hvac_res_net = np.c_[
            month_ao_hvac_res_net, np.zeros((month_ao_hvac_res_net.shape[0], columns_to_make))]
        epoch_ao_hvac_true = np.c_[epoch_ao_hvac_true, np.zeros((epoch_ao_hvac_true.shape[0], columns_to_make))]

    return month_ao_hvac_res_net, epoch_ao_hvac_true


def cluster_binned_data(input_data, binned_data, cluster_count, parameters):
    """
    Function to cluster binned data for alternate work hours
    Parameters:
        input_data    (pd.DataFrame): DataFrame with complete input data and KMeans labels
        binned_data   (list)        : list of arrays, with each array having seasonal data
        cluster_count (int)         : Count of clusters to be created using KMeans
        parameters    (dict)        : Dictionary with work hour specific parameters
    Returns:
        input_data    (pd.DataFrame): DataFrame with complete input data and KMeans labels
    """
    for valid_epochs in binned_data:
        if len(input_data['raw-ao'][valid_epochs]) > 2:

            kmean_data = np.nan_to_num(input_data['raw-ao'][valid_epochs].values.reshape(-1, 1))
            threshold_cluster = np.nanmedian(kmean_data) + (cluster_count * np.nanstd(kmean_data))
            kmean_data[kmean_data > threshold_cluster] = threshold_cluster
            kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(kmean_data)

            if np.max(kmeans.cluster_centers_) - np.min(kmeans.cluster_centers_) <= parameters.get('cluster_diff'):
                input_data['kmean_labels'][valid_epochs] = 1

            else:
                input_data['kmean_labels'][valid_epochs] = get_labels(kmeans.cluster_centers_, kmeans.labels_)

    return input_data


def get_alternate_work_hours(in_data, disagg_output_object, parameters, static_params, logger_base):
    """
    Alternate function to detect work hours. Used only when method 1 identifies 24x7 or very sparse work hours
    Parameters:
        in_data               (pd.DataFrame)   : Data frame with input raw data
        disagg_output_object  (dict)           : Dictionary with pipeline level output data
        parameters            (dict)           : Dictionary object wth variables needed for clustering
        static_params         (dict)           : Dictionary containing work hour specific constants
        logger_base           (logging object) : Writes logs during code flow

    Returns:
        labels_df             (np.array)   : 2D array with epoch level boolean flag indicating open / close status
        cons_level            (np.float64) : Float indicating consumption level of the user after removing detected apps
    """

    logger_local = logger_base.get("logger").getChild("get_alternate_work_hours")
    logger_alt_work_hours = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    input_data = copy.deepcopy(in_data)

    ao_hvac = disagg_output_object['ao_seasonality']['epoch_cooling'] + disagg_output_object['ao_seasonality'][
        'epoch_heating']

    input_data['raw-ao'] = input_data['raw-ao'] - ao_hvac
    input_data['raw-ao'][input_data['raw-ao'] < 0] = 0
    logger_alt_work_hours.info('Removed AO HVAC from raw input before calculating alternate work hours |')

    max_val = np.nanpercentile(input_data['raw-ao'], 99)
    input_data['raw-ao'][input_data['raw-ao'] > max_val] = max_val

    cons_level = np.nan_to_num(np.nanpercentile(input_data['raw-ao'], 75))
    input_data['kmean_labels'] = np.zeros_like(input_data['s_label'])
    input_data['s_label'] = input_data['s_label'].ffill(limit=3)

    binned_data = get_binned_data(input_data)
    logger_alt_work_hours.info('Binned the input data based on season label before beginning with clustering |')

    cluster_count = static_params.get('cluster_count').get(parameters.get('cons_level'))
    func_params = static_params.get('alternate_work_hours')

    logger_alt_work_hours.info('Count of clusters to be used for KMeans clustering | {}'.format(cluster_count))

    input_data = cluster_binned_data(input_data, binned_data, cluster_count, parameters)

    input_data['kmean_labels'] = apply_continuity_filter(input_data['kmean_labels'],
                                                         np.maximum(2 * parameters.get('sampling'), 4))
    logger_alt_work_hours.info('Obtained work hours using alternate method. Post processing the results |')

    labels_df = input_data.pivot_table(index='date', columns='time', values='kmean_labels')

    hours_per_day = labels_df.sum(axis=1)
    labels_df[hours_per_day <= parameters.get('sampling')] = 0
    disagg_output_object['alt_open_close_table'] = np.nan_to_num(labels_df)
    days_with_labels = hours_per_day > 0
    days_with_cons = input_data.pivot_table(index='date', values='raw-ao', aggfunc=np.sum).values > 0

    # Calculating diff to identify count of bands.
    logger_alt_work_hours.info('Calculating the count of work-hour bands for the user |')
    diff_labels = np.nan_to_num(labels_df.values[:, :-1] - labels_df.values[:, 1:])

    # Identify days with more than 3 work bands and days with more than 8 work hours
    days_with_gt_3_work_bands = np.sum(diff_labels < 0, axis=1) > func_params.get('defined_user_work_band_thresh')
    days_with_gt_8_work_hours = hours_per_day > func_params.get('defined_user_work_hour_thresh') * parameters.get('sampling')
    condition = np.logical_or(days_with_gt_3_work_bands, days_with_gt_8_work_hours)

    # If more than 70% of days are marked as either,
    # mark the user as 0x7 and send it back to be replaced by the originally detected work hours.
    if np.sum(condition) >= np.sum(days_with_labels) * func_params.get('defined_user_days_frac_thresh'):
        labels_df = np.zeros_like(labels_df)
        logger_alt_work_hours.info('More than 70% of days have >2 work-hour bands or >8 work hours | '
                                   'Marking overall work hours as 0x7')

    else:
        labels_df = input_data.pivot_table(index='date', columns='time', values='kmean_labels')

        # Large strike days to be made 0
        hours_per_day = labels_df.sum(axis=1)
        hours_per_hour = labels_df.sum(axis=0)

        # If the work hours are restricted to a 12 hour block, allow 12 hour long streaks
        if np.sum(hours_per_hour > np.nanmax(hours_per_hour) / 4) <= (func_params.get('max_allowed_hour_block') *
                                                                      parameters.get('sampling')):
            labels_df[hours_per_day > func_params.get('max_allowed_hour_block') * parameters.get('sampling')] = 0

        # If the work hours are spread out to larger than a 18 hour block, mark it a 24x7 user & return
        elif np.sum(hours_per_hour > np.nanmax(hours_per_hour) / 4) >= (func_params.get('min_hour_block_for_24x7') *
                                                                        parameters.get('sampling')):
            labels_df[:] = 1
            logger_alt_work_hours.info('Detected work hours are spread over >18 hours across different days | '
                                       'Marking overall work hour as 24x7')
            return np.nan_to_num(labels_df), cons_level

        # If the detected points are spread out, allow days with only up to 8 hours of work hours
        else:
            labels_df[hours_per_day >= func_params.get('max_work_hour_per_day') * parameters.get('sampling')] = 0

        # Days with more than 5 streaks to be made zero
        # TODO:(Neelabh) Change the logic to retain top 3 longest streaks for such days
        diff_labels = np.nan_to_num(labels_df.values[:, :-1] - labels_df.values[:, 1:])
        labels_df[np.sum(diff_labels < 0, axis=1) > func_params.get('max_allowed_streaks_per_day')] = 0
        logger_alt_work_hours.info('Removing days with >5 work hour bands |')

        # Check if the work hour is spread across the year
        labels_df = check_work_hour_spread(labels_df, days_with_cons, hours_per_day, parameters)

    # TODO: (Neelabh) Add logic to remove noise by calculating the median of long strikes and suppress the small dots
    logger_alt_work_hours.info('Successfully calculated work hours using the alternate method |')

    return np.nan_to_num(labels_df), cons_level


def get_work_hours_method1(input_df, input_pivot_data, disagg_output_object, data_parameters, static_params,
                           survey_work_hours, logger_pass, logger_work_hours, hsm_in, hsm_fail):
    """
    Wrapper function for identifying work hours using method 2
    Parameters:
        input_df             (pd.DataFrame) : Contains complete raw data:
        input_pivot_data     (np.array)     : 2-D array containing consumption data for the user
        disagg_output_object (dict)         : Object with all relevant o/p and allied info by other modules
        data_parameters      (dict)         : Dictionary containing work hour specific extracted parameters
        static_params        (dict)         : Dictionary containing work hour specific parameters
        survey_work_hours    (dict)         : Object containing survey input with regards to the business work hours
        logger_pass          (logger)       : Logger to pass in all called functions
        logger_work_hours    (logger)       : Logger to log statements in this function
        hsm_in               (dict)         : Dictionary object with work-hour hsm for the user
        hsm_fail             (Boolean)      : Identifies presence of HSM for the given user

    Returns:
        labels_df       (np.array)    : 2D numpy-array with epoch level open / close boolean information
        change_val      (int)         : Identifies if the detected work hours were changed using HSM
    """

    labels_df = np.zeros_like(input_pivot_data)
    logger_work_hours.info(' Getting overall work hours for the user | ')
    common_work_hours, all_labels = get_common_work_hours(input_pivot_data, data_parameters, static_params, logger_pass)

    logger_work_hours.info(' Post Processing overall work hours for the user | ')
    work_hours_year = user_work_hour_post_process(common_work_hours, all_labels, data_parameters, static_params,
                                                  logger_pass)

    logger_work_hours.debug(' Overall work hour array after post processing for the user is | {}'.format(work_hours_year))

    if not hsm_fail:
        hsm_params = static_params.get('hsm_params')
        data_parameters['hsm'] = hsm_in.get('attributes')
        logger_work_hours.info('Trying to use HSM to change the work hours of the user at user level| ')
        work_hours_year, change_val_user, hsm_used = align_work_hours_with_hsm(work_hours_year, data_parameters,
                                                                               all_labels, hsm_params,
                                                                               logger_work_hours)

    else:
        hsm_used = False
        data_parameters['hsm'] = None
        change_val_user = 0

    if hsm_used:
        logger_work_hours.info('HSM used to change the work hours of the user at user level| ')
    else:
        logger_work_hours.info('HSM NOT used to change the work hours of the user at user level| ')

    logger_work_hours.info(' Beginning wth epoch level work hour detection |')

    work_hours_year_processed = np.zeros_like(work_hours_year)
    if 1 < np.sum(work_hours_year):
        # Find daily variation only if detected yearly work hours is not 0x7
        work_hours_year_processed = epoch_work_hour_preprocess(work_hours_year, static_params,
                                                               data_parameters.get('sampling'))

        if not np.sum(work_hours_year_processed) > static_params.get('max_work_hours') * data_parameters.get('sampling'):

            yearly_label_df = np.tile(work_hours_year_processed.reshape(1, -1), (input_pivot_data.shape[0], 1))
            labels_df, clustering_info = get_epoch_work_hours(data_parameters, input_df, disagg_output_object,
                                                              static_params, logger_pass)
            clustering_info['yearly_work_hour'] = work_hours_year

            logger_work_hours.info(' Post processing epoch level work hours | ')
            labels_df = post_process_epoch_work_hours(labels_df, data_parameters.get('sampling'), yearly_label_df,
                                                      clustering_info, disagg_output_object, input_df, static_params,
                                                      survey_work_hours, logger_pass)

        else:
            labels_df = np.ones_like(labels_df)

    # If hsm is present and was used at user level to convert the user to 24x7 user, no need to re-align
    if not hsm_fail and not (hsm_used and np.sum(work_hours_year_processed) == len(work_hours_year_processed)):
        data_parameters['hsm'] = hsm_in.get('attributes')
        logger_work_hours.info('Trying to use HSM to change the work hours of the user at epoch level| ')
        labels_df, change_val_epoch, hsm_used = align_epoch_work_hours_with_hsm(labels_df, data_parameters, all_labels,
                                                                                hsm_params, logger_work_hours)
    else:
        hsm_used = False
        change_val_epoch = 0

    if hsm_used:
        logger_work_hours.info('HSM used to change the work hours of the user at epoch level| ')
    else:
        logger_work_hours.info('HSM NOT used to change the work hours of the user at epoch level| ')

    logger_work_hours.info(' Epoch work hour detection by Method 1 done | ')

    return labels_df, np.maximum(change_val_user, change_val_epoch)


def get_work_hours_method2(input_df, input_pivot_data, disagg_output_object, data_parameters, data_len, static_params,
                           logger_pass, logger_work_hours):
    """
    Wrapper function for identifying work hours using method 2
    Parameters:
        input_df             (pd.DataFrame)   : Contains complete raw data:
        input_pivot_data     (np.array)       : 2-D array containing consumption data for the user
        disagg_output_object (dict)           : Object with all relevant o/p and allied info by other modules
        data_parameters      (dict)           : Dictionary containing work hour specific extracted parameters
        data_len             (int)            : Identifies the total size of the data
        static_params        (dict)           : Dictionary containing work hour specific parameters
        logger_pass          (logger)         : Logger to pass in all called functions
        logger_work_hours    (logger)         : Logger to log statements in this function

    Returns:
        labels_df       (np.array)    : 2D numpy-array with epoch level open / close boolean information
        alt_label       (int)         : Flag to indicate if work hours was identified using alternate logic
    """

    alt_label = 0

    labels_df_, cons_level = get_alternate_work_hours(input_df, disagg_output_object, data_parameters,
                                                      static_params, logger_pass)

    if np.sum(labels_df_) <= data_len * static_params.get('min_work_label_perc'):

        if cons_level <= static_params.get('min_cons_thresh_24x7_user') / data_parameters.get('sampling'):
            labels_df = np.zeros_like(input_pivot_data)
            logger_work_hours.info(' User identified as 0x7 since Open Close detected is very sparse | ')

        else:
            labels_df = np.ones_like(input_pivot_data)
            logger_work_hours.info(' Open close re-marked as 24x7 since alternate labels not useful | ')

    elif np.sum(labels_df_) == labels_df_.size:

        labels_df = copy.deepcopy(labels_df_)
        alt_label = 0
        logger_work_hours.info(' Open close re-marked as 24x7 since alternate labels not useful | ')

    else:

        alt_label = 1
        labels_df = copy.deepcopy(labels_df_)

    return labels_df, alt_label


def get_work_hours(in_data, disagg_output_object, cons_level, smb_type, static_params, survey_work_hours, logger_base,
                   hsm_in=None, hsm_fail=True):
    """
    Function calculates user level as well as epoch level work hour booleans for the given user.
    Parameters:
        in_data              (pd.DataFrame)   : Contains complete raw data
        disagg_output_object (dict)           : Object with all relevant o/p and allied info by other modules
        cons_level           (string)         : Categorical variable to define consumption level of the user
        smb_type             (string)         : Survey provided SMB type of the user
        static_params        (dict)           : Dictionary containing work hour specific parameters
        survey_work_hours    (dict)           : Object containing survey input with regards to the business work hours
        logger_base          (logging object) : Writes logs during code flow
        hsm_in               (dict)           : Dictionary object with work-hour hsm for the user
        hsm_fail             (Boolean)        : Identifies presence of HSM for the given user

    Returns:
        labels_df            (np.array)    : 2D numpy-array with epoch level open / close boolean information
    """
    logger_local = logger_base.get("logger").getChild("work_hours")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_work_hours = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # We can write a small module here to differentiate the church smb from other smbs
    # Basically do clustering on the whole data divided into three seasons(summer, winter, transition).
    # We should ideally be removing AO HVAC, BaseLoad and external_Light before these calculations.
    # After that analysed the obtained clusters, if these clusters resemble that of a church (weekly pattern, 1/2strikes
    # during day hours, AO most of the year) then mark the user as church and results of this clustering as Work Hours.

    # # Main algo starts
    input_df = copy.deepcopy(in_data)

    # Initializing a default hsm dictionary
    hsm_data_dict = {
        'last_timestamp': np.NaN
    }

    data_parameters, input_pivot_data = user_work_hour_pre_process(input_df, cons_level, static_params, logger_pass)
    data_parameters['cons_level'] = cons_level

    labels_df = np.zeros_like(input_pivot_data)
    valid_days = data_parameters.get('valid_days')
    change_val = 0

    if np.sum(valid_days * 1) < static_params.get('min_non_zero_days_perc') * input_pivot_data.shape[0]:
        logger_work_hours.info(' Non-zero consumption value days < 5% | Not detecting any work hours for the user')
        return labels_df, valid_days, hsm_data_dict

    # The total cons should be at least 5KWh per month for us to identify work hours
    elif np.nansum(input_pivot_data) < static_params.get('min_avg_monthly_cons') * (input_pivot_data.shape[0] / Cgbdisagg.DAYS_IN_MONTH):
        logger_work_hours.info(' Avg monthly consumption is less than 5Kwh | Not detecting any work hours for the user')
        return labels_df, valid_days, hsm_data_dict

    alt_label = get_alt_label(hsm_in)
    if not alt_label and not smb_type == 'CHURCH':
        labels_df, change_val = get_work_hours_method1(input_df, input_pivot_data, disagg_output_object,
                                                       data_parameters, static_params, survey_work_hours, logger_pass,
                                                       logger_work_hours, hsm_in, hsm_fail)

    labels_df_valid = labels_df[valid_days]
    data_len = labels_df_valid.size

    run_alt_work_hours = (alt_label or hsm_fail) and np.sum(labels_df) <= data_len * static_params.get('min_non_zero_days_perc')

    if smb_type == 'CHURCH' or run_alt_work_hours:
        logger_work_hours.info(' Getting alternate work hours for the user | SMB type: {}'.format(smb_type))
        labels_df, alt_label = get_work_hours_method2(input_df, input_pivot_data, disagg_output_object, data_parameters,
                                                      data_len, static_params, logger_pass, logger_work_hours)

    hsm_data_dict = {
        'last_timestamp': in_data.values[-1, Cgbdisagg.INPUT_EPOCH_IDX],
        'work_arr': (np.nansum(labels_df, axis=0) > 0.25 * labels_df.shape[0]) * 1,
        'change_val': change_val,
        'alt_label': alt_label
    }

    return labels_df, valid_days, hsm_data_dict
