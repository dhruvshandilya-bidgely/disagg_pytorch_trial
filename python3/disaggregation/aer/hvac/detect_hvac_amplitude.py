"""
Author - Abhinav
Date - 10/10/2018
Module to detect HVAC amplitude
"""

# Import python packages
import os
import copy
import logging
import matplotlib
import numpy as np
from scipy import stats
from sklearn import mixture
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles
matplotlib.use('Agg')

# Import packages from within the pipeline
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params
from python3.disaggregation.aer.hvac.adjust_hvac_params import adjust_ac_detection_range
from python3.disaggregation.aer.hvac.adjust_hvac_params import get_ac_adjusted_detection_params
from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile as super_percentile
from python3.disaggregation.aer.hvac.get_gmm_cluster_ranges_detection import fit_optimum_gaussian
from python3.disaggregation.aer.hvac.get_gmm_cluster_ranges_detection import fit_residual_gaussian

hvac_static_params = hvac_static_params()


def manage_plot_location(global_config):
    """
    Function to get filename of detection diagnostic plot. Over-writes plot if already exists at location.

    Parameters:
        global_config       (dict)  : Dictionary containing  all user level global config parameters. Need uuid.

    Returns:
        file_name           (str)   : String name of the detection diagnostic plot
        figure_directory    (str)   : String name of the detection diagnostic plot directory
    """

    figure_directory = hvac_static_params.get('path').get('hvac_plots')

    disagg_mode = global_config.get('disagg_mode', 'historical')
    input_data = global_config['switch']['hvac_input_data_timed_removed'][:, Cgbdisagg.INPUT_EPOCH_IDX]
    start, end = int(np.nanmin(input_data)), int(np.nanmax(input_data))
    figure_directory = figure_directory + '/' + "{uuid}".format(uuid=global_config['uuid'])

    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    file_name = figure_directory + "/{disagg_mode}_amplitude_detection_{uuid}_{start}_{end}".format\
        (disagg_mode=disagg_mode, start=start, end=end, uuid=global_config['uuid']) + ".png"

    if os.path.isfile(file_name):
        os.remove(file_name)

    return file_name, figure_directory


def fill_center_labels(bin_centers, cluster_limits, center_labels):
    """
    Function to assign mode id's to bins

    Parameters:
        bin_centers     (np.ndarray)     : Array containing all the bin centers
        cluster_limits  (np.ndarray)     : Array containing all the bin limits (AC/SH)
        center_labels   (list)           : Initialized empty list of bin labels

    Returns:
        cluster_labels  (list)           : Filled list of bin labels
    """

    for center in bin_centers:

        if (center > cluster_limits[0][0]) and (center < cluster_limits[0][1]):
            center_labels.append(0)
        elif (center > cluster_limits[1][0]) and (center < cluster_limits[1][1]):
            center_labels.append(1)
        else:
            center_labels.append(-1)

    return center_labels


def beautify_detection_plot(bin_centers, mode_info, debug_detection, center_labels, axis_flat, appliance):
    """
    Function to add gaussian curves, to beautify modes

    Parameters:
        bin_centers         (np.ndarray)   : Array containing all the bin centers
        mode_info           (dict)         : Dictionary containing mode information about AC and SH
        debug_detection     (dict)         : Dictionary containing hvac detection related debug information
        center labels       (list)         : List containing labels for all bins
        axis_flat           (np.ndarray)   : Array of axes
        appliance           (str)          : Identifier of kind of HVAC appliance

    Returns:
        None
    """

    if appliance == 'AC':
        axis_location = 1
        identifier = 'cdd'
    else:
        axis_location = 2
        identifier = 'hdd'

    xpdf = np.linspace(np.min(bin_centers), np.max(bin_centers), 1000)
    models = mode_info['all_gaussians']
    n_components = mode_info['number_of_modes']
    clf = models[n_components - 1]
    sorted_mode_idx = mode_info['mode_idx_for_plotting']

    for i in range(clf.n_components):
        pdf = clf.weights_[i] * stats.norm(clf.means_[i, 0], np.sqrt(clf.covariances_[i, 0])).pdf(xpdf)
        scale = np.max(debug_detection[identifier]['histDiff'][center_labels == sorted_mode_idx[i]]) / np.max(pdf)
        pdf = pdf * scale
        axis_flat[axis_location].plot(xpdf, pdf, color='grey', alpha=0.50)
        axis_flat[axis_location].fill_between(xpdf, pdf, alpha=0.20, color='grey')


def plot_amplitude_detection(global_config, debug_detection, quan_005, quan_995, hvac_params, logger_base):
    """
    Function to plot the AC and SH amplitudes

    Parameters:
        global_config   (dict)          : Dictionary containing user level global config parameters
        debug_detection (dict)          : Dictionary containing HVAC detection related debugging information
        quan_005        (np.ndarray)    : 0.5th percentile consumption at epoch level
        quan_995        (np.ndarray)    : 99.5th percentile consumption at epoch level
        hvac_params     (dict)          : Dictionary containing hvac algo related config parameters
        logger_base     (logger)        : Writes logs during code flow

    Returns:
        None
    """

    logger_local = logger_base.get("logger").getChild("plot_amplitude_detection")
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    file_name, figure_directory = manage_plot_location(global_config)

    _, axis_array = plt.subplots(2, 2, figsize=(20, 7))
    axis_flat = axis_array.flatten()
    width = (quan_995 - quan_005 - 100) / hvac_params['detection']['NUM_BINS']

    bin_centers = debug_detection['mid']['edges']
    mid_temp_hist_cooling, mid_temp_hist_heating = debug_detection['mid']['hist'], debug_detection['mid']['hist']
    if 'baseline' in debug_detection['cdd']:
        mid_temp_hist_cooling = debug_detection['cdd']['baseline']['hist']
    if 'baseline' in debug_detection['hdd']:
        mid_temp_hist_heating = debug_detection['hdd']['baseline']['hist']

    axis_flat[0].plot(bin_centers, mid_temp_hist_cooling, color='yellow')
    axis_flat[2].plot(bin_centers, mid_temp_hist_heating, color='yellow')

    try:
        axis_flat[2].plot(bin_centers, debug_detection['hdd']['hist'], color='red',
                          label='hdd : {}'.format(debug_detection['hdd']['setpoint']))
        axis_flat[2].legend(prop={'size': 10})
    except (ValueError, IndexError, KeyError):
        pass
    try:
        axis_flat[0].plot(bin_centers, debug_detection['cdd']['hist'], color='blue',
                          label='cdd : {}'.format(debug_detection['cdd']['setpoint']))
        axis_flat[0].legend(prop={'size': 10})
    except (ValueError, IndexError, KeyError):
        pass

    axis_flat[0].fill_between(bin_centers, mid_temp_hist_cooling, color='yellow', alpha=0.7)
    axis_flat[2].fill_between(bin_centers, mid_temp_hist_heating, color='yellow', alpha=0.7)
    try:
        axis_flat[2].fill_between(bin_centers, debug_detection['hdd']['hist'], color='orangered', alpha=0.7)
    except (ValueError, IndexError, KeyError):
        pass
    try:
        axis_flat[0].fill_between(bin_centers, debug_detection['cdd']['hist'], color='dodgerblue', alpha=0.7)
    except (ValueError, IndexError, KeyError):
        pass

    logger_hvac.info(' ------------------ Plotting : AC Modes --------------------- |')
    mode_info = debug_detection['cdd']['amplitude_cluster_info']

    if len(mode_info) >= 8:
        cluster_limits = mode_info['cluster_limits']
        center_labels = []
        if len(cluster_limits) == 2:
            center_labels = fill_center_labels(bin_centers, cluster_limits, center_labels)
    else:
        center_labels = np.zeros(len(bin_centers))

    color_map = {-1: 'lavender', 0: 'blue', 1: 'darkblue'}
    bar_color = [color_map[i] for i in center_labels]

    axis_flat[1].bar(bin_centers, debug_detection['cdd']['histDiff'], width, color=bar_color, alpha=0.7)
    axis_flat[1].set_title('Cooling \n mu : {}, sigma : {}'.format(np.around(debug_detection['cdd']['mu']),
                                                                   np.around(debug_detection['cdd']['sigma'])),
                           fontsize=15)

    logger_hvac.info(' ------------------ Plotting : SH Modes --------------------- |')
    mode_info = debug_detection['hdd']['amplitude_cluster_info']

    if len(mode_info) >= 8:
        cluster_limits = mode_info['cluster_limits']
        center_labels = []
        if len(cluster_limits) == 2:
            center_labels = fill_center_labels(bin_centers, cluster_limits, center_labels)
    else:
        center_labels = np.zeros(len(bin_centers))

    color_map = {-1: 'black', 0: 'orange', 1: 'red'}
    bar_color = [color_map[i] for i in center_labels]

    axis_flat[3].bar(bin_centers, debug_detection['hdd']['histDiff'], width, color=bar_color, alpha=0.7)
    axis_flat[3].set_title('Heating \n mu : {}, sigma : {}'.format(np.around(debug_detection['hdd']['mu']),
                                                                   np.around(debug_detection['hdd']['sigma'])),
                           fontsize=15)

    plt.tight_layout()
    if figure_directory:
        plt.savefig(file_name, dpi=250)
    plt.close()


def detect_hvac_amplitude(hvac_input_consumption, hvac_input_temperature, invalid_idx, hvac_params,
                          pre_pipeline_params, logger_base, global_config, hvac_exit_status):
    """
    Function detects hvac appliance amplitude and standard deviation and stores detection level parameters

    Parameters:
        hvac_input_consumption  (np.ndarray)       : Array of epoch level consumption flowing into hvac module
        hvac_input_temperature  (np.ndarray)       : Array of epoch level temperature flowing into hvac module
        invalid_idx             (np.ndarray)       : Array of invalid epochs based on consumption and temperature
        hvac_params             (dict)             : Dictionary containing hvac algo related initialized parameters
        pre_pipeline_params     (dict)             : Dictionary containing user consumption and temperature type related parameters
        logger_base             (logger)           : Writes logs during code flow
        global_config           (dict)             : Dictionary containing user profile related information
        hvac_exit_status        (dict)             : Dictionary containing hvac exit code and list of handled errors

    Returns:
        debug_detection         (dict)             : Dictionary containing hvac detection related debugging information
    """

    logger_local = logger_base.get("logger").getChild("detect_hvac_amplitude")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # initializing dictionary for storing hvac detection related key debugging information
    amplitude_info = {}
    amplitude_info['means'] = [np.inf, np.inf]
    amplitude_info['std'] = [0, 0]
    amplitude_info['cluster_overlap'] = False
    amplitude_info['cluster_limits'] = ((np.inf, np.inf), (np.inf, np.inf))
    amplitude_info['bin_centers'] = []
    amplitude_info['all_gaussians'] = []
    amplitude_info['number_of_modes'] = 0

    debug_detection = {
        'mid': {},
        'hdd': {'found': False, 'setpoint': False, 'mu': False, 'sigma': False,
                'hist': np.zeros(hvac_params['detection']['NUM_BINS']),
                'histDiff': np.zeros(hvac_params['detection']['NUM_BINS']), 'amplitude_cluster_info': amplitude_info,
                'baseline': {'hist': np.zeros(hvac_params['detection']['NUM_BINS']),
                             'mid_temp_idx': np.zeros(len(hvac_input_consumption))}},
        'cdd': {'found': False, 'setpoint': False, 'mu': False, 'sigma': False,
                'hist': np.zeros(hvac_params['detection']['NUM_BINS']),
                'histDiff': np.zeros(hvac_params['detection']['NUM_BINS']), 'amplitude_cluster_info': amplitude_info,
                'baseline': {'hist': np.zeros(hvac_params['detection']['NUM_BINS']),
                             'mid_temp_idx': np.zeros(len(hvac_input_consumption))}}
    }

    # getting qualified epoch level consumption points, excluding outliers. Followed by getting its histogram edges and centers.

    exist_outlier = True
    perc_99p5 = np.around(super_percentile(hvac_input_consumption, 99.5), 5)
    perc_99p9 = np.around(super_percentile(hvac_input_consumption, 99.9), 5)
    if perc_99p9 < perc_99p5 * 2:
        exist_outlier = False

    quan_005 = np.around(super_percentile(hvac_input_consumption, 0.5), 5)

    if exist_outlier:
        quan_995 = np.around(super_percentile(hvac_input_consumption, 99.5), 5)
    else:
        quan_995 = np.around(super_percentile(hvac_input_consumption, 99.9), 5)

    qualified_epoch_idx = np.logical_and(hvac_input_consumption >= quan_005, hvac_input_consumption <= quan_995)
    _, raw_hist_edges = np.histogram(hvac_input_consumption[qualified_epoch_idx], hvac_params['detection']['NUM_BINS'],
                                     density=True)
    raw_hist_centers = np.r_[0.5 * (raw_hist_edges[:-1] + raw_hist_edges[1:])]

    logger_hvac.info(' ------------------ Mid Temperature Range --------------------- |')
    # getting epoch points qualified for being in mid temperature range scope
    mid_temp_scope_idx = np.logical_and(~np.isnan(hvac_input_temperature),
                                        np.logical_and(hvac_input_temperature >=
                                                       hvac_params['detection']['MID_TEMPERATURE_RANGE'][0],
                                                       hvac_input_temperature <=
                                                       hvac_params['detection']['MID_TEMPERATURE_RANGE'][1]))
    logger_hvac.info(
        ' {} data points in mid temperature scope before taking quantiles |'.format(np.sum(mid_temp_scope_idx)))

    mid_temp_scope = hvac_input_temperature[mid_temp_scope_idx]
    low_limit_mid_range = mquantiles(mid_temp_scope, hvac_params['detection']['MID_TEMPERATURE_QUANTILE'][0],
                                     alphap=0.5, betap=0.5)
    high_limit_mid_range = mquantiles(mid_temp_scope, hvac_params['detection']['MID_TEMPERATURE_QUANTILE'][1],
                                      alphap=0.5, betap=0.5)
    logger_hvac.info('mid temperature range identified {} - {} |'.format(low_limit_mid_range, high_limit_mid_range))

    # getting mid temperature epochs, from low_limit_mid_range and high_limit_mid_range
    mid_temp_idx = (hvac_input_consumption > 0) & (hvac_input_temperature >= low_limit_mid_range) & \
                   (hvac_input_temperature <= high_limit_mid_range) & (~invalid_idx.astype(bool))
    if np.sum(mid_temp_scope_idx) == 0:
        mid_temp_idx = mid_temp_idx.filled(0)
    logger_hvac.info(
        ' >> After quantiles, {} data points in final mid temperature range {}F - {}F |'.format(np.sum(mid_temp_idx),
                                                                                                low_limit_mid_range,
                                                                                                high_limit_mid_range))

    # making mid temperature histogram - handling edge - standardizing - removing outlier bins
    bin_counts, _ = np.histogram(hvac_input_consumption[mid_temp_idx], bins=raw_hist_centers)
    bin_counts = np.r_[bin_counts, 0]
    bin_counts += 1
    hist_mid_temp = bin_counts.T / np.sum(bin_counts)
    hist_mid_temp[0: hvac_params['detection']['NUM_BINS_TO_REMOVE']] = 0

    # assigning important attributes to debug detection dictionary for debugging
    debug_detection['mid'] = {
        'temp': [low_limit_mid_range, high_limit_mid_range],
        'edges': raw_hist_centers, 'hist': hist_mid_temp,
        'mid_temp_idx': mid_temp_idx
    }
    debug_detection['cdd']['baseline']['hist'] = hist_mid_temp
    debug_detection['cdd']['baseline']['mid_temp_idx'] = mid_temp_idx
    debug_detection['hdd']['baseline']['hist'] = hist_mid_temp
    debug_detection['hdd']['baseline']['mid_temp_idx'] = mid_temp_idx

    logger_hvac.info(' mid temperature detection steps complete |')

    sh_params = {
        'invalid_idx': invalid_idx,
        'raw_hist_centers': raw_hist_centers,
        'found_wh': False,
        'mu_wh': np.nan,
        'sigma_wh': np.nan,
        'hist_mid_temp': hist_mid_temp,
        'mid_temp_idx': mid_temp_idx,
        'hvac_input_consumption': hvac_input_consumption
    }

    logger_hvac.info(' ------------------ SH Amplitude detection --------------------- |')
    detect_sh_amplitude(hvac_params, low_limit_mid_range, hvac_input_consumption, hvac_input_temperature, sh_params,
                        debug_detection, logger_pass, hvac_exit_status)

    baseline_params, hvac_params = get_ac_adjusted_detection_params(hvac_input_consumption, hvac_input_temperature,
                                                                    raw_hist_centers, pre_pipeline_params,
                                                                    debug_detection, hvac_params)

    low_limit_mid_range, high_limit_mid_range = debug_detection['mid']['temp'][0], debug_detection['mid']['temp'][1]

    ac_params = {
        'invalid_idx': invalid_idx,
        'raw_hist_centers': debug_detection['mid']['edges'],
        'found_wh': False,
        'mu_wh': np.nan,
        'sigma_wh': np.nan,
        'hist_mid_temp': baseline_params['epoch_level']['hist_mid_temp'],
        'mid_temp_idx': baseline_params['epoch_level']['mid_temp_idx'],
        'baseline_params': baseline_params,
        'pre_pipeline_params': pre_pipeline_params,
        'hvac_input_consumption': hvac_input_consumption
    }

    logger_hvac.info(' ------------------ AC Amplitude detection --------------------- |')
    detect_ac_amplitude(hvac_params, high_limit_mid_range, hvac_input_consumption, hvac_input_temperature, ac_params,
                        debug_detection, logger_pass, hvac_exit_status)
    debug_detection['cdd']['baseline_params'] = baseline_params

    ac_amplitude_info = debug_detection['cdd']['amplitude_cluster_info']
    debug_detection['cdd']['amplitude_cluster_info'] = adjust_ac_detection_range(ac_amplitude_info, raw_hist_centers,
                                                                                 hvac_input_consumption,
                                                                                 pre_pipeline_params,
                                                                                 hvac_params)

    generate_amplitude_plot = ('hvac' in global_config['generate_plots']) or ('all' in global_config['generate_plots'])
    if generate_amplitude_plot and (global_config['switch']['plot_level'] >= 3):
        plot_amplitude_detection(global_config, debug_detection, quan_005, quan_995, hvac_params, logger_pass)

    return debug_detection


def get_l1_distance_metric(dist_1, dist_2):
    """
    Get the weighted L1 distance between two arrays

    Parameters:
        dist_1      (np.ndarray)  : First Input array
        dist_2      (np.ndarray)  : Second Input array

    Returns:
        l1_distance (float)       : total weighted L1 distance / length of array
    """
    diff = dist_1 - dist_2
    diff[diff < 0] = 0
    n = len(diff) + 1
    diff = np.array([diff[i] * (i + 1) for i in range(len(diff))]) / n
    l1_distance = np.sum(diff)
    return l1_distance


def select_baseline_hist(original_dist, to_compare_dist, candidate_idx, validity):
    """
    Select the best mid-temperature/baseline histogram wrt hvac consumption histogram from the candidates based on L1 distance

    Parameters:
        original_dist               (np.ndarray)  : Cooling consumption reference histogram
        to_compare_dist             (list)        : List of all mid-temp histogram candidates
        candidate_idx               (list)        : List of candidate indices
        validity                    (list)        : List of boolean flags for validity of all mid-temp histogram candidates

    Returns:
        best_baseline_histogram     (np.ndarray) : selected mid-temperature/baseline histogram based on maximum L1 distance from reference
        best_candidate_index        (np.ndarray) : selected candidate index
    """
    distance_metric_values = [get_l1_distance_metric(original_dist, to_compare_dist[i]) for i in
                              range(len(to_compare_dist))]
    if validity[0] or (validity[1] == 0):
        max_difference_idx = 0
    else:
        max_difference_idx = 1

    for i in range(len(validity)):
        if validity[i] and distance_metric_values[i] > distance_metric_values[max_difference_idx]:
            max_difference_idx = i
    best_baseline_histogram = to_compare_dist[max_difference_idx]
    best_candidate_index = candidate_idx[max_difference_idx]
    return best_baseline_histogram, best_candidate_index


def detect_sh_amplitude(hvac_params, low_limit_mid_range, hvac_input_consumption, hvac_input_temperature, params_dict,
                        debug_detection, logger_base, hvac_exit_status):
    """
    Function detects sh amplitude and standard deviation and stores detection in debug object

    Parameters:
        hvac_params             (dict)           : Dictionary containing hvac algo related initialized parameters
        low_limit_mid_range     (float)          : Lower limit of mid temperature range
        hvac_input_consumption  (np.ndarray)     : Array of epoch level consumption flowing into hvac module
        hvac_input_temperature  (np.ndarray)     : Array of epoch level temperature flowing into hvac module
        params_dict             (dict)           : Dictionary containing HVAC related information
        debug_detection         (dict)           : Dictionary containing debug information from hvac detection stage
        logger_base             (logger)         : Writes logs during code flow
        hvac_exit_status        (dict)           : Dictionary containing hvac exit code and list of handled errors

    Returns:
        None
    """

    logger_local = logger_base.get("logger").getChild("detect_sh_amplitude")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    invalid_idx = params_dict['invalid_idx']
    raw_hist_centers = params_dict['raw_hist_centers']
    hist_mid_temp = params_dict['hist_mid_temp']

    for hdd_setpoint in hvac_params['detection']['SH']['SETPOINTS']:

        # sh setpoint should be less than permissible low limit of mid range temperature
        if hdd_setpoint < low_limit_mid_range:
            # getting sh valid epochs
            valid_idx = np.logical_and(hvac_input_consumption > 0,
                                       np.logical_and(hvac_input_temperature < hdd_setpoint, ~invalid_idx))

            # there should be minimum enough valid epochs for getting setpoint. To avoid false detection.
            if np.sum(valid_idx) >= hvac_params['detection']['MIN_POINTS']:
                logger_hvac.debug('sh detection setpoint is {}F. # of valid points are {} |'.format(hdd_setpoint,
                                                                                                    np.sum(valid_idx)))
                # creating standardized sh histogram
                bin_counts, _ = np.histogram(hvac_input_consumption[valid_idx], raw_hist_centers)
                bin_counts = np.r_[bin_counts, 0]
                hist_hdd = bin_counts / np.sum(bin_counts)
                logger_hvac.info(' scaled heating histogram created |')

                # removing low consumption bins, lurking appliances
                hist_hdd[0:hvac_params['detection']['NUM_BINS_TO_REMOVE']] = 0
                logger_hvac.info(' dropping first {} bins from heating histogram |'.format(
                    hvac_params['detection']['NUM_BINS_TO_REMOVE']))

                logger_hvac.info(' fitting residual gaussian to get sh mu and sigma |')

                # getting sh related mu and sigma from gaussian fitting
                logger_hvac.info(' fitting residual gaussian to get ac mu and sigma |')
                hdd_kl_divergence, mu_hdd, sigma_hdd, hist_diff_hdd, found_hdd, amplitude_info, hvac_exit_status = \
                    fit_optimum_gaussian(hist_hdd, hist_mid_temp, raw_hist_centers, hvac_params['detection']['SH'],
                                         'SH', params_dict, logger_pass, hvac_exit_status)

                logger_hvac.info('got sh mu and sigma |')
                # assigning important attributes to debug detection dictionary for debugging
                debug_detection['hdd'] = {
                    'setpoint': hdd_setpoint,
                    'hist': hist_hdd,
                    'histDiff': hist_diff_hdd,
                    'dist': hdd_kl_divergence,
                    'mu': mu_hdd,
                    'sigma': sigma_hdd,
                    'found': found_hdd,
                    'amplitude_cluster_info': amplitude_info
                }

                logger_hvac.debug('sh : found={}, mu={}, sigma={}, hist_diff={} |'.format(found_hdd, mu_hdd, sigma_hdd,
                                                                                          str(hist_diff_hdd).replace(
                                                                                              '\n', ' ')))
                break


def detect_ac_amplitude(hvac_params, high_limit_mid_range, hvac_input_consumption, hvac_input_temperature, params_dict,
                        debug_detection, logger_base, hvac_exit_status):
    """
    Function detects ac amplitude and standard deviation and stores detection in debug object

    Parameters:
        hvac_params             (dict)           : Dictionary containing hvac algo related initialized parameters
        high_limit_mid_range    (float)          : Higher limit of mid temperature range
        hvac_input_consumption  (np.ndarray)     : Array of epoch level consumption flowing into hvac module
        hvac_input_temperature  (np.ndarray)     : Array of epoch level temperature flowing into hvac module
        params_dict             (dict)           : Dictionary containing HVAC related information
        debug_detection         (dict)           : Dictionary containing debug information from hvac detection stage
        logger_base             (logger)         : Writes logs during code flow
        hvac_exit_status        (dict)           : Dictionary containing hvac exit code and list of handled errors

    Returns:
        None
    """

    logger_local = logger_base.get("logger").getChild("detect_ac_amplitude")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    invalid_idx = params_dict['invalid_idx']
    raw_hist_centers = params_dict['raw_hist_centers']
    hist_mid_temp = params_dict['hist_mid_temp']
    mid_temp_idx = params_dict['mid_temp_idx']
    baseline_params = params_dict['baseline_params']
    pre_pipeline_params = params_dict.get('pre_pipeline_params', {})
    adjust_midtemp_flag = pre_pipeline_params.get('all_flags', {}).get('adjust_midtemp_flag', False)
    exclude_detection_idx = pre_pipeline_params.get('hvac', {}).get('cooling', {}).get('exclude_detection_idx',
                                                                                       np.zeros(hvac_input_consumption.shape))

    for cdd_setpoint in hvac_params['detection']['AC']['SETPOINTS']:
        # ac setpoint should be greater than permissible high limit of mid range temperature
        if cdd_setpoint > high_limit_mid_range:
            # getting ac valid epochs
            valid_idx = np.logical_and(hvac_input_consumption > 0,
                                       np.logical_and(hvac_input_temperature > cdd_setpoint, ~invalid_idx))

            if pre_pipeline_params.get('all_flags', {}).get('is_night_ac', False) and \
                    np.sum(np.logical_and(valid_idx, ~exclude_detection_idx)) >= hvac_params['detection']['MIN_POINTS']:
                valid_idx = np.logical_and(valid_idx, ~exclude_detection_idx)

            # there should be minimum enough valid epochs for getting setpoint. to avoid false detection.
            if np.sum(valid_idx) >= hvac_params['detection']['MIN_POINTS']:
                logger_hvac.debug('ac detection setpoint is {}F. # of valid points are {} |'.format(cdd_setpoint,
                                                                                                    np.sum(valid_idx)))
                # creating standardized ac histogram
                bin_counts, _ = np.histogram(hvac_input_consumption[valid_idx], raw_hist_centers)
                bin_counts = np.r_[bin_counts, 0]
                hist_cdd = bin_counts / np.sum(bin_counts)
                logger_hvac.info(' scaled cooling histogram created |')

                # removing low consumption bins, lurking appliances
                hist_cdd[0:hvac_params['detection']['AC']['NUM_BINS_TO_REMOVE']] = 0
                logger_hvac.info(' dropping first {} bins from cooling histogram |'.format(
                    hvac_params['detection']['NUM_BINS_TO_REMOVE']))

                if adjust_midtemp_flag:
                    labels = ['epoch_level', 'relative_winter']
                    to_compare_dist = [baseline_params[label]['hist_mid_temp'] for label in labels]
                    candidate_idx = [baseline_params[label]['mid_temp_idx'] for label in labels]
                    validity = [baseline_params[label]['valid'] for label in labels]
                    hist_mid_temp, mid_temp_idx = select_baseline_hist(hist_cdd, to_compare_dist, candidate_idx,
                                                                       validity)

                logger_hvac.info(' fitting residual gaussian to get ac mu and sigma |')
                # getting ac related mu and sigma from gaussian fitting
                cdd_kl_divergence, mu_cdd, sigma_cdd, hist_diff_cdd, found_cdd, amplitude_info, hvac_exit_status = \
                    fit_optimum_gaussian(hist_cdd, hist_mid_temp, raw_hist_centers, hvac_params['detection']['AC'],
                                         'AC', params_dict, logger_pass, hvac_exit_status)

                logger_hvac.info('got ac mu and sigma |')
                # assigning important attributes to debug detection dictionary for debugging
                debug_detection['cdd'] = {
                    'setpoint': cdd_setpoint,
                    'hist': hist_cdd,
                    'histDiff': hist_diff_cdd,
                    'dist': cdd_kl_divergence,
                    'mu': mu_cdd,
                    'sigma': sigma_cdd,
                    'found': found_cdd,
                    'baseline': {
                        'hist': hist_mid_temp,
                        'mid_temp_idx': mid_temp_idx
                    },
                    'amplitude_cluster_info': amplitude_info
                }

                logger_hvac.debug('ac : found={}, mu={}, sigma={}, hist_diff={} |'.format(found_cdd, mu_cdd, sigma_cdd,
                                                                                          str(hist_diff_cdd).replace(
                                                                                              '\n', ' ')))
                break


def detect_wh_amplitude(hist_mid_temp, raw_hist_centers, hvac_params, logger_base, hvac_exit_status, debug_detection):
    """
    Function finds water-heater amplitude and standard deviation and returns them

    Parameters:
        hist_mid_temp       (np.ndarray)     : Array containing only mid temperature histogram distribution
        raw_hist_centers    (np.ndarray)     : Array containing overall energy's histogram distribution
        hvac_params         (dict)           : Dictionary containing hvac algo related initialized parameters
        logger_base         (logger)         : Writes logs during code flow
        hvac_exit_status    (dict)           : Dictionary containing hvac exit code and list of handled errors
        debug_detection     (dict)           : Dictionary containing debug information from hvac detection stage

    Returns:
        found_wh            (int)            : Flag for whether water-heater is found or not
        mu_wh               (int)            : amplitude of water-heater device, if found
        sigma_wh            (int)            : standard deviation of water-heater device, if found
        hvac_exit_status    (dict)           : Dictionary containing hvac exit code and list of handled errors
    """

    logger_local = logger_base.get("logger").getChild("detect_wh_amplitude")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    hist_mid_temp_min = copy.deepcopy(hist_mid_temp)

    for i in range(len(hist_mid_temp)):
        hist_mid_temp_min[i] = np.min(hist_mid_temp[0:i + 1])

    logger_hvac.info('fitting residual gaussian to get water-heater mu and sigma |')
    _, mu_wh, sigma_wh, _, found_wh, hvac_exit_status = fit_residual_gaussian(hist_mid_temp, hist_mid_temp_min,
                                                                              raw_hist_centers,
                                                                              hvac_params['detection']['WH'],
                                                                              'WaterHeating', logger_pass,
                                                                              hvac_exit_status)
    # checking whether appropriate water heater is detected
    found_wh = np.logical_and(found_wh, mu_wh < hvac_params['detection']['WH']['MAX_AMPLITUDE'])
    logger_hvac.info('got water-heater mu and sigma |')
    logger_hvac.info('water-heater : found={}, mu={}, sigma={} |'.format(found_wh, mu_wh, sigma_wh))

    if found_wh:
        logger_hvac.info('removing water-heater bins from mid range histogram |')
        # removing water-heater bins from mid range histogram
        hist_mid_temp[
            np.logical_and(raw_hist_centers >= mu_wh - hvac_params['detection']['WH']['MIN_DETECTION_STD'] * sigma_wh,
                           raw_hist_centers <= mu_wh + hvac_params['detection']['WH'][
                               'MIN_DETECTION_STD'] * sigma_wh)] = 0
        hist_mid_temp = hist_mid_temp / np.sum(hist_mid_temp)
        debug_detection['mid']['hist'] = hist_mid_temp
        logger_hvac.info('removed water-heater bins from mid range histogram |')

    return found_wh, mu_wh, sigma_wh, hvac_exit_status


