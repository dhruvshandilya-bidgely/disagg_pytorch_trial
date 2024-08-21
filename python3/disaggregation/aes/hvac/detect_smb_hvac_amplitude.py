"""
Author - Mirambika Sikdar
Date - 21/12/2023
Detection for smb hvac amplitude
"""

# Import python packages

import copy
import numpy as np
import logging
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats.mstats import mquantiles
from sklearn import mixture

from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile as super_percentile
from python3.disaggregation.aes.hvac.init_hourly_smb_hvac_params import hvac_static_params

hvac_static_params = hvac_static_params()


def manage_plot_location(global_config):
    '''
    Function to get filename of detection diagnostic plot. Over-writes plot if already exists at location.

    Parameters:
        global_config (dict)  : Dictionary containing  all user level global config parameters. Need uuid.

    Returns:
        file_name (str) : String name of the detection diagnostic plot
    '''

    figure_directory = hvac_static_params.get('path').get('hvac_plots')

    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    figure_directory = figure_directory + '/' + global_config['uuid']
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    file_name = figure_directory + '/amplitude_detection_' + global_config['uuid'] + '.png'

    if os.path.isfile(file_name):
        os.remove(file_name)

    return file_name, figure_directory


def fill_center_labels(bin_centers, cluster_limits, center_labels):
    """
    Function to assign mode id's to bins

    Parameters:
        bin_centers (np.ndarray)        : Array containing all the bin centers
        cluster_limits (np.ndarray)     : Array containing all the bin limits (AC/SH)
        center_labels (list)            : Initialized empty list of bin labels

    Returns:
        cluster_labels (list)           : Filled list of bin labels
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
        bin_centers (np.ndarray)        : Array containing all the bin centers
        mode_info (dict)                : Dictionary containing mode information about AC and SH
        debug_detection (dict)          : Dictionary containing hvac detection related debug information
        center labels (list)            : List containing labels for all bins
        axis_flat (np.ndarray)          : Array of axes
        appliance                       : Identifier of kind of HVAC appliance

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
        global_config (dict)        : Dictionary containing user level global config parameters
        debug_detection (dict)      : Dictionary containing HVAC detection related debugging information
        quan_005 (np.ndarray)            : 0.5th percentile consumption at epoch level
        quan_995 (np.ndarray)            : 99.5th percentile consumption at epoch level
        hvac_params (dict)          : Dictionary containing hvac algo related config parameters
        logger_base (logger)        : Writes logs during code flow

    Returns:
        None
    """

    logger_local = logger_base.get("logger").getChild("plot_amplitude_detection")
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    file_name, figure_directory = manage_plot_location(global_config)

    _, axis_array = plt.subplots(1, 3, figsize=(20, 7))
    axis_flat = axis_array.flatten()
    width = (quan_995 - quan_005 - 100) / hvac_params['detection']['NUM_BINS']

    bin_centers = debug_detection['mid']['edges']
    axis_flat[0].plot(bin_centers, debug_detection['mid']['hist'], color='yellow',
                      label='Mid Temp Range : {} - {}'.format(np.around(debug_detection['mid']['temp'][0][0], 1),
                                                              np.around(debug_detection['mid']['temp'][1][0], 1)))
    try:
        axis_flat[0].plot(bin_centers, debug_detection['hdd']['hist'], color='red',
                          label='hdd : {}'.format(debug_detection['hdd']['setpoint']))
    except (ValueError, IndexError, KeyError):
        pass
    try:
        axis_flat[0].plot(bin_centers, debug_detection['cdd']['hist'], color='blue',
                          label='cdd : {}'.format(debug_detection['cdd']['setpoint']))
    except (ValueError, IndexError, KeyError):
        pass
    axis_flat[0].legend(prop={'size': 10})
    axis_flat[0].fill_between(bin_centers, debug_detection['mid']['hist'], color='yellow', alpha='0.7')
    try:
        axis_flat[0].fill_between(bin_centers, debug_detection['hdd']['hist'], color='orangered', alpha='0.7')
    except (ValueError, IndexError, KeyError):
        pass
    try:
        axis_flat[0].fill_between(bin_centers, debug_detection['cdd']['hist'], color='dodgerblue', alpha='0.7')
    except (ValueError, IndexError, KeyError):
        pass

    logger_hvac.info(' ------------------ Plotting : AC Modes --------------------- |')
    cooling_mode_info = debug_detection['cdd']['amplitude_cluster_info']

    if len(cooling_mode_info) == 8:
        cooling_cluster_limits = cooling_mode_info['cluster_limits']
        cooling_center_labels = []
        if len(cooling_cluster_limits) == 2:
            cooling_center_labels = fill_center_labels(bin_centers, cooling_cluster_limits, cooling_center_labels)
    else:
        cooling_center_labels = np.zeros(len(bin_centers))

    cooling_color_map = {-1: 'lavender', 0: 'blue', 1: 'darkblue'}
    cooling_bar_color = [cooling_color_map[i] for i in cooling_center_labels]

    axis_flat[1].bar(bin_centers, debug_detection['cdd']['histDiff'], width, color=cooling_bar_color, alpha=0.7)
    axis_flat[1].set_title('Cooling \n mu : {}, sigma : {}'.format(np.around(debug_detection['cdd']['mu']),
                                                                   np.around(debug_detection['cdd']['sigma'])),
                           fontsize=15)
    # Plotting of multiple Gaussians
    beautify = False
    if beautify:
        beautify_detection_plot(bin_centers, cooling_mode_info, debug_detection, cooling_center_labels, axis_flat, 'AC')

    logger_hvac.info(' ------------------ Plotting : SH Modes --------------------- |')
    heating_mode_info = debug_detection['hdd']['amplitude_cluster_info']

    if len(heating_mode_info) == 8:
        heating_cluster_limits = heating_mode_info['cluster_limits']
        heating_center_labels = []
        if len(heating_cluster_limits) == 2:
            heating_center_labels = fill_center_labels(bin_centers, heating_cluster_limits, heating_center_labels)
    else:
        heating_center_labels = np.zeros(len(bin_centers))

    heating_color_map = {-1: 'lavender', 0: 'red', 1: 'orangered'}
    heating_bar_color = [heating_color_map[i] for i in heating_center_labels]

    axis_flat[2].bar(bin_centers, debug_detection['hdd']['histDiff'], width, color=heating_bar_color, alpha=0.7)
    axis_flat[2].set_title('Heating \n mu : {}, sigma : {}'.format(np.around(debug_detection['hdd']['mu']),
                                                                   np.around(debug_detection['hdd']['sigma'])),
                           fontsize=15)
    # Plotting of multiple Gaussians
    if beautify:
        beautify_detection_plot(bin_centers, heating_mode_info, debug_detection, heating_center_labels, axis_flat, 'SH')

    plt.tight_layout()
    if figure_directory:
        plt.savefig(file_name, dpi=250)
    plt.close()


def variance(bin_centers, hist_diff):
    """
    Function calculates variance for a histogram distribution based on histogram bar weights

    Parameters:
        bin_centers (np.ndarray)       : Array of histogram centers from which variance has to be calculated
        hist_diff (np.ndarray)         : Array of bar weights for each of histogram bins

    Returns:
        variance (np.ndarray)          : Variance of histogram
    """

    hist_diff = hist_diff / np.sum(hist_diff)
    x1 = np.sum(hist_diff * bin_centers)
    variance = np.sum(hist_diff * (bin_centers - x1) ** 2)

    return variance


def detect_hvac_amplitude(hvac_input_consumption, hvac_input_temperature, invalid_idx, hvac_params,
                          logger_base, global_config, hvac_exit_status):
    """
    Function detects hvac appliance amplitude and standard deviation and stores detection level parameters

    Parameters:
        hvac_input_consumption (np.ndarray)       : Array of epoch level consumption flowing into hvac module
        hvac_input_temperature (np.ndarray)       : Array of epoch level temperature flowing into hvac module
        invalid_idx (np.ndarray)                  : Array of invalid epochs based on consumption and temperature
        hvac_params (dict)                        : Dictionary containing hvac algo related initialized parameters
        logger_base (logger)                      : Writes logs during code flow
        global_config (dict)                      : Dictionary containing user profile related information
        hvac_exit_status(dict)                    : Dictionary containing hvac exit code and list of handled errors

    Returns:
        debug_detection (dict)                    : Dictionary containing hvac detection related debugging information
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
        'hdd': {'found': False, 'setpoint': False, 'mu': False, 'sigma': False, 'hist': np.zeros(30),
                'histDiff': np.zeros(30), 'amplitude_cluster_info': amplitude_info},
        'cdd': {'found': False, 'setpoint': False, 'mu': False, 'sigma': False, 'hist': np.zeros(30),
                'histDiff': np.zeros(30), 'amplitude_cluster_info': amplitude_info}
    }

    # getting qualified epoch level consumption points, excluding outliers. Followed by getting its histogram edges and centers.
    quan_005 = np.around(super_percentile(hvac_input_consumption, 0.5), 5)
    quan_995 = np.around(super_percentile(hvac_input_consumption, 99.5), 5)
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
        'edges': raw_hist_centers, 'hist': hist_mid_temp
    }
    logger_hvac.info(' mid temperature detection steps complete |')

    sh_ac_params = {
        'invalid_idx': invalid_idx,
        'raw_hist_centers': raw_hist_centers,
        'found_wh': False,
        'mu_wh': np.nan,
        'sigma_wh': np.nan,
        'hist_mid_temp': hist_mid_temp,
    }

    logger_hvac.info(' ------------------ SH Amplitude detection --------------------- |')
    detect_sh_amplitude(hvac_params, low_limit_mid_range, hvac_input_consumption, hvac_input_temperature, sh_ac_params,
                        debug_detection, logger_pass, hvac_exit_status)

    logger_hvac.info(' ------------------ AC Amplitude detection --------------------- |')
    detect_ac_amplitude(hvac_params, high_limit_mid_range, hvac_input_consumption, hvac_input_temperature, sh_ac_params,
                        debug_detection, logger_pass, hvac_exit_status)

    generate_amplitude_plot = ('hvac' in global_config['generate_plots']) or ('all' in global_config['generate_plots'])
    if generate_amplitude_plot and (global_config['switch']['plot_level'] >= 3):
        plot_amplitude_detection(global_config, debug_detection, quan_005, quan_995, hvac_params, logger_pass)

    return debug_detection


def detect_sh_amplitude(hvac_params, low_limit_mid_range, hvac_input_consumption, hvac_input_temperature, params_dict,
                        debug_detection, logger_base, hvac_exit_status):
    """
    Function detects sh amplitude and standard deviation and stores detection in debug object

    Parameters:
        hvac_params (dict)                         : Dictionary containing hvac algo related initialized parameters
        low_limit_mid_range (float)                : Lower limit of mid temperature range
        hvac_input_consumption (np.ndarray)        : Array of epoch level consumption flowing into hvac module
        hvac_input_temperature (np.ndarray)        : Array of epoch level temperature flowing into hvac module
        params_dict(dict)                          : Dictionary containing water-heater related information
        debug_detection (dict)                     : Dictionary containing debug information from hvac detection stage
        logger_base (logger)                       : Writes logs during code flow
        hvac_exit_status(dict)                     : Dictionary containing hvac exit code and list of handled errors

    Returns:
        debug_detection (dict)                     : Dictionary containing SH detection related debugging information
    """

    logger_local = logger_base.get("logger").getChild("sh_amplitude")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    invalid_idx = params_dict['invalid_idx']
    raw_hist_centers = params_dict['raw_hist_centers']
    found_wh = params_dict['found_wh']
    mu_wh = params_dict['mu_wh']
    sigma_wh = params_dict['sigma_wh']
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

                # removing water heater bins from sh histogram to estimate sh amplitude accurately
                if found_wh:
                    hist_hdd[np.logical_and(raw_hist_centers >=
                                            mu_wh - hvac_params['detection']['WH']['MIN_DETECTION_STD'] * sigma_wh,
                                            raw_hist_centers <=
                                            mu_wh + hvac_params['detection']['WH']['MIN_DETECTION_STD'] * sigma_wh)] = 0

                    logger_hvac.info(' removed water-heater bins from heating histogram |')
                    # again standardizing sh histogram after wh removal
                    hist_hdd = hist_hdd / np.sum(hist_hdd)
                    logger_hvac.info(' re-scaled heating histogram created |')

                # removing low consumption bins, lurking appliances
                hist_hdd[0:hvac_params['detection']['NUM_BINS_TO_REMOVE']] = 0
                logger_hvac.info(' dropping first {} bins from heating histogram |'.format(
                    hvac_params['detection']['NUM_BINS_TO_REMOVE']))

                logger_hvac.info(' fitting residual gaussian to get sh mu and sigma |')

                # getting sh related mu and sigma from gaussian fitting
                hdd_kl_divergence, mu_hdd, sigma_hdd, hist_diff_hdd, found_hdd, amplitude_info, hvac_exit_status = \
                    fit_optimum_gaussian(hist_hdd, hist_mid_temp, raw_hist_centers, hvac_params['detection']['SH'],
                                         'SH', logger_pass, hvac_exit_status)

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

        hvac_params (dict)                          : Dictionary containing hvac algo related initialized parameters
        high_limit_mid_range (float)                : Higher limit of mid temperature range
        hvac_input_consumption (np.ndarray)         : Array of epoch level consumption flowing into hvac module
        hvac_input_temperature (np.ndarray)         : Array of epoch level temperature flowing into hvac module
        params_dict(dict)                           : Dictionary containing water-heater related information
        debug_detection (dict)                      : Dictionary containing debug information from hvac detection stage
        logger_base (logger)                        : Writes logs during code flow
        hvac_exit_status(dict)                      : Dictionary containing hvac exit code and list of handled errors

    Returns:
        debug_detection (dict)                      : Dictionary containing AC detection related debugging information
    """

    logger_local = logger_base.get("logger").getChild("ac_amplitude")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    invalid_idx = params_dict['invalid_idx']
    raw_hist_centers = params_dict['raw_hist_centers']
    found_wh = params_dict['found_wh']
    mu_wh = params_dict['mu_wh']
    sigma_wh = params_dict['sigma_wh']
    hist_mid_temp = params_dict['hist_mid_temp']

    for cdd_setpoint in hvac_params['detection']['AC']['SETPOINTS']:
        # ac setpoint should be greater than permissible high limit of mid range temperature
        if cdd_setpoint > high_limit_mid_range:
            # getting ac valid epochs
            valid_idx = np.logical_and(hvac_input_consumption > 0,
                                       np.logical_and(hvac_input_temperature > cdd_setpoint, ~invalid_idx))

            # there should be minimum enough valid epochs for getting setpoint. to avoid false detection.
            if np.sum(valid_idx) >= hvac_params['detection']['MIN_POINTS']:
                logger_hvac.debug('ac detection setpoint is {}F. # of valid points are {} |'.format(cdd_setpoint,
                                                                                                    np.sum(valid_idx)))
                # creating standardized ac histogram
                bin_counts, _ = np.histogram(hvac_input_consumption[valid_idx], raw_hist_centers)
                bin_counts = np.r_[bin_counts, 0]
                hist_cdd = bin_counts / np.sum(bin_counts)
                logger_hvac.info(' scaled cooling histogram created |')

                # removing water heater bins from sh histogram to estimate sh amplitude accurately
                if found_wh:
                    hist_cdd[np.logical_and(raw_hist_centers >=
                                            mu_wh - hvac_params['detection']['WH']['MIN_DETECTION_STD'] * sigma_wh,
                                            raw_hist_centers <=
                                            mu_wh + hvac_params['detection']['WH']['MIN_DETECTION_STD'] * sigma_wh)] = 0

                    logger_hvac.info(' removed water-heater bins from cooling histogram |')
                    # again standardizing ac histogram after wh removal
                    hist_cdd = hist_cdd / np.sum(hist_cdd)
                    logger_hvac.info(' re-scaled cooling histogram created |')

                # removing low consumption bins, lurking appliances
                hist_cdd[0:hvac_params['detection']['NUM_BINS_TO_REMOVE']] = 0
                logger_hvac.info(' dropping first {} bins from cooling histogram |'.format(
                    hvac_params['detection']['NUM_BINS_TO_REMOVE']))

                logger_hvac.info(' fitting residual gaussian to get ac mu and sigma |')
                # getting ac related mu and sigma from gaussian fitting
                cdd_kl_divergence, mu_cdd, sigma_cdd, hist_diff_cdd, found_cdd, amplitude_info, hvac_exit_status = \
                    fit_optimum_gaussian(hist_cdd, hist_mid_temp, raw_hist_centers, hvac_params['detection']['AC'],
                                         'AC', logger_pass, hvac_exit_status)

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

        hist_mid_temp (np.ndarray)                  : Array containing only mid temperature histogram distribution
        raw_hist_centers (np.ndarray)               : Array containing overall energy's histogram distribution
        hvac_params (dict)                          : Dictionary containing hvac algo related initialized parameters
        logger_base (logger)                        : Writes logs during code flow
        hvac_exit_status(dict)                      : Dictionary containing hvac exit code and list of handled errors
        debug_detection (dict)                      : Dictionary containing debug information from hvac detection stage

    Returns:

        found_wh (int)                              : Flag for whether water-heater is found or not
        mu_wh (int)                                 : amplitude of water-heater device, if found
        sigma_wh (int)                              : standard deviation of water-heater device, if found
        hvac_exit_status(dict)                      : Dictionary containing hvac exit code and list of handled errors
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


def get_kl_divergence(first_distribution, second_distribution):
    """
    Function measures the difference between Two normal distributions

    Parameters:
        first_distribution      :  First Normal distribution
        second_distribution     :  Second Normal Distribution

    Return:
        kl_divergence           : Measure of difference between between two distributions
    """

    temp = first_distribution * np.log(first_distribution / second_distribution)
    # resolving the case when P(i) or Q(i)==0
    temp[np.logical_or(np.isnan(temp), np.isinf(temp))] = 0
    kl_divergence = np.sum(temp)

    return kl_divergence


def get_amplitude_cluster(model):
    """
    Function to fill dictionary containing information number of modes, means and standard deviations (AC/SH)

    Parameters:
        model (list) : List containing all the gaussian models tried while finding best number of estimators

    Returns:
        amplitude_info (dict) : Dictionary containing all mode related information (mu, sigma, mode limits)
    """

    cluster_mu = np.squeeze(model.means_)
    cluster_sigma = np.sqrt(np.squeeze(model.covariances_))
    smaller_mu_idx, larger_mu_idx = np.argsort(cluster_mu)

    small_mu = cluster_mu[smaller_mu_idx]
    large_mu = cluster_mu[larger_mu_idx]
    small_sigma = cluster_sigma[smaller_mu_idx]
    large_sigma = cluster_sigma[larger_mu_idx]

    arm_factor = hvac_static_params['arm_3_std']

    arm_of_small_mu = arm_factor * small_sigma
    arm_of_large_mu = arm_factor * large_sigma

    clusters_overlap = (large_mu - small_mu) < (arm_of_small_mu + arm_of_large_mu)

    if clusters_overlap:
        arm_factor = (large_mu - small_mu) / (small_sigma + large_sigma)

    small_mode_limits = (small_mu - hvac_static_params['arm_3_std'] * small_sigma, small_mu + arm_factor * small_sigma)

    large_mode_limits = (large_mu - arm_factor * large_sigma, large_mu + hvac_static_params['arm_3_std'] * large_sigma)

    amplitude_info = {'means': [np.around(small_mu), np.around(large_mu)],
                      'std': [np.around(small_sigma), np.around(large_sigma)],
                      'cluster_overlap': clusters_overlap,
                      'cluster_limits': [np.around(small_mode_limits), np.around(large_mode_limits)]
                      }

    return amplitude_info


def get_num_gauss_components(filled_gauss_data):
    """
    Function to determine optimum number of modes for AC/SH.

    Parameters:
        filled_gauss_data (np.ndarray) : Array over which fitting process of gaussian model is tried (AC/SH)

    Returns:
        n_components : Optimum number of modes found
        clfs  The list containing all of gaussian models tried
    """

    gaussian_data = filled_gauss_data.T

    # gaussian models for number of estimators, and their corresponding AICs and BICs
    n_estimators = np.arange(1, 4)
    clfs = [mixture.GaussianMixture(n).fit(gaussian_data) for n in n_estimators]
    bics = [clf.bic(gaussian_data) for clf in clfs]
    aics = [clf.aic(gaussian_data) for clf in clfs]

    # if required to plot : plt.plot(n_estimators, bics, label='BIC'), plt.plot(n_estimators, aics, label='AIC')
    # second deribatives are calculated to get optimum number of modes
    aic_2nd_derivative = [0] * len(aics)
    bic_2nd_derivative = [0] * len(bics)
    for i in range(len(aics) - 1):
        aic_2nd_derivative[i] = aics[i + 1] + aics[i - 1] - 2 * aics[i]
        bic_2nd_derivative[i] = bics[i + 1] + bics[i - 1] - 2 * bics[i]

    aic_n_components = aic_2nd_derivative.index(np.max(aic_2nd_derivative)) + 1
    bic_n_components = bic_2nd_derivative.index(np.max(bic_2nd_derivative)) + 1

    if aic_n_components == 2:

        clf = clfs[1]
        sorted_model = np.argsort(np.squeeze(clf.means_))
        mu_small = clf.means_[sorted_model[0], 0]
        sigma_small = np.sqrt(clf.covariances_[sorted_model[0]])[0, 0]
        mu_large = clf.means_[sorted_model[1], 0]
        sigma_large = np.sqrt(clf.covariances_[sorted_model[1]])[0, 0]

        if (mu_small + sigma_small) > (mu_large - sigma_large):
            aic_n_components = 1

    elif aic_n_components >= 3:
        aic_n_components = 2

    n_components = min(aic_n_components, bic_n_components)

    return n_components, clfs


def fit_residual_gaussian(hist_degree_day, hist_mid_temp, raw_hist_centers, hvac_params_app, app_type, logger_base,
                          hvac_exit_status):
    """
    Function fits a gaussian mixture model to calculate mean and standard deviation of calculated app amplitudes

   Parameters:

       hist_degree_day (np.ndarray)               : Histogram of consumptions in cdd or hdd scope
       hist_mid_temp (np.ndarray)                 : Histogram of consumptions, in mid temperature range scope
       raw_hist_centers (np.ndarray)              : Array of all raw consumption histogram centers
       hvac_params_app (dict)                     : Dictionary containing appliance detection related parameters
       app_type (string)                          : String containing name of appliance type (SH, AC or WH)
       logger_base (logger)                       : Writes logs during code flow
       hvac_exit_status(dict)                     : Dictionary containing hvac exit code and list of handled errors

   Returns:

       kl_divergence (float)                      : KL Divergence (separation b/w app and mid temp distributions)
       mu (float)                                 : Detected appliance mean
       sigma (float)                              : Detected standard deviation in measurement of app mean amplitude
       hist_diff (np.ndarray)                     : Contains difference of standardized app and mid temp histograms
       found (bool)                               : Indicator of whether applicance is found or not
       hvac_exit_status(dict)                     : Dictionary containing hvac exit code and list of handled errors
   """

    logger_local = logger_base.get("logger").getChild("fit_residual_gaussian")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # raw consumption histogram related features
    bin_centers = raw_hist_centers
    bin_size = np.median(raw_hist_centers[1:] - raw_hist_centers[:-1])
    # finding the extent of difference between mid temperature histogram and histogram of sh and ac
    kl_divergence = get_kl_divergence(hist_degree_day, hist_mid_temp)
    logger_hvac.info('unused kl-divergence is {} |'.format(kl_divergence))

    # hist_diff: measure of difference between mid temperature histogram and histogram of sh and ac
    hist_diff = hist_degree_day - hist_mid_temp
    # ensuring negative difference is not there
    hist_diff[hist_diff < 0] = 0

    if np.sum(hist_diff) == 0:
        logger_hvac.info('no difference in {} histogram vs mid temp histogram. mu=0 sigma=0 |'.format(app_type))
        # if no difference in mid temp and appliance histogram, mu=0, sigma=0 and found=0, to avoid crash hvac algo
        mu = 0
        sigma = 0
        found = False
        return kl_divergence, mu, sigma, hist_diff, found, hvac_exit_status

    # diffprob: a measure of the probability of finding a sh or ac appliance
    diff_prob = np.sum(hist_diff)
    logger_hvac.debug('>> {} diff prob is {} |'.format(app_type, diff_prob))

    # standardizing measure of diference between mid temperature histogram and histogram of sh and ac
    hist_diff = hist_diff / diff_prob
    # first estimate of mu is weighted average of bin centers and bin counts
    mu = np.sum(hist_diff * bin_centers)
    # getting estimate of variance in measurement of appliance amplitude (mu)
    var = variance(bin_centers, hist_diff)
    # getting standard deviation from variance
    sigma = np.sqrt(var)
    logger_hvac.debug('>> {} preliminary mu:{} and sigma:{} |'.format(app_type, mu, sigma))

    # array to keep probables of ac or sh amplitudes
    mu_zzz = np.zeros(shape=(3, 1))
    # array to keep probables of ac or sh standard deviation in measurement of amplitude
    sigma_zzz = np.zeros(shape=(3, 1))
    # array to keep component proportion of each amplitudes
    comp_proportion_zzz = np.zeros(shape=(3, 1))
    # assigning first entry to amplitude probables
    mu_zzz[0] = mu
    # assigning first entry to sigma probables
    sigma_zzz[0] = sigma

    # condition for possibility to multi-modal distribution, or minimum permissible amplitude is too small
    sigma_arm = hvac_static_params['gaussian_related']['sigma_arm']

    if mu - sigma_arm * sigma <= hvac_params_app['MIN_AMPLITUDE']:

        logger_hvac.info('2-mode gmm will be fit for {} |'.format(app_type))
        num_points = hvac_static_params['gaussian_related']['points_to_fill']
        filled_gauss_data = np.empty(shape=(1, 0))

        # filling distribution with filler data, for enabling better gaussian fit
        for i in range(len(bin_centers)):

            if hist_diff[i] > 0:
                n = np.ceil(hist_diff[i] * num_points)
                np.random.seed(12345)
                r1 = np.random.rand(1, int(n)) * bin_size + raw_hist_centers[i]
                filled_gauss_data = np.concatenate((filled_gauss_data, r1), axis=1)

        # initializing gaussian mixture model to data
        model_init = np.full(shape=(2, 1, 1), fill_value=0.1)
        model_init = np.linalg.inv(model_init)
        model = mixture.GaussianMixture(n_components=2, covariance_type='full',
                                        tol=hvac_static_params['gaussian_related']['tolerance'],
                                        max_iter=120, random_state=1,
                                        means_init=np.array(
                                            [np.min(filled_gauss_data), np.max(filled_gauss_data)]).reshape((2, 1)),
                                        weights_init=np.array([0.5, 0.5]),
                                        precisions_init=model_init)

        # fitting gaussian model to made up data
        model.fit(filled_gauss_data.T)

        logger_hvac.info('gmm model fitted now finding mu and sigma for {} |'.format(app_type))

        mu, sigma = find_mu_sigma(model, mu_zzz, sigma_zzz, comp_proportion_zzz, hvac_params_app, diff_prob,
                                  logger_pass)
        logger_hvac.info('gmm model sorting gave mu:{} and sigma:{} for {} |'.format(mu, sigma, app_type))

    logger_hvac.info('checking if {} is found |'.format(app_type))

    # determining if a valid hvac appliance is found
    found = hvac_params_app['getFound'](mu, sigma, diff_prob, kl_divergence, hvac_params_app) and \
            (mu - hvac_params_app['MIN_DETECTION_STD'] * sigma > hvac_params_app['MIN_AMPLITUDE'])
    logger_hvac.info('>> {} found is {} |'.format(app_type, found))

    return kl_divergence, mu, sigma, hist_diff, found, hvac_exit_status


def fit_optimum_gaussian(hist_degree_day, hist_mid_temp, raw_hist_centers, hvac_params_app, app_type, logger_base,
                         hvac_exit_status):
    """
    Function fits a gaussian mixture model to calculate mean and standard deviation of calculated app amplitudes

   Parameters:
       hist_degree_day (np.ndarray)               : Histogram of consumptions in cdd or hdd scope
       hist_mid_temp (np.ndarray)                 : Histogram of consumptions, in mid temperature range scope
       raw_hist_centers (np.ndarray)              : Array of all raw consumption histogram centers
       hvac_params_app (dict)                     : Dictionary containing appliance detection related parameters
       app_type (string)                          : String containing name of appliance type (SH, AC or WH)
       logger_base (logger)                       : Writes logs during code flow
       hvac_exit_status(dict)                     : Dictionary containing hvac exit code and list of handled errors

   Returns:
       kl_divergence (float)                      : KL Divergence (separation b/w app and mid temp distributions)
       mu (float)                                 : Detected appliance mean
       sigma (float)                              : Detected standard deviation in measurement of app mean amplitude
       hist_diff (np.ndarray)                     : Contains difference of standardized app and mid temp histograms
       found (bool)                               : Indicator of whether applicance is found or not
       amplitude_info (dict)                      : Dictionary containing multi-mode information. Mu sigma etc
       hvac_exit_status(dict)                     : Dictionary containing hvac exit code and list of handled errors

   """

    logger_local = logger_base.get("logger").getChild("fit_residual_gaussian")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # raw consumption histogram related features
    bin_centers = raw_hist_centers
    bin_size = np.median(raw_hist_centers[1:] - raw_hist_centers[:-1])
    # finding the extent of difference between mid temperature histogram and histogram of sh and ac
    kl_divergence = get_kl_divergence(hist_degree_day, hist_mid_temp)
    logger_hvac.info('unused kl-divergence is {} |'.format(kl_divergence))

    # hist_diff: measure of difference between mid temperature histogram and histogram of sh and ac
    hist_diff = hist_degree_day - hist_mid_temp
    # ensuring negative difference is not there
    hist_diff[hist_diff < 0] = 0

    if np.nansum(hist_diff) == 0:
        logger_hvac.info('no difference in {} histogram vs mid temp histogram. mu=0 sigma=0 |'.format(app_type))
        # if no difference in mid temp and appliance histogram, mu=0, sigma=0 and found=0, to avoid crash hvac algo
        mu = 0
        sigma = 0
        found = False

        amplitude_info = {}
        amplitude_info['means'] = [np.inf, np.inf]
        amplitude_info['std'] = [0, 0]
        amplitude_info['cluster_overlap'] = False
        amplitude_info['cluster_limits'] = ((np.inf, np.inf), (np.inf, np.inf))
        amplitude_info['bin_centers'] = []
        amplitude_info['all_gaussians'] = []
        amplitude_info['number_of_modes'] = 0

        return kl_divergence, mu, sigma, hist_diff, found, amplitude_info, hvac_exit_status

    # diffprob: a measure of the probability of finding a sh or ac appliance
    diff_prob = np.sum(hist_diff)
    logger_hvac.debug('>> {} diff prob is {} |'.format(app_type, diff_prob))

    # standardizing measure of diference between mid temperature histogram and histogram of sh and ac
    hist_diff = hist_diff / diff_prob
    # first estimate of mu is weighted average of bin centers and bin counts
    mu = np.sum(hist_diff * bin_centers)
    # getting estimate of variance in measurement of appliance amplitude (mu)
    var = variance(bin_centers, hist_diff)
    # getting standard deviation from variance
    sigma = np.sqrt(var)
    logger_hvac.debug('>> {} preliminary mu:{} and sigma:{} |'.format(app_type, mu, sigma))

    mu_probabbles = np.zeros(shape=(3, 1))
    sigma_probables = np.zeros(shape=(3, 1))
    proportions = np.zeros(shape=(3, 1))
    mu_probabbles[0] = mu
    sigma_probables[0] = sigma

    num_points = hvac_static_params['gaussian_related']['points_to_fill']
    filled_gauss_data = np.empty(shape=(1, 0))

    # filling distribution with made up data, for enabling better gaussian fit
    for i in range(len(bin_centers)):

        if hist_diff[i] > 0:
            n = np.ceil(hist_diff[i] * num_points)
            np.random.seed(12345)
            r1 = np.random.rand(1, int(n)) * bin_size + raw_hist_centers[i]
            filled_gauss_data = np.concatenate((filled_gauss_data, r1), axis=1)

    n_components, models = get_num_gauss_components(filled_gauss_data)

    # Plotting multi mode artifacts. This block can be commented
    # plt.bar(bin_centers, hist_diff, width=20)
    # xpdf = np.linspace(np.min(filled_gauss_data), np.max(filled_gauss_data), 1000)
    # clf = models[n_components - 1]
    # for i in range(clf.n_components):
    #     pdf = clf.weights_[i] * stats.norm(clf.means_[i, 0], np.sqrt(clf.covariances_[i, 0])).pdf(xpdf)
    #     plt.plot(xpdf, pdf, color='yellow')
    #     plt.fill_between(xpdf, pdf, alpha=0.25, color='yellow')

    if n_components == 1:

        model = models[n_components - 1]
        mu = np.squeeze(model.means_)
        sigma = np.sqrt(np.squeeze(model.covariances_))
        amplitude_info = {'means': [np.around(mu), np.Inf],
                          'std': [np.around(sigma), 0],
                          'cluster_overlap': False,
                          'cluster_limits': (
                              (np.around(mu - 2 * sigma), np.around(mu + 3 * sigma)), (np.Inf, np.Inf))
                          }
    elif n_components >= 2:

        model = models[n_components - 1]
        amplitude_info = get_amplitude_cluster(model)
        mu, sigma = find_mu_sigma(model, mu_probabbles, sigma_probables, proportions, hvac_params_app, diff_prob,
                                  logger_pass)

    logger_hvac.info('checking if {} is found |'.format(app_type))
    found = hvac_params_app['getFound'](mu, diff_prob) and \
            (mu - hvac_params_app['MIN_DETECTION_STD'] * sigma > hvac_params_app['MIN_AMPLITUDE'])
    logger_hvac.info('>> {} found is {} |'.format(app_type, found))

    amplitude_info['bin_centers'] = bin_centers
    amplitude_info['all_gaussians'] = models
    amplitude_info['number_of_modes'] = n_components
    amplitude_info['mode_idx_for_plotting'] = np.argsort(np.squeeze(model.means_))

    return kl_divergence, mu, sigma, hist_diff, found, amplitude_info, hvac_exit_status


def find_mu_sigma(model, mu_zzz, sigma_zzz, comp_proportion_zzz, hvac_params, diff_prob, logger_base):
    """
    Function selects the best mean and standard deviation of hvac appliance, from a set of probables

        Parameters:
            model (object)                   : Contains gaussian mixture model related attributes
            mu_zzz (np.ndarray)             : Array of appliance detected amplitude probables
            sigma_zzz (np.ndarray)          : Array of appliance amplitude standard deviations
            comp_proportion_zzz (np.ndarray): Array of component proportions from gaussian mixture for each amplitudes
            hvac_params (dict)               : Dictionary containing hvac algo related initialized parameters
            diff_prob(float)                 : Measure of difference between two normal distributions
            logger_base (logger)     : Writes logs during code flow
        Returns:
            mu (float)                       : Detected appliance mean
            sigma (float)                    : Detected standard deviation in measurement of appliance mean amplitude
        """

    logger_local = logger_base.get("logger").getChild("find_mu_sigma")
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # sorting probable amplitudes by their importance
    sorted_model = np.argsort(np.sqrt(model.weights_) * np.squeeze(model.means_))[::-1]
    logger_hvac.debug(('sorted model based on model weights {} and model means {} |'.format(model.weights_,
                                                                                            model.means_)).replace('\n',
                                                                                                                   ' '))
    for i in sorted_model:
        mu = model.means_[i, 0]
        sigma = np.sqrt(model.covariances_[i])[0, 0]

        mu = np.around(mu, 2)
        sigma = np.around(sigma, 2)

        mu_zzz[i + 1] = mu
        sigma_zzz[i + 1] = sigma
        comp_proportion_zzz[i + 1] = model.weights_[i]
        # checking if selected mu and sigma is valid and if it is valid then stop searching
        if mu - hvac_static_params['arm_65'] * sigma > hvac_params['MIN_AMPLITUDE'] and model.weights_[i] * diff_prob > \
                hvac_params['MIN_PROPORTION']:
            logger_hvac.info('appropriate mu and sigma estimated based on minimum amplitude |')
            break
    return mu, sigma
