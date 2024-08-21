"""
Author - Kris Duszak
Date - 3/8/2019
Module to detect HVAC amplitude for Japan
"""

# Import python packages

import copy
import logging
import numpy as np
from sklearn import mixture
from scipy.stats.mstats import mquantiles

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile as super_percentile

from python3.disaggregation.aer.hvac.hvac_japan_functions.get_vacations import get_vacation_info
from python3.disaggregation.aer.hvac.hvac_japan_functions.compute_sh_detection_features import compute_features_for_sh_detection


def variance(bin_centers, hist_diff):

    """
    Function calculates variance for a histogram distribution based on histogram bar weights

    Parameters:
        bin_centers     (np.ndarray)            : Array of histogram centers from which variance has to be calculated
        hist_diff       (np.ndarray)            : Array of bar weights for each of histogram bins

    Returns:
        variance        (np.ndarray)            : Variance of histogram
    """

    hist_diff = hist_diff / np.sum(hist_diff)
    x1 = np.sum(hist_diff * bin_centers)
    vari = np.sum(hist_diff * (bin_centers - x1) ** 2)

    return vari


def detect_hvac_amplitude_jp(hvac_input, invalid_idx, hvac_params, vacation_periods, logger_pass, hvac_exit_status):

    """
    Function detects JP hvac appliance amplitude and standard deviation and stores detection level parameters

    Parameters:
        hvac_input          (np.ndarray)        : Array of epoch level consumption flowing into hvac module
        invalid_idx         (np.ndarray)        : Array of invalid epochs based on consumption and temperature
        hvac_params         (dict)              : Dictionary containing hvac algo related initialized parameters
        vacation_periods    (np.ndarray)        : Array containing vacation periods
        logger_pass         (dict)              : Writes logs during code flow
        hvac_exit_status    (dict)              : Dictionary containing hvac exit code and list of handled errors

    Returns:
        debug_detection     (dict)              : Dictionary containing hvac detection related debugging information
    """

    # Initialize the logger for this function

    logger_base = logger_pass.get("logger").getChild("detect_hvac_amplitude")
    logger_hvac = logging.LoggerAdapter(logger_base, logger_pass.get("logging_dict"))

    # Prepare the logger pass dictionary

    logger_pass['logger'] = logger_base

    # initializing dictionary for storing hvac detection related key debugging information

    debug_detection = {
        'mid': {},
        'hdd': {'found': False, 'setpoint': np.nan, 'mu': np.nan, 'sigma': np.nan},
        'cdd': {'found': False, 'setpoint': np.nan, 'mu': np.nan, 'sigma': np.nan}
    }

    # get variables

    hvac_input_consumption = hvac_input[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    hvac_input_temperature = hvac_input[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]

    # removing consumption outliers from detection process

    quan_005 = np.around(super_percentile(hvac_input_consumption, 0.5), 5)
    quan_995 = np.around(super_percentile(hvac_input_consumption, 99.5), 5)

    # getting qualified epoch level consumption points falling in between above percentile ranges

    mid_inlier_idx = np.logical_and(hvac_input_consumption >= quan_005, hvac_input_consumption <= quan_995)

    # getting raw consumption histogram

    _, raw_hist_edges = np.histogram(hvac_input_consumption[mid_inlier_idx], hvac_params['detection']['NUM_BINS'],
                                     density=True)

    logger_hvac.info('histogram edges for entire consumption range is established, except outliers |')

    logger_hvac.info(' ------------------ Mid Temperature Range --------------------- |')

    # getting epoch points qualified for being in mid temperature range scope

    mid_temp_scope_idx = np.logical_and(~np.isnan(hvac_input_temperature),
                                        np.logical_and(hvac_input_temperature >=
                                                       hvac_params['detection']['MID_TEMPERATURE_RANGE'][0],
                                                       hvac_input_temperature <=
                                                       hvac_params['detection']['MID_TEMPERATURE_RANGE'][1]))

    logger_hvac.debug('{} data points in mid temperature scope before taking quantiles |'.format(
        np.sum(mid_temp_scope_idx)))

    mid_temp_scope = hvac_input_temperature[mid_temp_scope_idx]

    # getting low limit of mid temperature range

    low_limit_mid_range = mquantiles(mid_temp_scope, hvac_params['detection']['MID_TEMPERATURE_QUANTILE'][0],
                                     alphap=0.5, betap=0.5)

    # getting high limit of mid temperature range

    high_limit_mid_range = mquantiles(mid_temp_scope, hvac_params['detection']['MID_TEMPERATURE_QUANTILE'][1],
                                      alphap=0.5, betap=0.5)

    logger_hvac.info('mid temperature range identified |')
    logger_hvac.debug('mid temperature range identified {} - {} |'.format(low_limit_mid_range, high_limit_mid_range))

    # getting mid temperature epochs, from low_limit_mid_range and high_limit_mid_range

    mid_temp_idx = (hvac_input_consumption > 0) & (hvac_input_temperature >= low_limit_mid_range) & \
                   (hvac_input_temperature <= high_limit_mid_range) & (~invalid_idx.astype(bool))

    logger_hvac.debug('>> {} data points in final mid temperature range {}F - {}F |'.format(np.sum(mid_temp_idx),
                                                                                            low_limit_mid_range,
                                                                                            high_limit_mid_range))
    # getting histogram centers from raw consumption histogram

    raw_hist_centers = np.r_[0.5 * (raw_hist_edges[:-1] + raw_hist_edges[1:])]
    logger_hvac.info('attempting to make mid temperature histogram |')

    # making mid temperature histogram

    bin_counts, _ = np.histogram(hvac_input_consumption[mid_temp_idx], bins=raw_hist_centers)
    bin_counts = np.r_[bin_counts, 0]
    bin_counts += 1

    # standardizing mid temperature histogram bins

    hist_mid_temp = bin_counts.T / np.sum(bin_counts)
    hist_mid_temp[0: hvac_params['detection']['NUM_BINS_TO_REMOVE']] = 0

    # assigning important attributes to debug detection dictionary for debugging

    debug_detection['mid'] = {
        'temp': [low_limit_mid_range, high_limit_mid_range],
        'edges': raw_hist_centers, 'hist': hist_mid_temp
    }

    logger_hvac.info('mid temperature detection steps complete |')

    logger_hvac.info(' ------------------ WH Amplitude detection --------------------- |')

    # detecting water heater amplitude and standard deviation

    hist_mid_temp_min = copy.deepcopy(hist_mid_temp)

    for i in range(len(hist_mid_temp)):
        hist_mid_temp_min[i] = np.min(hist_mid_temp[0:i + 1])

    logger_hvac.info('fitting residual gaussian to get water-heater mu and sigma |')

    # fitting residual gaussian to get water-heater mu and sigma

    _, mu_wh, sigma_wh, _, found_wh, hvac_exit_status = \
        fit_residual_gaussian(hist_mid_temp, hist_mid_temp_min, raw_hist_centers, hvac_params['detection']['WH'],
                              'WaterHeating', logger_pass, hvac_exit_status, debug_detection)

    # checking whether appropriate water heater is detected

    found_wh = np.logical_and(found_wh, mu_wh < hvac_params['detection']['WH']['MAX_AMPLITUDE'])
    logger_hvac.info('got water-heater mu and sigma |')
    logger_hvac.debug('water-heater : found={}, mu={}, sigma={} |'.format(found_wh, mu_wh, sigma_wh))

    if found_wh:

        logger_hvac.info('removing water-heater bins from mid range histogram |')

        # removing water-heater bins from mid range histogram

        hist_mid_temp[np.logical_and(
            raw_hist_centers >= mu_wh - hvac_params['detection']['WH']['MIN_DETECTION_STD'] * sigma_wh,
            raw_hist_centers <= mu_wh + hvac_params['detection']['WH']['MIN_DETECTION_STD'] * sigma_wh)] = 0

        hist_mid_temp = hist_mid_temp / np.sum(hist_mid_temp)
        debug_detection['mid']['hist'] = hist_mid_temp

        logger_hvac.info('removed water-heater bins from mid range histogram |')

    sh_ac_params = {
        'invalid_idx': invalid_idx,
        'raw_hist_centers': raw_hist_centers,
        'found_wh': found_wh,
        'mu_wh': mu_wh,
        'sigma_wh': sigma_wh,
        'hist_mid_temp': hist_mid_temp,
        'logger_pass': logger_pass
    }

    logger_hvac.info(' ------------------ SH Amplitude detection --------------------- |')

    # detecting sh amplitude and standard deviation

    # get vacation related information
    vacation_days_idx, vacation_days, agg_vacation_month = get_vacation_info(hvac_input, vacation_periods,
                                                                             debug_detection,
                                                                             percent_vacation_daily_thresh=0.6)

    logger_hvac.info('Number of vacation days from vacation periods | {}'.format(len(vacation_days)))

    # compute features which will be used for user level logistic regression SH model

    logger_hvac.info('attempting to compute features for SH detection logistic regression model |')
    compute_features_for_sh_detection(hvac_input, hvac_params, agg_vacation_month, logger_pass, debug_detection)

    detect_sh_amplitude(hvac_params, low_limit_mid_range, hvac_input_consumption, hvac_input_temperature, sh_ac_params,
                        debug_detection, hvac_exit_status)

    logger_hvac.info(' ------------------ AC Amplitude detection --------------------- |')

    # detecting ac amplitude and standard deviation

    detect_ac_amplitude(hvac_params, high_limit_mid_range, hvac_input_consumption, hvac_input_temperature, sh_ac_params,
                        debug_detection, hvac_exit_status)

    return debug_detection


def detect_sh_amplitude(hvac_params, low_limit_mid_range, hvac_input_consumption, hvac_input_temperature, params_dict,
                        debug_detection, hvac_exit_status):

    """
    Function detects sh amplitude and standard deviation and stores detection in debug object

    Parameters:
        hvac_params (dict)                         : Dictionary containing hvac algo related initialized parameters
        low_limit_mid_range (float)                : Lower limit of mid temperature range
        hvac_input_consumption (np.ndarray)        : Array of epoch level consumption flowing into hvac module
        hvac_input_temperature (np.ndarray)        : Array of epoch level temperature flowing into hvac module
        params_dict(dict)
        debug_detection (dict)                     : Dictionary containing debug information from hvac detection stage
        hvac_exit_status(dict)                     : Dictionary containing hvac exit code and list of handled errors

    Returns:
        debug_detection (dict)                     : Dictionary containing SH detection related debugging information
    """

    invalid_idx = params_dict['invalid_idx']
    vacation_days_idx = debug_detection['vacation_days_idx']
    raw_hist_centers = params_dict['raw_hist_centers']
    found_wh = params_dict['found_wh']
    mu_wh = params_dict['mu_wh']
    sigma_wh = params_dict['sigma_wh']
    hist_mid_temp = params_dict['hist_mid_temp']
    logger_base = params_dict['logger_pass']

    logger_local = logger_base.get("logger").getChild("detect_sh_amplitude")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    for hdd_setpoint in hvac_params['detection']['SH']['SETPOINTS']:
        # sh setpoint should be less than permissible low limit of mid range temperature
        if hdd_setpoint < low_limit_mid_range:
            # getting sh valid epochs
            valid_idx = np.logical_and(hvac_input_consumption > 0,
                                       np.logical_and(hvac_input_temperature < hdd_setpoint, ~invalid_idx),
                                       np.logical_and(hvac_input_temperature < hdd_setpoint, ~vacation_days_idx))

            # there should be minimum enough valid epochs for getting setpoint. to avoid false detection.
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

                # else
                hdd_kl_divergence, mu_hdd, sigma_hdd, hist_diff_hdd, found_hdd, hvac_exit_status = \
                    fit_residual_gaussian(hist_hdd, hist_mid_temp, raw_hist_centers, hvac_params['detection']['SH'],
                                          'SH', logger_pass, hvac_exit_status, debug_detection)

                logger_hvac.info('got sh mu and sigma |')
                # assigning important attributes to debug detection dictionary for debugging
                debug_detection['hdd'] = {
                    'setpoint': hdd_setpoint,
                    'hist': hist_hdd,
                    'histDiff': hist_diff_hdd,
                    'dist': hdd_kl_divergence,
                    'mu': mu_hdd,
                    'sigma': sigma_hdd,
                    'found': found_hdd
                }

                logger_hvac.debug('sh : found={}, mu={}, sigma={}, hist_diff={} |'.format(found_hdd, mu_hdd, sigma_hdd,
                                                                                          str(hist_diff_hdd).replace(
                                                                                              '\n', ' ')))
                break


def detect_ac_amplitude(hvac_params, high_limit_mid_range, hvac_input_consumption, hvac_input_temperature, params_dict,
                        debug_detection, hvac_exit_status):

    """
    Function detects ac amplitude and standard deviation and stores detection in debug object

    Parameters:
        hvac_params (dict)                         : Dictionary containing hvac algo related initialized parameters
        high_limit_mid_range (float)               : Higher limit of mid temperature range
        hvac_input_consumption (np.ndarray)       : Array of epoch level consumption flowing into hvac module
        hvac_input_temperature (np.ndarray)       : Array of epoch level temperature flowing into hvac module
        params_dict(dict)
        debug_detection (dict)                     : Dictionary containing debug information from hvac detection stage
        hvac_exit_status(dict)                     : Dictionary containing hvac exit code and list of handled errors
    Returns:
        debug_detection (dict)                     : Dictionary containing AC detection related debugging information
    """

    invalid_idx = params_dict['invalid_idx']
    vacation_days_idx = debug_detection['vacation_days_idx']
    raw_hist_centers = params_dict['raw_hist_centers']
    found_wh = params_dict['found_wh']
    mu_wh = params_dict['mu_wh']
    sigma_wh = params_dict['sigma_wh']
    hist_mid_temp = params_dict['hist_mid_temp']
    logger_base = params_dict['logger_pass']

    logger_local = logger_base.get("logger").getChild("detect_ac_amplitude")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    for cdd_setpoint in hvac_params['detection']['AC']['SETPOINTS']:
        # ac setpoint should be greater than permissible high limit of mid range temperature
        if cdd_setpoint > high_limit_mid_range:
            # getting ac valid epochs
            valid_idx = np.logical_and(hvac_input_consumption > 0,
                                       np.logical_and(hvac_input_temperature > cdd_setpoint, ~invalid_idx),
                                       np.logical_and(hvac_input_temperature > cdd_setpoint, ~vacation_days_idx))

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
                cdd_kl_divergence, mu_cdd, sigma_cdd, hist_diff_cdd, found_cdd, hvac_exit_status =\
                    fit_residual_gaussian(hist_cdd, hist_mid_temp, raw_hist_centers, hvac_params['detection']['AC'],
                                          'AC', logger_pass, hvac_exit_status, debug_detection)

                logger_hvac.info('got ac mu and sigma |')
                # assigning important attributes to debug detection dictionary for debugging
                debug_detection['cdd'] = {
                    'setpoint': cdd_setpoint,
                    'hist': hist_cdd,
                    'histDiff': hist_diff_cdd,
                    'dist': cdd_kl_divergence,
                    'mu': mu_cdd,
                    'sigma': sigma_cdd,
                    'found': found_cdd
                }

                logger_hvac.debug('ac : found={}, mu={}, sigma={}, hist_diff={} |'.format(found_cdd, mu_cdd, sigma_cdd,
                                                                                          str(hist_diff_cdd).replace(
                                                                                              '\n', ' ')))
                break


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
    kl_div = np.sum(temp)
    return kl_div


def fit_residual_gaussian(hist_degree_day, hist_mid_temp, raw_hist_centers, hvac_params_app, app_type, logger_base,
                          hvac_exit_status, debug_detection):

    """
    Function fits a gaussian mixture model to calculate mean and standard deviation of calculated app amplitudes

       Parameters:
           hist_degree_day (np.ndarray)              : Histogram of consumptions in cdd or hdd scope
           hist_mid_temp (np.ndarray)                : Histogram of consumptions, in mid temperature range scope
           raw_hist_centers (np.ndarray)             : Array of all raw consumption histogram centers
           hvac_params_app (dict)                     : Dictionary containing appliance detection related parameters
           app_type (string)                          : String containing name of appliance type (SH, AC or WH)
           logger_base (logger)               : Writes logs during code flow
           hvac_exit_status(dict)                     : Dictionary containing hvac exit code and list of handled errors
       Returns:
           kl_divergence (float)                      : KL Divergence (separation b/w app and mid temp distributions)
           mu (float)                                 : Detected appliance mean
           sigma (float)                              : Detected standard deviation in measurement of app mean amplitude
           hist_diff (np.ndarray)                    : Contains difference of standardized app and mid temp histograms
           found (bool)                               : Indicator of whether applicance is found or not
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
    if mu - 1.65 * sigma <= hvac_params_app['MIN_AMPLITUDE']:
        logger_hvac.info('2-mode gmm will be fit for {} |'.format(app_type))
        num_points = 10000
        fake_data = np.empty(shape=(1, 0))
        # filling distribution with made up data, for enabling better gaussian fit
        for i in range(len(bin_centers)):
            if hist_diff[i] > 0:
                n = np.ceil(hist_diff[i] * num_points)
                np.random.seed(12345)
                r1 = np.random.rand(1, int(n)) * bin_size + raw_hist_centers[i]
                fake_data = np.concatenate((fake_data, r1), axis=1)

        logger_hvac.info('getting gmm ... |')
        # initializing gaussian mixture model to data
        model_init = np.full(shape=(2, 1, 1), fill_value=0.1)
        model_init = np.linalg.inv(model_init)
        model = mixture.GaussianMixture(n_components=2, covariance_type='full', tol=0.000000000000001, max_iter=120,
                                        random_state=1,
                                        means_init=np.array([np.min(fake_data), np.max(fake_data)]).reshape((2, 1)),
                                        weights_init=np.array([0.5, 0.5]),
                                        precisions_init=model_init)
        # fitting gaussian model to made up data
        model.fit(fake_data.T)

        logger_hvac.info('gmm model fitted now finding mu and sigma for {} |'.format(app_type))
        # finding amplitude and standard deviation of hvac appliance
        mu, sigma = find_mu_sigma(model, mu_zzz, sigma_zzz,comp_proportion_zzz, hvac_params_app, diff_prob, logger_pass)
        logger_hvac.info('gmm model sorting gave mu:{} and sigma:{} for {} |'.format(mu, sigma, app_type))

    logger_hvac.info('checking if {} is found |'.format(app_type))

    # determining if a valid hvac appliance is found
    # only SH probability model exists for Japan
    if app_type == 'SH':
        found = hvac_params_app['getFound'](mu, sigma, diff_prob, kl_divergence, hvac_params_app, debug_detection) and \
                (mu - hvac_params_app['MIN_DETECTION_STD'] * sigma > hvac_params_app['MIN_AMPLITUDE'])
    else:
        found = hvac_params_app['getFound'](mu, sigma, diff_prob, kl_divergence, hvac_params_app) and \
                (mu - hvac_params_app['MIN_DETECTION_STD'] * sigma > hvac_params_app['MIN_AMPLITUDE'])
    logger_hvac.info('>> {} found is {} |'.format(app_type, found))

    return kl_divergence, mu, sigma, hist_diff, found, hvac_exit_status


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
        if mu - 1.65 * sigma > hvac_params['MIN_AMPLITUDE'] and model.weights_[i] * diff_prob > hvac_params['MIN_PROPORTION']:
            logger_hvac.info('appropriate mu and sigma estimated based on minimum amplitude |')
            break

    return mu, sigma
