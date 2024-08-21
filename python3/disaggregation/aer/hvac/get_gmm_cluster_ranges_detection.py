"""
Author - Mirambika
Date - 09/01/2024
Module to perform gaussian means clustering and identify ranges of modes of hvac consumption
"""

# Import python packages
import logging
import matplotlib
import numpy as np
from sklearn import mixture

matplotlib.use('Agg')

# Import packages from within the pipeline
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params

hvac_static_params = hvac_static_params()


def get_kl_divergence(first_distribution, second_distribution):
    """
    Function measures the difference between Two normal distributions

    Parameters:
        first_distribution  (np.ndarray)   :  First Normal distribution
        second_distribution (np.ndarray)   :  Second Normal Distribution

    Return:
        kl_divergence       (float)        : Measure of difference between between two distributions
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
        model           (list) : List containing all the gaussian models tried while finding best number of estimators

    Returns:
        amplitude_info  (dict) : Dictionary containing all mode related information (mu, sigma, mode limits)
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

    amplitude_info = {
        'means': [np.around(small_mu), np.around(large_mu)],
        'std': [np.around(small_sigma), np.around(large_sigma)],
        'cluster_overlap': clusters_overlap,
        'cluster_limits': [np.around(small_mode_limits), np.around(large_mode_limits)]
    }

    return amplitude_info


def get_num_gauss_components(filled_gauss_data, adjust_ac_nclusters_flag=False):
    """
    Function to determine optimum number of modes for AC/SH.

    Parameters:
        filled_gauss_data           (np.ndarray)    : Array over which fitting process of gaussian model is tried (AC/SH)
        adjust_ac_nclusters_flag    (int)           : Integer indicating discontinuous AC/SH residential usage type where
                                                    epochs might be scarce
    Returns:
        n_components                (int)           : Optimum number of modes found
        clfs                        (list)          : The list containing all of gaussian models tried
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

    # Avoiding multiple modes in the case of scarce hvac epochs
    if adjust_ac_nclusters_flag:
        n_components = 1

    return n_components, clfs


def fit_residual_gaussian(hist_degree_day, hist_mid_temp, raw_hist_centers, hvac_params_app, app_type, logger_base,
                          hvac_exit_status):
    """
    Function fits a gaussian mixture model to calculate mean and standard deviation of calculated app amplitudes

   Parameters:

       hist_degree_day          (np.ndarray)    : Histogram of consumptions in cdd or hdd scope
       hist_mid_temp            (np.ndarray)    : Histogram of consumptions, in mid temperature range scope
       raw_hist_centers         (np.ndarray)    : Array of all raw consumption histogram centers
       hvac_params_app          (dict)          : Dictionary containing appliance detection related parameters
       app_type                 (string)        : String containing name of appliance type (SH, AC or WH)
       logger_base              (logger)        : Writes logs during code flow
       hvac_exit_status         (dict)          : Dictionary containing hvac exit code and list of handled errors

   Returns:

       kl_divergence            (float)         : KL Divergence (separation b/w app and mid temp distributions)
       mu                       (float)         : Detected appliance mean
       sigma                    (float)         : Detected standard deviation in measurement of app mean amplitude
       hist_diff                (np.ndarray)    : Contains difference of standardized app and mid temp histograms
       found                    (bool)          : Indicator of whether applicance is found or not
       hvac_exit_status         (dict)          : Dictionary containing hvac exit code and list of handled errors
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
    diff_prob = float(np.sum(hist_diff))
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


def fit_optimum_gaussian(hist_degree_day, hist_mid_temp, raw_hist_centers, hvac_params_app, app_type, params_dict,
                         logger_base,
                         hvac_exit_status):
    """
    Function fits a gaussian mixture model to calculate mean and standard deviation of calculated app amplitudes

   Parameters:
       hist_degree_day          (np.ndarray)       : Histogram of consumptions in cdd or hdd scope
       hist_mid_temp            (np.ndarray)       : Histogram of consumptions, in mid temperature range scope
       raw_hist_centers         (np.ndarray)       : Array of all raw consumption histogram centers
       hvac_params_app          (dict)             : Dictionary containing appliance detection related parameters
       app_type                 (string)           : String indicating appliance type (SH, AC or WH)
       params_dict              (dict)             : Dictionary containing pre detection pipeline user parameters
       logger_base              (logger)           : Writes logs during code flow
       hvac_exit_status         (dict)             : Dictionary containing hvac exit code and list of handled errors

   Returns:
       kl_divergence            (float)            : KL Divergence (separation b/w app and mid temp distributions)
       mu                       (float)            : Detected appliance mean
       sigma                    (float)            : Detected standard deviation in measurement of app mean amplitude
       hist_diff                (np.ndarray)       : Contains difference of standardized app and mid temp histograms
       found (bool)                                : Indicator of whether applicance is found or not
       amplitude_info           (dict)             : Dictionary containing multi-mode information. Mu sigma etc
       hvac_exit_status         (dict)              Dictionary containing hvac exit code and list of handled errors
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

    if np.nansum(hist_diff) == 0 or (
            app_type == 'AC' and params_dict.get('pre_pipeline_params').get('all_flags').get('isNotAC')):
        logger_hvac.info('no difference in {} histogram vs mid temp histogram. mu=0 sigma=0 |'.format(app_type))
        # if no difference in mid temp and appliance histogram, mu=0, sigma=0 and found=0, to avoid crash hvac algo
        mu = 0
        sigma = 0
        found = False
        amplitude_info = {'means': [np.inf, np.inf], 'std': [0, 0], 'cluster_overlap': False,
                          'cluster_limits': ((np.inf, np.inf), (np.inf, np.inf)), 'bin_centers': [],
                          'all_gaussians': [], 'number_of_modes': 0}

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

    # To mark discontinous HVAC usage type / cyclic temp profile type
    adjust_ac_nclusters_flag = False
    if app_type == 'AC':
        adjust_ac_nclusters_flag = params_dict['pre_pipeline_params']['all_flags']['adjust_ac_nclusters_flag']

    n_components, models = get_num_gauss_components(filled_gauss_data, adjust_ac_nclusters_flag)

    amplitude_info = {
        'mean': [np.Inf, np.Inf],
        'std': [0, 0],
        'cluster_overlap': False,
        'cluster_limits': ((np.Inf, np.Inf), (np.Inf, np.Inf))
    }

    if n_components == 1:

        model = models[n_components - 1]
        mu = np.squeeze(model.means_)
        sigma = np.sqrt(np.squeeze(model.covariances_))
        amplitude_info = {'means': [np.around(mu), np.Inf],
                          'std': [np.around(sigma), 0],
                          'cluster_overlap': False,
                          'cluster_limits': ((np.around(mu - 2 * sigma), np.around(mu + 3 * sigma)), (np.Inf, np.Inf))

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
            model               (object)        : Contains gaussian mixture model related attributes
            mu_zzz              (np.ndarray)    : Array of appliance detected amplitude probables
            sigma_zzz           (np.ndarray)    : Array of appliance amplitude standard deviations
            comp_proportion_zzz (np.ndarray)    : Array of component proportions from gaussian mixture for each amplitudes
            hvac_params         (dict)          : Dictionary containing hvac algo related initialized parameters
            diff_prob           (float)         : Measure of difference between two normal distributions
            logger_base         (logger)        : Writes logs during code flow

        Returns:
            mu                  (float)         : Detected appliance mean
            sigma               (float)         : Detected standard deviation in measurement of appliance mean amplitude
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


def variance(bin_centers, hist_diff):
    """
    Function calculates variance for a histogram distribution based on histogram bar weights

    Parameters:
        bin_centers (np.ndarray)       : Array of histogram centers from which variance has to be calculated
        hist_diff   (np.ndarray)       : Array of bar weights for each of histogram bins

    Returns:
        variance    (np.ndarray)       : Variance of histogram
    """

    hist_diff = hist_diff / np.sum(hist_diff)
    x1 = np.sum(hist_diff * bin_centers)
    variance = np.sum(hist_diff * (bin_centers - x1) ** 2)
    return variance
