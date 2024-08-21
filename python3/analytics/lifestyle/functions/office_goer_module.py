"""
Author - Prasoon Patidar
Date - 18th June 2020
Lifestyle Submodule to calculate office goer probability
"""

# import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.logs_utils import log_prefix
from python3.utils.time.get_time_diff import get_time_diff
from python3.analytics.lifestyle.functions.topological_peak_detection import get_closest_peaks
from python3.analytics.lifestyle.functions.topological_peak_detection import topological_peak_detection


def get_office_goer_probability(day_input_data, day_input_idx, day_clusters, weekday_cluster_fractions,
                                lowcooling_constant, lifestyle_input_object, logger_pass):

    """
    Parameters:
        day_input_data (np.ndarray)                : custom trimmed weekday input data in 2d-day level matrix
        day_input_idx  (np.ndarray)                : day index for custom trimmed weekday input
        day_clusters   (np.ndarray)                : cluster ids for custom trimmed weekday input
        weekday_cluster_fractions(np.ndarray)      : cluster fraction for weekday data in trimmed weekday input
        lowcooling_constant                        : lowcooling constant for this user
        lifestyle_input_object(dict)               : dictionary containing inputs for lifestyle modules
        logger_pass(dict)                          : contains base logger and logging dictionary
    Returns:
        office_goer_prob(float)                    : probability of being a office goer given input data
        debug(dict)                                : step wise info for debugging and plotting purposes
    """

    t_office_goer_module_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_office_goer_probability')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Office Goer Probability", log_prefix('OfficeGoer'))

    # Initialize office goer probability, lowcooling based office goer prob and debug object

    office_goer_prob = 0.

    office_goer_lowcooling_prob = 0.

    debug = dict()

    # return if input data is empty

    if day_input_data.shape[0] == 0:
        logger.info("%s unable to retrieve office goer probability..", log_prefix('OfficeGoer'))

        return None, debug

    # Get office goer and other relevant config from lifestyle input object

    office_goer_config = lifestyle_input_object.get('office_goer_config')
    daily_load_type = lifestyle_input_object.get('daily_load_type')
    daily_kmeans_model = lifestyle_input_object.get('daily_profile_kmeans_model')
    day_vacation_info = lifestyle_input_object.get('day_vacation_info')

    vacation_days = day_vacation_info[day_vacation_info[:,1] == 1, 0]

    # get cluster labels and cluster center for this kmeans model

    cluster_labels = daily_kmeans_model.get('cluster_labels')

    cluster_centers = daily_kmeans_model.get('cluster_centers')

    # get lowcooling weight based on lowcooling constant

    min_lowcooling_ratio = office_goer_config.get('MIN_LOW_COOLING_RATIO')

    lowcooling_weight = lowcooling_constant + min_lowcooling_ratio

    logger.debug("%s lowcooling Weight: %s", log_prefix('OfficeGoer'), str(lowcooling_weight))

    # Loop over all clusters to get office goer score

    primary_office_goer_clusters = office_goer_config.get('OFFICE_CLUSTERS_PRIMARY')

    non_winter_office_goer_clusters = office_goer_config.get('OFFICE_CLUSTERS_NONWINTER')

    min_allowed_cluster_fraction = office_goer_config.get('MIN_DAYS_CLUSTER_FRACTION')

    break_threshold_percentile = office_goer_config.get('peak_detection_break_threshold_percentile')

    for cluster_id in np.argsort(weekday_cluster_fractions)[::-1]:

        # exit the loop if cluster fraction is less than min allowed cluster fraction

        if weekday_cluster_fractions[cluster_id] < min_allowed_cluster_fraction:
            break

        # execute the loop if cluster fraction in primary office clusters

        if daily_load_type(cluster_id) in primary_office_goer_clusters:

            logger.debug("%s Process office goer for cluster %s with fraction %s",
                         log_prefix('OfficeGoer'), str(daily_load_type(cluster_id).name), str(weekday_cluster_fractions[cluster_id]))

            # Get cluster input data and day idx

            cluster_input_data = day_input_data[day_clusters == cluster_id]

            cluster_input_idx = day_input_idx[day_clusters == cluster_id]

            # Get cluster center for given cluster_id

            cluster_name = daily_load_type(cluster_id).name

            cluster_center_idx = cluster_labels.index(cluster_name)

            cluster_center = cluster_centers[cluster_center_idx]

            # Get peak for cluster center based on topological peak detection

            #TODO(Nisha) : change usage of all percentile methods to one present in utils folder

            center_peaks = topological_peak_detection(cluster_center,
                                                      np.percentile(cluster_center, break_threshold_percentile),
                                                      logger_pass)

            # remove any center peaks which start from midnight or have started just before midnight
            # in the scenario when we receive more than 2 peaks

            if len(center_peaks) > 2:
                center_peaks = [center_peak for center_peak in center_peaks
                                if (center_peak.born > 0) & (center_peak.left < 23)]

            # sort center peaks based on when a peak is born

            center_peaks.sort(key=lambda x: x.born)

            # get peaks for all days in cluster input data

            get_consumption_peaks = lambda x: topological_peak_detection(x,
                                                                         np.percentile(x, break_threshold_percentile),
                                                                         logger_pass)

            cons_peaks_arr = [get_consumption_peaks(x) for x in cluster_input_data]

            # get morning peakstart and peakend times for consumption peaks

            peak_information = get_morning_evening_peak_information(cons_peaks_arr,
                                                                    center_peaks,
                                                                    office_goer_config,
                                                                    logger_pass)

            # preprocess morning peak times

            morning_peak_start_times = peak_information.get('morning_peak_start_times')

            morning_peak_end_times = peak_information.get('morning_peak_end_times')

            # if more than 50% of morning peaks are not available, exit the cluster loop

            if np.median(morning_peak_end_times) == -1:
                logger.info("%s Sufficient morning peaks not available for cluster: %s, fraction: %s",
                            log_prefix('OfficeGoer'), cluster_name, str(weekday_cluster_fractions[cluster_id]))

                continue

            # set all non available peak end and start times to median times

            morning_peak_start_times[morning_peak_start_times == -1] = np.median(morning_peak_start_times)

            morning_peak_end_times[morning_peak_end_times == -1] = np.median(morning_peak_end_times)

            # pre process evening peaks

            evening_peak_start_times = peak_information.get('evening_peak_start_times')

            # if more than 50% of evening peaks are not available, exit the cluster loop

            if np.median(evening_peak_start_times) == -1:
                logger.info("%s Sufficient evening peaks not available for cluster: %s, fraction: %s",
                            log_prefix('OfficeGoer'), cluster_name, str(weekday_cluster_fractions[cluster_id]))

                continue

            # set all non available peak start times to median times

            evening_peak_start_times[evening_peak_start_times == -1] = np.median(evening_peak_start_times)

            # get office time consumption(non peak consumption) based on morning and evening peak times(excluding evening peak_start hour)

            office_time_indexes = np.array([morning_peak_end_times, evening_peak_start_times - 1]).T

            col_range = np.arange(cluster_input_data.shape[1])

            office_time_mask = \
                (office_time_indexes[:, 0, None] <= col_range) & (office_time_indexes[:, 1, None] >= col_range)

            # get day time mask based on morning start times

            day_end_times = np.full(cluster_input_data.shape[0], fill_value=Cgbdisagg.HRS_IN_DAY)

            day_time_indexes = np.array([morning_peak_start_times, day_end_times]).T

            day_time_mask = (day_time_indexes[:, 0, None] <= col_range) & (day_time_indexes[:, 1, None] >= col_range)

            # Get fraction of consumption in non peak hours

            cluster_input_data_copy = copy.deepcopy(cluster_input_data)

            cluster_input_data_copy[~day_time_mask] = 0

            overall_daytime_consumption = np.sum(cluster_input_data_copy, axis=1)

            cluster_input_data_copy[~office_time_mask] = 0

            overall_office_time_consumption = np.sum(cluster_input_data_copy, axis=1)

            daily_non_peak_consumption_fraction = overall_office_time_consumption / overall_daytime_consumption

            # set nonpeak consumption to 1 for days which are vacation days

            is_vacation_day = np.isin(cluster_input_idx, vacation_days)

            daily_non_peak_consumption_fraction[is_vacation_day] = 1

            non_peak_consumption_percentile = office_goer_config.get('NONPEAK_CONSUMPTION_PERCENTILE')

            non_peak_consumption_fraction = np.nanpercentile(daily_non_peak_consumption_fraction,
                                                             non_peak_consumption_percentile)

            logger.debug("%s Non peak consumption fraction | %s",
                         log_prefix('OfficeGoer'), str(round(non_peak_consumption_fraction, 2)))

            # Update fraction based on min non peak ratio

            MIN_NONPEAK_CONSUMPTION_RATIO = office_goer_config.get('MIN_NONPEAK_CONSUMPTION_RATIO')

            non_peak_consumption_fraction = max(0, non_peak_consumption_fraction - MIN_NONPEAK_CONSUMPTION_RATIO)

            # update office goer consumption based on non peak consumption ratio

            office_goer_prob += weekday_cluster_fractions[cluster_id] * (1 - non_peak_consumption_fraction)

            # update debug object

            debug.update({
                cluster_id: {
                    'non_peak_fraction'      : non_peak_consumption_fraction,
                    'cluster_fraction'       : weekday_cluster_fractions[cluster_id],
                    'office_time_consumption': overall_office_time_consumption,
                    'day_time_consumption'   : overall_daytime_consumption,
                    'num_days'               : cluster_input_data.shape[0],
                    'cluster_day_idx'        : cluster_input_idx,
                    'office_time_mask'       : office_time_mask,
                }
            })

        elif daily_load_type(cluster_id) in non_winter_office_goer_clusters:

            logger.debug("%s Process lowcooling for cluster %s with fraction %s",
                         log_prefix('OfficeGoer'), daily_load_type(cluster_id).name, str(weekday_cluster_fractions[cluster_id]))

            # get lowcooling constant based

            office_goer_lowcooling_prob += weekday_cluster_fractions[cluster_id] * lowcooling_weight

            # Add information in debug object

            debug.update({
                cluster_id: {
                    'cluster_fraction' : weekday_cluster_fractions[cluster_id],
                    'lowcooling_weight': lowcooling_weight
                }
            })

    # add overall lowcooling prob in debug

    debug.update({
        'office_lowcooling_prob'  : office_goer_lowcooling_prob,
        'office_goer_primary_prob': office_goer_prob
    })

    logger.debug("%s Office goer lowcooling prob | %s",
                 log_prefix('OfficeGoer'), str(round(office_goer_lowcooling_prob, 2)))

    t_office_goer_module_end = datetime.now()

    logger.debug("%s Got Office Goer probability in | %.3f s", log_prefix('OfficeGoer'),
                 get_time_diff(t_office_goer_module_start, t_office_goer_module_end))

    return office_goer_prob, debug


def get_morning_evening_peak_information(cons_peaks_arr, center_peaks, office_goer_config, logger_pass):

    """
    Parameters:
        cons_peaks_arr   (np.ndarray)              : arr of peaks information for each day in cluster input
        center_peaks     (np.ndarray)              : arr of peaks in cluster_center
        office_goer_config(dict)                   : static config for office goers
        logger_pass(dict)                          : contains base logger and logging dictionary
    Returns:
        peak_information(dict)                     : dict containing information for regarding peak start and end times
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_morning_evening_peak_information')
    logger_pass['logger_base'] = logger_base

    # get relevant config from office goer config

    morning_peak_start_limit = office_goer_config.get('morning_peak_start_min_hour')

    morning_peak_end_limit = office_goer_config.get('morning_peak_end_max_hour')

    # Get reference morning peak from center_peaks and limit hours based on config

    morning_center_peak = center_peaks[0]

    morning_center_peak.right = min(morning_center_peak.right, morning_peak_end_limit)

    morning_center_peak.left = min(morning_center_peak.left, morning_peak_end_limit)

    # Get morning day peaks for all cons peaks

    best_morning_peak_start_limit = office_goer_config.get('best_morning_peak_start_limit')

    best_morning_peak_end_limit = office_goer_config.get('best_morning_peak_end_limit')

    duration_threshold = office_goer_config.get('duration_threshold_in_hour')

    distance_threshold = office_goer_config.get('distance_threshold_in_hour')

    distance_weight = office_goer_config.get('distance_scoring_weight')

    duration_weight = office_goer_config.get('duration_scoring_weight')

    get_best_morning_peak = lambda x: get_closest_peaks(np.array(x), morning_center_peak,
                                                        distance_threshold,
                                                        duration_threshold,
                                                        distance_weight,
                                                        duration_weight,
                                                        best_morning_peak_start_limit,
                                                        best_morning_peak_end_limit,
                                                        logger_pass)

    day_morning_peaks = np.array([get_best_morning_peak(x) for x in cons_peaks_arr])

    # morning peak end and peak start times(limit them to peak limits)

    fn_morning_peak_end_times = np.vectorize(lambda x: min(x.right, morning_peak_end_limit) if x is not None else -1)

    morning_peak_end_times = fn_morning_peak_end_times(day_morning_peaks)

    # Make any morning peak start times less than 5am, if not less, then make them 0, used for day consumption

    fn_morning_peak_start_times = np.vectorize(lambda x: min(x.left, morning_peak_start_limit) if x is not None else 0)

    morning_peak_start_times = fn_morning_peak_start_times(day_morning_peaks)

    # Get reference evening peaks from center peaks and limit hours based on config

    evening_peak_start_limit = office_goer_config.get('evening_peak_start_min_hour')

    evening_center_peak = center_peaks[-1]

    evening_center_peak.left = max(evening_center_peak.left, evening_peak_start_limit)

    evening_center_peak.right = max(evening_center_peak.right, evening_peak_start_limit)

    # Get evening day peaks for all cons peaks

    best_evening_peak_start_limit = office_goer_config.get('best_evening_peak_start_limit')

    best_evening_peak_end_limit = office_goer_config.get('best_evening_peak_end_limit')

    get_best_evening_peak = lambda x: get_closest_peaks(np.array(x), evening_center_peak,
                                                        distance_threshold,
                                                        duration_threshold,
                                                        distance_weight,
                                                        duration_weight,
                                                        best_evening_peak_start_limit,
                                                        best_evening_peak_end_limit,
                                                        logger_pass)

    day_evening_peaks = np.array([get_best_evening_peak(x) for x in cons_peaks_arr])

    # evening peak start times(limit them to peak limits)

    fn_evening_peak_start_times = np.vectorize(lambda x: max(x.left, evening_peak_start_limit) if x is not None else -1)

    evening_peak_start_times = fn_evening_peak_start_times(day_evening_peaks)

    # write final peak information dict

    peak_information = {
        'morning_peak_start_times': morning_peak_start_times,
        'morning_peak_end_times'  : morning_peak_end_times,
        'evening_peak_start_times': evening_peak_start_times
    }

    return peak_information


def get_office_goer_annual_prob(lifestyle_input_object, lifestyle_output_object, logger_pass):

    """
    Parameters:
        lifestyle_input_object(dict)               : dictionary containing inputs for lifestyle modules
        lifestyle_output_object(dict)              : dictionary containing step wise outputs for lifestyle modules
        logger_pass(dict)                          : contains base logger and logging dictionary
    Returns:
        is_office_goer(float)                      : Whether this user is an office goer or not
        office_goer_prob(float)                    : probability of this user being a office goer
    """

    t_office_goer_annual_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('office_goer_annual_probability')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Office goer annual probability", log_prefix('OfficeGoer'))

    # get relevant inputs for office goer probability

    seasons = lifestyle_input_object.get('weather_config').get('season')

    office_goer_config = lifestyle_input_object.get('office_goer_config')

    season_fractions = lifestyle_output_object.get('annual').get('season_fraction')

    # Get winter season relevant features

    winter_office_goer_info =\
        lifestyle_output_object.get('season').get(seasons.winter.name).get('office_goer_debug', {})
    winter_office_goer_prob = winter_office_goer_info.get('office_goer_primary_prob', 0)
    winter_low_prob_threshold = office_goer_config.get('winter_low_prob_threshold')
    winter_high_prob_threshold = office_goer_config.get('winter_high_prob_threshold')

    # Get Transition relevant features

    transition_office_goer_info =\
        lifestyle_output_object.get('season').get(seasons.transition.name).get('office_goer_debug', {})
    transition_office_goer_prob = transition_office_goer_info.get('office_goer_primary_prob', 0)
    transition_office_goer_lowcooling_prob = transition_office_goer_info.get('office_lowcooling_prob', 0)
    transition_high_prob_threshold = office_goer_config.get('transition_high_prob_threshold')

    # Get summer relevant features

    summer_office_goer_info = \
        lifestyle_output_object.get('season').get(seasons.summer.name).get('office_goer_debug', {})
    summer_office_goer_prob = summer_office_goer_info.get('office_goer_primary_prob', 0)
    summer_office_goer_lowcooling_prob = summer_office_goer_info.get('office_lowcooling_prob', 0)
    summer_low_prob_threshold = office_goer_config.get('summer_low_prob_threshold')

    # include lowcooling score in actual summer and transition probabilities if winter is atleast lower threshold

    if (winter_office_goer_prob > winter_low_prob_threshold) | (season_fractions[seasons.winter.value] <= 0.):
        logger.debug("%s Including lowcooling prob for summer and transition", log_prefix('OfficeGoer'))

        summer_office_goer_prob += summer_office_goer_lowcooling_prob

        transition_office_goer_prob += transition_office_goer_lowcooling_prob

    logger.debug("%s seasonal probs: winter: %s, summer: %s, transition: %s",
                 log_prefix('OfficeGoer'), str(round(winter_office_goer_prob, 2)),
                 str(round(summer_office_goer_prob, 2)), str(round(transition_office_goer_prob, 2)))

    # check if we need to include summer based on winter and transition thresholds

    include_summer = True

    if (0 < winter_office_goer_prob > winter_high_prob_threshold) & \
            (0 < transition_office_goer_prob > transition_high_prob_threshold) & \
            (0 < summer_office_goer_prob < summer_low_prob_threshold):
        # if winter and transition probabilities are high enough, do not include summer

        logger.debug("%s Removing summer season from calculation", log_prefix('OfficeGoer'))

        include_summer = False

    # get final probabilities by doing a weighted(by season fraction) sum based on whether to include summer or not

    office_goer_prob = 0.

    if not include_summer:

        office_goer_prob += winter_office_goer_prob * season_fractions[seasons.winter.value]

        office_goer_prob += transition_office_goer_prob * season_fractions[seasons.transition.value]

        office_goer_prob /= (season_fractions[seasons.winter.value] + season_fractions[seasons.transition.value])

    else:
        office_goer_prob += winter_office_goer_prob * season_fractions[seasons.winter.value]

        office_goer_prob += transition_office_goer_prob * season_fractions[seasons.transition.value]

        office_goer_prob += summer_office_goer_prob * season_fractions[seasons.summer.value]

    # Based on overall probability, judge whether a user is office goer based on threshold

    office_goer_prob_threshold = office_goer_config.get('office_goer_prob_threshold')

    is_office_goer = False

    if office_goer_prob > office_goer_prob_threshold:
        is_office_goer = True

    office_goer_score_soft_margin = office_goer_config.get('office_goer_score_soft_margin')

    # updating office goer tag based on previous run ouput

    lifestyle_hsm = lifestyle_input_object.get('lifestyle_hsm')

    # fetching HSM info

    if (lifestyle_hsm is None) or (lifestyle_hsm.get('attributes') is None):
        logger.warning('Lifestyle HSM attributes are absent | ')

    else:
        if lifestyle_hsm.get('attributes').get('office_goer_score') is None:
            logger.warning('office goer score attribute is missing in lifestyle HSM | ')
        else:
            hsm_in = lifestyle_hsm.get('attributes').get('office_goer_score')[0]

            # checking if current run score is close to previous run score
            # if yes, the attribute tag is kept similar to previous run

            if is_office_goer and (hsm_in <= office_goer_prob_threshold) and ((office_goer_prob-hsm_in) < office_goer_score_soft_margin):
                logger.info('Office goer probability in HSM is | %s ', hsm_in)
                logger.info('Updating office goer tag to | %s ', False)
                is_office_goer = False
                office_goer_prob = hsm_in

            elif (not is_office_goer) and (hsm_in > office_goer_prob_threshold) and ((hsm_in-office_goer_prob) < office_goer_score_soft_margin):
                logger.info('Office goer probability in HSM is | %s ', hsm_in)
                logger.info('Updating office goer tag to | %s ', True)
                is_office_goer = True
                office_goer_prob = hsm_in

    # Get final Seasonal probabilites for debugging purposes

    office_goer_seasonal_prob = {
        'winter'    : winter_office_goer_prob,
        'summer'    : summer_office_goer_prob,
        'transition': transition_office_goer_prob
    }

    t_office_goer_annual_end = datetime.now()

    logger.debug("%s Got annual office goer probability in | %.3f s", log_prefix('OfficeGoer'),
                 get_time_diff(t_office_goer_annual_start,
                               t_office_goer_annual_end))

    return is_office_goer, office_goer_prob, office_goer_seasonal_prob
