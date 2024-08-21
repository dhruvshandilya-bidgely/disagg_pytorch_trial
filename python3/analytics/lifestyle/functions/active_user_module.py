"""
Author - Prasoon Patidar
Date - 18th June 2020
Lifestyle Submodule to calculate active user probability
"""

# import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.logs_utils import log_prefix
from python3.utils.time.get_time_diff import get_time_diff


def get_active_user_probability(day_input_data, weights_config, lifestyle_input_object, logger_pass):

    """
    Parameters:
        day_input_data (np.ndarray)                : custom trimmed input data in 2d-day level matrix
        weights_config                             : custom config weights for calculating active user probability
        lifestyle_input_object(dict)               : dictionary containing inputs for lifestyle modules
        logger_pass(dict)                          : contains base logger and logging dictionary
    Returns:
        active_user_prob(float)                    : probability of being a active user given input data
        debug(dict)                                : step wise info for debugging and plotting purposes
    """

    t_active_user_module_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_active_user_probability')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Active User Probability", log_prefix('ActiveUser'))

    # Initialize active user probability, and debug object

    active_user_prob = None

    debug = dict()

    # return if input data is empty

    if day_input_data.shape[0] == 0:

        logger.info("%s unable to retrieve active user probability..", log_prefix('ActiveUser'))

        return active_user_prob, debug

    # Get active_user config from lifestyle input object to get indexing values

    active_user_config = lifestyle_input_object.get('active_user_config')

    NON_BASE_IDX = active_user_config.get('NON_BASELOAD_FRACTION_IDX')
    WITHIN_DAY_IDX = active_user_config.get('WITHIN_DAY_DEVIATION_IDX')
    ACROSS_DAY_IDX = active_user_config.get('ACROSS_DAY_DEVIATION_IDX')
    ACTIVITY_IDX = active_user_config.get('AVERAGE_ACTIVITY_IDX')

    MIN_IDX, MAX_IDX, WEIGHT_IDX = map(active_user_config.get,
                                       ['MIN_VAL_IDX', 'MAX_VAL_IDX', 'WEIGHT_IDX'])

    # Initialize feature vector

    feature_count = active_user_config.get('feature_count')

    features = np.zeros(feature_count)

    # set any nan values from data to 0
    # TODO(Nisha): removing this nan value setting to match results with dev, wrong practice.

    # find baseload for each day

    baseload_percentile = active_user_config.get('baseload_percentile')

    daily_baseload = np.nanpercentile(day_input_data, baseload_percentile, axis=1)

    # get fraction of consumption NOT going into baseload

    total_baseload = np.nansum(daily_baseload * Cgbdisagg.HRS_IN_DAY)

    total_consumption = np.nansum(day_input_data)

    logger.debug("%s Total baseload: %s, consumption: %s",
                 str(round(total_baseload,2)), str(round(total_consumption,2)), log_prefix('ActiveUser'))

    non_baseload_fraction = 1 - (total_baseload / total_consumption)

    features[NON_BASE_IDX] = non_baseload_fraction

    # Get within day deviation for each day

    daily_deviation = np.nanstd(day_input_data, axis=1)

    # trim daily deviation array based on low/high percentiles accepted

    low_deviation_percentile = active_user_config.get('within_day_low_percentile')

    high_deviation_percentile = active_user_config.get('within_day_high_percentile')

    deviation_lower_limit, deviation_upper_limit = \
        np.nanpercentile(daily_deviation, [low_deviation_percentile, high_deviation_percentile])

    daily_deviation_trimmed = daily_deviation[
        (daily_deviation >= deviation_lower_limit) & (daily_deviation <= deviation_upper_limit)]

    # Get average daily deviation based on trimmed values to get sense on within day deviation

    within_day_deviation = np.mean(daily_deviation_trimmed)

    features[WITHIN_DAY_IDX] = within_day_deviation

    # Get deviation in total daily consumption

    total_day_consumption = np.nansum(day_input_data, axis=1)

    across_day_deviation = np.std(total_day_consumption)

    features[ACROSS_DAY_IDX] = across_day_deviation

    logger.debug("%s Deviation within-day: %s across-day: %s", log_prefix('ActiveUser'),
                 str(round(within_day_deviation, 2)), str(round(across_day_deviation, 2)))

    # get activity count for input data based on threshold and start hour

    activity_threshold = active_user_config.get('activity_threshold')

    activity_start_hour = active_user_config.get('activity_start_hour')

    # get diff matrix to count how many times change in consumption is beyond threshold

    diff_matrix = day_input_data[:, activity_start_hour + 1:] - day_input_data[:, activity_start_hour:-1]

    diff_matrix[np.isnan(diff_matrix)] = 0.

    # count no. of times diff is beyond activity threshold

    activity_count_up = np.count_nonzero(diff_matrix > activity_threshold)

    activity_count_down = np.count_nonzero(diff_matrix < -1 * activity_threshold)

    # Calculate average daily activity

    num_days = day_input_data.shape[0]

    average_activity = (activity_count_up + activity_count_down) / (2 * num_days)

    features[ACTIVITY_IDX] = average_activity

    logger.debug("%s Average Daily Activity: %s", log_prefix('ActiveUser'), str(round(average_activity, 2)))

    # Get Min/Max vals for feature normalization

    min_vals = np.array([weights_config[idx][MIN_IDX] for idx in range(feature_count)])

    max_vals = np.array([weights_config[idx][MAX_IDX] for idx in range(feature_count)])

    # Normalize features based on weight config to calculate activity score

    features_normed = (features - min_vals) / (max_vals - min_vals)

    logger.debug("%s Features Normed: %s", log_prefix('ActiveUser'), str(features_normed))

    # Get active_user_score based on normed features

    feature_weights = np.array([weights_config[idx][WEIGHT_IDX] for idx in range(feature_count)])

    active_user_score = np.sum(features_normed * feature_weights) / np.sum(feature_weights)

    active_user_prob = min(active_user_score, 1)

    # Fill debug information

    debug.update({
        'baseload'             : total_baseload,
        'total_consumption'    : total_consumption,
        'non_baseload_fraction': non_baseload_fraction,
        'within_day_deviation' : within_day_deviation,
        'across_day_deviation' : across_day_deviation,
        'activity_count_up'    : activity_count_up,
        'activity_count_down'  : activity_count_down,
        'days_count'           : num_days,
        'average_activity'     : average_activity,
        'features_normed'      : features_normed,
        'active_user_prob'     : active_user_prob,
    })

    t_active_user_module_end = datetime.now()

    logger.debug("%s Got Active User probability in | %.3f s", log_prefix('ActiveUser'),
                 get_time_diff(t_active_user_module_start, t_active_user_module_end))

    return active_user_prob, debug


def get_active_user_annual_prob(lifestyle_input_object, lifestyle_output_object, logger_pass):

    """
    Parameters:
        lifestyle_input_object(dict)               : dictionary containing inputs for lifestyle modules
        lifestyle_output_object(dict)              : dictionary containing step wise outputs for lifestyle modules
        logger_pass(dict)                          : contains base logger and logging dictionary
    Returns:
        is_active_user(float)                      : Whether this user is an active user or not
        active_user_prob(float)                    : probability of this user being a active user
    """

    t_active_user_annual_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('active_user_annual_probability')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Active User Annual probability", log_prefix('ActiveUser'))

    # get relevant inputs for active user probability

    seasons = lifestyle_input_object.get('weather_config').get('season')

    active_user_config = lifestyle_input_object.get('active_user_config')

    season_fractions = lifestyle_output_object.get('annual').get('season_fraction')

    # Get winter season relevant features

    winter_active_user_prob =\
        lifestyle_output_object.get('season').get(seasons.winter.name).get('active_user_prob', 0)

    # Get Transition relevant features

    transition_active_user_prob = \
        lifestyle_output_object.get('season').get(seasons.transition.name).get('active_user_prob', 0)

    transition_season_fraction_high_threshold = active_user_config.get('transition_season_fraction_high_threshold')

    # Get summer relevant features

    summer_active_user_prob =\
        lifestyle_output_object.get('season').get(seasons.summer.name).get('active_user_prob', 0)

    summer_season_fraction_low_threshold = active_user_config.get('summer_season_fraction_low_threshold')

    summer_season_fraction_high_threshold = active_user_config.get('summer_season_fraction_high_threshold')

    summer_season_fraction_static = active_user_config.get('summer_season_fraction_static')

    logger.debug("%s seasonal probs: winter: %s summer: %s transition: %s", log_prefix('ActiveUser'),
                 str(round(winter_active_user_prob, 2)), str(round(summer_active_user_prob, 2)),
                 str(round(transition_active_user_prob, 2)))

    logger.debug("%s season fractions: %s", log_prefix('ActiveUser'), str(season_fractions))

    # loop over conditions to get active user prob

    active_user_prob = 0.

    if (season_fractions[seasons.transition.value] > transition_season_fraction_high_threshold):

        # If transition season fraction is high, use only transition probability

        logger.debug("%s Using only transition months for annual calculation", log_prefix('ActiveUser'))

        active_user_prob = transition_active_user_prob

    elif (season_fractions[seasons.summer.value] < summer_season_fraction_low_threshold):

        # If Summer season fraction is low, don't use summer probability

        logger.debug("%s Using winter and transition months for annual calculation", log_prefix('ActiveUser'))

        active_user_prob += winter_active_user_prob * season_fractions[seasons.winter.value]

        active_user_prob += transition_active_user_prob * season_fractions[seasons.transition.value]

        active_user_prob /= (season_fractions[seasons.winter.value] + season_fractions[seasons.transition.value])

    elif (season_fractions[seasons.summer.value] < summer_season_fraction_high_threshold):

        # Calculate probability based on winter and transition

        logger.debug("%s Using all months for annual calculation with static season fraction for summer", log_prefix('ActiveUser'))

        active_user_prob += winter_active_user_prob * season_fractions[seasons.winter.value]

        active_user_prob += transition_active_user_prob * season_fractions[seasons.transition.value]

        active_user_prob /= (season_fractions[seasons.winter.value] + season_fractions[seasons.transition.value])

        # As summer is not very high, use a static season fraction for summer

        active_user_prob = ((1 - summer_season_fraction_static) * active_user_prob) + \
                           (summer_season_fraction_static * summer_active_user_prob)
    else:

        # get weighted average of all seasons probability as summer season fraction is high

        logger.debug("%s Using all months for annual calculation", log_prefix('ActiveUser'))

        active_user_prob += winter_active_user_prob * season_fractions[seasons.winter.value]

        active_user_prob += transition_active_user_prob * season_fractions[seasons.transition.value]

        active_user_prob += summer_active_user_prob * season_fractions[seasons.summer.value]

    # Based on overall probability, judge whether a user is office goer based on threshold

    active_user_prob_threshold = active_user_config.get('active_user_prob_threshold')

    is_active_user = False

    if active_user_prob > active_user_prob_threshold:
        is_active_user = True

    active_user_prob_soft_margin = active_user_config.get('active_user_prob_soft_margin')

    # updating active user tag based on previous run output

    lifestyle_hsm = lifestyle_input_object.get('lifestyle_hsm')

    # fetching HSM info

    if (lifestyle_hsm is None) or (lifestyle_hsm.get('attributes') is None):
        logger.warning('Lifestyle HSM attributes are absent | ')

    else:
        if lifestyle_hsm.get('attributes').get('active_user_score') is None:
            logger.warning('Active user score attribute is missing in lifestyle HSM | ')
        else:
            hsm_in = lifestyle_hsm.get('attributes').get('active_user_score')[0]

            # checking if current run score is close to previous run score
            # if yes, the attribute tag is kept similar to previous run

            if is_active_user and (hsm_in <= active_user_prob_threshold) and ((active_user_prob-hsm_in) < active_user_prob_soft_margin):
                logger.info('Active user probability in HSM is | %s ', hsm_in)
                logger.info('Updating active user tag to | %s ', False)
                is_active_user = False
                active_user_prob = hsm_in

            elif (not is_active_user) and (hsm_in > active_user_prob_threshold) and ((hsm_in-active_user_prob) < active_user_prob_soft_margin):
                logger.info('Active user probability in HSM is | %s ', hsm_in)
                logger.info('Updating active user tag to | %s ', True)
                is_active_user = True
                active_user_prob = hsm_in

    # Get final Seasonal probabilites for debugging purposes

    active_user_seasonal_prob = {
        'winter': winter_active_user_prob,
        'summer': summer_active_user_prob,
        'transition': transition_active_user_prob
    }

    t_active_user_annual_end = datetime.now()

    logger.debug("%s Got annual active user probability in | %.3f s", log_prefix('ActiveUser'),
                 get_time_diff(t_active_user_annual_start, t_active_user_annual_end))

    return is_active_user, active_user_prob, active_user_seasonal_prob
