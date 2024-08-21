"""
Author - Prasoon Patidar
Date - 18th June 2020
Lifestyle Submodule to calculate weekend warrior probability
"""

# import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.logs_utils import log_prefix
from python3.utils.time.get_time_diff import get_time_diff
from python3.analytics.lifestyle.functions.zscore_peak_detection import zscore_peak_detection


def get_weekend_warrior_probability(input_data, lifestyle_input_object, logger_pass):

    """
    Parameters:
        input_data (np.ndarray)                    : custom trimmed input data in 15 col matrix
        lifestyle_input_object(dict)               : dictionary containing inputs for lifestyle modules
        logger_pass(dict)                          : contains base logger and logging dictionary
    Returns:
        weekend_warrior_prob(float)                : probability of being a weekend warrior given input data
        debug(dict)                                : step wise info for debugging and plotting purposes
    """

    t_weekend_warrior_module_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_weekend_warrior_probability')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Weekend Warrior Probability", log_prefix('WeekendWarrior'))

    # Initialize weekend_warrior_probability, and debug object

    weekend_warrior_prob = None

    debug = dict()

    # Get weekend warrior config from lifestyle input object

    sampling_rate = lifestyle_input_object.get('sampling_rate')

    samples_per_hour = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    weekend_warrior_config = lifestyle_input_object.get('weekend_warrior_config')

    weekday_idx = lifestyle_input_object.get('WEEKDAY_IDX')

    peak_detection_config = lifestyle_input_object.get('peak_detection_config')

    lag_idx, threshold_idx, influence_idx = map(peak_detection_config.get,
                                                ['LAG_IDX', 'THRESHOLD_IDX', 'INFLUENCE_IDX'])

    diff_peak_params = weekend_warrior_config.get('WEEKEND_WARRIOR_PEAKS')

    # Get weekend warrior input data from lifestyle input object

    weekend_warrior_input_dict = lifestyle_input_object.get('weekend_warrior_input_dict')

    day_input_idx = weekend_warrior_input_dict.get('sample_rate_day_idx')

    day_input_data = weekend_warrior_input_dict.get('sample_rate_day_data')

    # Get day epoch for weekday in input data

    weekdays_epoch = np.unique(input_data[input_data[:, weekday_idx] == True, Cgbdisagg.INPUT_DAY_IDX])

    weekend_epoch = np.unique(input_data[input_data[:, weekday_idx] == False, Cgbdisagg.INPUT_DAY_IDX])

    # get Day data for weekend and weekdays from lifestyle input object

    weekday_days_idx = np.isin(day_input_idx, weekdays_epoch).nonzero()[0]
    weekday_data = day_input_data[weekday_days_idx, :]

    weekend_days_idx = np.isin(day_input_idx, weekend_epoch).nonzero()[0]
    weekend_data = day_input_data[weekend_days_idx, :]

    # Return if there is no weekend/weekday data present

    if weekend_data.shape[0] == 0:
        logger.warning("%s Weekend Data not available, Unable to get weekend warrior probability", log_prefix('WeekendWarrior'))

        return weekend_warrior_prob, debug

    if weekday_data.shape[0] == 0:
        logger.warning("%s Weekday Data not available, Unable to get weekend warrior probability", log_prefix('WeekendWarrior'))

        return weekend_warrior_prob, debug

    # remove any nan vals from weekday and weekend data

    weekday_data[np.isnan(weekday_data)] = 0

    weekend_data[np.isnan(weekend_data)] = 0

    # Get mean hourly values for weekday and weekend data

    weekday_hourly_mean_copy = np.mean(weekday_data, axis=0)
    weekend_hourly_mean_copy = np.mean(weekend_data, axis=0)

    weekend_hourly_mean = np.zeros_like(weekend_hourly_mean_copy)
    weekday_hourly_mean = np.zeros_like(weekday_hourly_mean_copy)

    window_size = int((weekend_hourly_mean_copy.shape[0]/Cgbdisagg.HRS_IN_DAY)/ 2)

    for i in range(window_size, len(weekend_hourly_mean) - window_size):
        weekend_hourly_mean[i] = np.mean(weekend_hourly_mean_copy[i - window_size:i + window_size + 1])
        weekday_hourly_mean[i] = np.mean(weekday_hourly_mean_copy[i - window_size:i + window_size + 1])

    weekend_hourly_mean[:window_size + 1] = weekend_hourly_mean_copy[:window_size + 1]
    weekend_hourly_mean[(len(weekend_hourly_mean) - window_size):] = weekend_hourly_mean_copy[(len(weekend_hourly_mean) - window_size):]

    weekday_hourly_mean[:window_size + 1] = weekday_hourly_mean_copy[:window_size + 1]
    weekday_hourly_mean[(len(weekend_hourly_mean) - window_size):] = weekday_hourly_mean_copy[(len(weekend_hourly_mean) - window_size):]

    # Difference in mean hourly values biased towards weekends

    diff_hourly_mean = np.maximum(weekend_hourly_mean - weekday_hourly_mean, 0)

    # Get fractional difference in mean hourly values from weekend values

    diff_hourly_mean_fraction = diff_hourly_mean / weekend_hourly_mean

    # get peak signals for difference and fractional difference

    diff_hourly_mean_peaks = zscore_peak_detection(diff_hourly_mean,
                                                   diff_peak_params[lag_idx],
                                                   diff_peak_params[threshold_idx],
                                                   diff_peak_params[influence_idx])

    diff_hourly_mean_peak_signal = diff_hourly_mean_peaks.signal

    diff_hourly_mean_fraction_peaks = zscore_peak_detection(diff_hourly_mean_fraction,
                                                            diff_peak_params[lag_idx],
                                                            diff_peak_params[threshold_idx],
                                                            diff_peak_params[influence_idx])

    diff_hourly_mean_fraction_peak_signal = diff_hourly_mean_fraction_peaks.signal

    # add strength to peaks based on median of peak start- peak end
    # TODO(Nisha): improve this part of code

    start_idx = -1
    for idx in range(len(diff_hourly_mean_peak_signal)):
        if (start_idx < 0) & (diff_hourly_mean_peak_signal[idx] > 0):
            start_idx = idx
        elif (start_idx < 0) & (diff_hourly_mean_peak_signal[idx] == 0):
            continue
        elif (start_idx >= 0) & (diff_hourly_mean_peak_signal[idx] == 0):
            diff_hourly_mean_peak_signal[start_idx:idx + 1] = np.median(diff_hourly_mean[start_idx:idx + 1])
            start_idx = -1

    if (start_idx >= 0):
        diff_hourly_mean_peak_signal[start_idx:] = np.median(diff_hourly_mean[start_idx:])

    start_idx = -1
    for idx in range(len(diff_hourly_mean_fraction_peak_signal)):
        if (start_idx < 0) & (diff_hourly_mean_fraction_peak_signal[idx] > 0):
            start_idx = idx
        elif (start_idx < 0) & (diff_hourly_mean_fraction_peak_signal[idx] == 0):
            continue
        elif (start_idx >= 0) & (diff_hourly_mean_fraction_peak_signal[idx] == 0):
            diff_hourly_mean_fraction_peak_signal[start_idx:idx + 1] =\
                np.median(diff_hourly_mean_fraction[start_idx:idx + 1])
            start_idx = -1
    if (start_idx >= 0):
        diff_hourly_mean_fraction_peak_signal[start_idx:] = np.median(diff_hourly_mean_fraction[start_idx:])

    # Get Weekend Warrior Probability(using diff_hourly_mean_fraction_peak_signal)

    day_start_hour = weekend_warrior_config.get('day_start_hour')

    day_end_hour = weekend_warrior_config.get('day_end_hour')

    max_consumption_difference_ratio = weekend_warrior_config.get('MAX_CONSUMPTION_DIFFERENCE_RATIO')

    area_max_fraction = weekend_warrior_config.get('MAX_PEAK_NORMED_AREA_FRACTION')

    day_peak_area_total = \
        np.sum(diff_hourly_mean_fraction_peak_signal[(day_start_hour - 1) * samples_per_hour:day_end_hour * samples_per_hour])

    day_peak_area_max = max_consumption_difference_ratio * (day_end_hour - day_start_hour) * samples_per_hour

    logger.debug("%s Diff Peak Area Total: %s, Max: %s",
                 log_prefix('WeekendWarrior'), str(round(day_peak_area_total,2)), str(round(day_peak_area_max,2)))

    weekend_warrior_score = round(day_peak_area_total / (area_max_fraction * day_peak_area_max), 2)

    weekend_warrior_prob = min(weekend_warrior_score, 1)

    # add information in debug object

    debug.update({
        'weekday_hourly_mean': weekday_hourly_mean,
        'weekend_hourly_mean': weekend_hourly_mean,
        'diff_hourly_mean': diff_hourly_mean,
        'diff_hourly_mean_fraction': diff_hourly_mean_fraction,
        'diff_hourly_mean_peak_signal': diff_hourly_mean_peak_signal,
        'diff_hourly_mean_fraction_peak_signal': diff_hourly_mean_fraction_peak_signal,
    })

    t_weekend_warrior_module_end = datetime.now()

    logger.debug("%s Got Weekend Warrior probability in | %.3f s", log_prefix('WeekendWarrior'),
                 get_time_diff(t_weekend_warrior_module_start,
                               t_weekend_warrior_module_end))

    return weekend_warrior_prob, debug


def get_weekend_warrior_annual_prob(lifestyle_input_object, lifestyle_output_object, logger_pass):

    """
    Parameters:
        lifestyle_input_object(dict)               : dictionary containing inputs for lifestyle modules
        lifestyle_output_object(dict)              : dictionary containing step wise outputs for lifestyle modules
        logger_pass(dict)                          : contains base logger and logging dictionary
    Returns:
        is_weekend_warrior(float)                      : Whether this user is an weekend warrior or not
        weekend_warrior_prob(float)                    : probability of this user being a weekend warrior
        weekend_warrior_seasonal_probability           : Season wise probability of user being a weekend warrior
    """

    t_weekend_warrior_annual_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('weekend_warrior_annual_probability')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Weekend Warrior Annual probability", log_prefix('WeekendWarrior'))

    # get relevant inputs for weekend warrior probability

    seasons = lifestyle_input_object.get('weather_config').get('season')

    weekend_warrior_config = lifestyle_input_object.get('weekend_warrior_config')

    season_fractions = lifestyle_output_object.get('annual').get('season_fraction')

    weekend_warrior_seasonal_prob = np.zeros(seasons.count.value)

    # Get winter season relevant features

    weekend_warrior_seasonal_prob[seasons.winter.value] =\
        lifestyle_output_object.get('season').get(seasons.winter.name).get('weekend_warrior_prob', 0)

    # Get Transition relevant features

    weekend_warrior_seasonal_prob[seasons.transition.value] =\
        lifestyle_output_object.get('season').get(seasons.transition.name).get('weekend_warrior_prob', 0)

    # Get summer relevant features

    weekend_warrior_seasonal_prob[seasons.summer.value] = \
        lifestyle_output_object.get('season').get(seasons.summer.name).get('weekend_warrior_prob', 0)

    logger.debug("%s seasonal probabilities: %s", log_prefix('WeekendWarrior'), str(weekend_warrior_seasonal_prob))

    # loop over seasons to get weekend warrior prob

    season_fraction_sum_min_threshold = weekend_warrior_config.get('MIN_SEASON_FRACTION')

    weekend_warrior_score = 0.

    season_fraction_sum = 0.

    for season_id in np.argsort(weekend_warrior_seasonal_prob)[::-1]:

        # loop over from max to min weekend warrior prob until season fractions > threshold

        logger.debug("%s including Season %s for probability calculations",
                     log_prefix('WeekendWarrior'), seasons(season_id).name)

        weekend_warrior_score += season_fractions[season_id] * weekend_warrior_seasonal_prob[season_id]

        season_fraction_sum += season_fractions[season_id]

        if season_fraction_sum > season_fraction_sum_min_threshold:
            break

    # get weekend warrior probability

    weekend_warrior_prob = weekend_warrior_score / season_fraction_sum

    # Based on overall probability, judge whether a user is office goer based on threshold

    weekend_warrior_prob_threshold = weekend_warrior_config.get('weekend_warrior_prob_threshold')

    is_weekend_warrior = False

    if weekend_warrior_prob > weekend_warrior_prob_threshold:
        is_weekend_warrior = True

    weekend_warrior_soft_margin = weekend_warrior_config.get('weekend_warrior_soft_margin')

    # updating weekend warrior tag based on previous run ouput

    lifestyle_hsm = lifestyle_input_object.get('lifestyle_hsm')

    # fetching HSM info

    if (lifestyle_hsm is None) or (lifestyle_hsm.get('attributes') is None):
        logger.warning('Lifestyle HSM attributes are absent | ')

    else:
        if lifestyle_hsm.get('attributes').get('weekend_warrior_score') is None:
            logger.warning('weekend warrior score attribute is missing in lifestyle HSM | ')
        else:
            hsm_in = lifestyle_hsm.get('attributes').get('weekend_warrior_score')[0]

            # checking if current run score is close to previous run score
            # if yes, the attribute tag is kept similar to previous run

            if is_weekend_warrior and (hsm_in <= weekend_warrior_prob_threshold) and \
                    ((weekend_warrior_prob - hsm_in) < weekend_warrior_soft_margin):
                logger.info('Weekend warrior probability in HSM is | %s ', hsm_in)
                logger.info('Updating Weekend warrior tag to | %s ', False)
                is_weekend_warrior = False
                weekend_warrior_prob = hsm_in

            elif (not is_weekend_warrior) and (hsm_in > weekend_warrior_prob_threshold) and \
                    ((hsm_in - weekend_warrior_prob) < weekend_warrior_soft_margin):
                logger.info('Weekend warrior probability in HSM is | %s ', hsm_in)
                logger.info('Updating Weekend warrior tag to | %s ', True)
                is_weekend_warrior = True
                weekend_warrior_prob = hsm_in

    # Get final Seasonal probabilites for debugging purposes

    weekend_warrior_seasonal_prob = {
        'winter': weekend_warrior_seasonal_prob[seasons.winter.value],
        'summer': weekend_warrior_seasonal_prob[seasons.summer.value],
        'transition': weekend_warrior_seasonal_prob[seasons.transition.value]
    }

    t_weekend_warrior_annual_end = datetime.now()

    logger.debug("%s Got annual weekend warrior probability in | %.3f s", log_prefix('WeekendWarrior'),
                 get_time_diff(t_weekend_warrior_annual_start,
                               t_weekend_warrior_annual_end))

    return is_weekend_warrior, weekend_warrior_prob, weekend_warrior_seasonal_prob
