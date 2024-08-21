"""
Author - Prasoon Patidar
Date - 16th June 2020
get Wake Up and sleep times for event level data
"""

# import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.logs_utils import log_prefix
from python3.utils.time.get_time_diff import get_time_diff


def get_wakeup_attributes(lifestyle_input_object, input_data, lighting_tou, wakeup_sleep_config, logger_pass):

    """

    Fetch Wakeup time and calculate related attributes

    Parameters:
        lifestyle_input_object (dict)              : Dictionary containing all inputs for lifestyle modules
        input_data (np.ndarray)                    : 15 col raw data required for lifestyle module
        lighting_tou(np.ndarray)                   : TOU array for given input_data range
        wakeup_sleep_config(dict)                  : static config for wakeup/sleep times
        logger_pass(dict)                          : Contains base logger and logging dictionary

    Returns:
        wakeup_attributes(dict)                    : Dictionary containing all wakeup regarding outputs
    """

    t_wakeup_attributes_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_wakeup_attributes')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Wakeup Attributes Detection", log_prefix('WakeUpTime'))

    # Initialize wake up attribute dict

    wakeup_attributes = dict()

    start_hour, end_hour = 0, Cgbdisagg.HRS_IN_DAY - 1

    # get sunrise time in hours

    sunrise_times = (input_data[input_data[:, Cgbdisagg.INPUT_HOD_IDX] == start_hour, Cgbdisagg.INPUT_SUNRISE_IDX] -
                     input_data[input_data[:, Cgbdisagg.INPUT_HOD_IDX] == start_hour, Cgbdisagg.INPUT_DAY_IDX]) /\
                    Cgbdisagg.SEC_IN_HOUR

    # filter abnormal sunrise values based on hour of day

    sunrise_start_times_limit = wakeup_sleep_config.get('sunrise_start_hour_limit')

    sunrise_end_times_limit = wakeup_sleep_config.get('sunrise_end_hour_limit')

    allowed_sunrise_deviation = wakeup_sleep_config.get('allowed_sunrise_deviation')

    sunrise_times = sunrise_times[
        (sunrise_times > sunrise_start_times_limit) & (sunrise_times < sunrise_end_times_limit)]

    # Get sunrise band (5-95 percentile of sunrise hour) for given time period if array is not empty

    if len(sunrise_times) > 0:

        sunrise_band_start, sunrise_band_end = np.percentile(sunrise_times, [5, 95])

        sunrise_band_start -= allowed_sunrise_deviation

        sunrise_band_end += allowed_sunrise_deviation

    else:

        logger.debug("%s Valid Sunrise times not found, Using complete sunrise band", log_prefix('WakeUpTime'))

        sunrise_band_start, sunrise_band_end = sunrise_start_times_limit, sunrise_end_times_limit

    logger.debug("%s Sunrise Band: %s - %s", log_prefix('WakeUpTime'), str(sunrise_band_start), str(sunrise_band_end))

    # Check if lighting tou is of right size

    if not lighting_tou.shape[0] == Cgbdisagg.HRS_IN_DAY:
        logger.info("%s lighting time of usage reported incorrectly, unable to get wakeup attributes", log_prefix('WakeUpTime'))

        return None

    # get lighting band start and end times based on tou

    lighting_tou[lighting_tou > 0] = 1

    lighting_start_times = np.where(lighting_tou[:-1] < lighting_tou[1:])[0]

    lighting_end_times = np.where(lighting_tou[:-1] > lighting_tou[1:])[0]

    # update start/end times if start/end hours included in lighting

    if lighting_tou[start_hour] > 0:
        lighting_start_times = np.insert(lighting_start_times, 0, start_hour)

    if lighting_tou[end_hour] > 0:
        lighting_end_times = np.insert(lighting_end_times, len(lighting_end_times), end_hour)

    logger.debug("%s Lighting Bands start/end times:%s - %s",
                 log_prefix('WakeUpTime'), str(lighting_start_times), str(lighting_end_times))

    # Get lighting bands which fall into sunrise band

    buffer_hours_after_sunrise = 3

    li_morning_idx = np.where((lighting_end_times >= sunrise_band_start) &
                              (lighting_start_times <= sunrise_band_end+buffer_hours_after_sunrise))[0]

    if li_morning_idx.shape[0] == 0:
        logger.info("%s Unable to get wakeup times, No lighting bands found during morning times", log_prefix('WakeUpTime'))

        return None

    # Get wakeup band start time based on first li morning band

    wakeup_band_start_time = lighting_start_times[li_morning_idx][0]

    # Check if wakeup band start time is beyond valid range

    wakeup_band_start_time_min = wakeup_sleep_config.get('wakeup_band_start_hour_min')

    if wakeup_band_start_time < wakeup_band_start_time_min:

        logger.info("%s Unable to get wakeup times, wakeup Band Start Hour is less than allowed hour", log_prefix('WakeUpTime'))

        logger.info("%s lighting band found: %s-%s hrs",
                    log_prefix('WakeUpTime'), str(lighting_start_times[li_morning_idx][0]), str(lighting_end_times[li_morning_idx][0]))

        wakeup_attributes = None

    else:

        # Get wakeup end time based on last morning li band

        wakeup_band_end_time = lighting_end_times[li_morning_idx][-1]

        logger.debug("%s Wakeup band start/end times:%s - %s",
                     log_prefix('WakeUpTime'), str(round(wakeup_band_start_time, 2)), str(round(wakeup_band_end_time, 2)))

        # get confidence in wakeup time based on length of wakeup band

        confidence_band_length_ratio = wakeup_sleep_config.get('confidence_band_length_ratio')

        samples_per_hour = int(Cgbdisagg.SEC_IN_HOUR / lifestyle_input_object.get('sampling_rate'))

        wakeup_band_length = (wakeup_band_end_time - wakeup_band_start_time) / samples_per_hour

        wakeup_confidence = confidence_band_length_ratio * wakeup_band_length

        # get wakeup time based on confidence value

        wakeup_time = wakeup_band_start_time + wakeup_confidence

        #TODO(Nisha): quickfix
        # lighting is running throughout the night, probably over estimation due to heating,
        # Thus, using end of wakeup band to detect wakeup time, not logically consistent for all cases

        logger.debug("%s Wakeup time:%s +/- %s hrs",
                     log_prefix('WakeUpTime'), str(round(wakeup_time, 2)), str(round(wakeup_confidence, 2)))

        # fill wakeup attributes in dictionary

        wakeup_attributes.update({
            'wakeup_time'      : lifestyle_input_object.get('behavioural_profile').get('wakeup_time'),
            'wakeup_confidence': wakeup_confidence
        })

        t_wakeup_attributes_end = datetime.now()

        logger.info("%s Got wakeup attributes in | %.3f s", log_prefix('WakeUpTime'),
                    get_time_diff(t_wakeup_attributes_start,
                                  t_wakeup_attributes_end))

    return wakeup_attributes


def get_sleep_attributes(lifestyle_input_object, input_data, lighting_tou, wakeup_sleep_config, logger_pass):

    """
    Parameters:

    Fetch Sleep time and calculate related attributes

        lifestyle_input_object (dict)              : Dictionary containing all inputs for lifestyle modules
        input_data (np.ndarray)                    : 15 col raw data required for lifestyle module
        lighting_tou(np.ndarray)                   : TOU array for given input_data range
        wakeup_sleep_config(dict)                  : static config for wakeup/sleep times
        logger_pass(dict)                          : Contains base logger and logging dictionary

    Returns:
        sleep_attributes(dict)                    : Dictionary containing all sleep regarding outputs
    """

    t_sleep_attributes_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_sleep_attributes')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Sleep Attributes Detection", log_prefix('SleepTime'))

    # Initialize wake up attribute dict

    sleep_attributes = dict()

    start_hour, end_hour = 0, Cgbdisagg.HRS_IN_DAY - 1

    # get sunrise time in hours

    sunset_times = (input_data[input_data[:, Cgbdisagg.INPUT_HOD_IDX] == start_hour, Cgbdisagg.INPUT_SUNSET_IDX] -
                    input_data[input_data[:, Cgbdisagg.INPUT_HOD_IDX] == start_hour, Cgbdisagg.INPUT_DAY_IDX]) /\
                   Cgbdisagg.SEC_IN_HOUR

    # filter abnormal sunset values based on hour of day

    sunset_start_times_limit = wakeup_sleep_config.get('sunset_start_hour_limit')

    sunset_end_times_limit = wakeup_sleep_config.get('sunset_end_hour_limit')

    sunset_times = sunset_times[
        (sunset_times > sunset_start_times_limit) & (sunset_times < sunset_end_times_limit)]

    # Get sunset band (5-95 percentile of sunset hour) for given time period if sunset times is not empty

    if len(sunset_times) > 0:

        sunset_band_start, sunset_band_end = np.percentile(sunset_times, [5, 95])

    else:

        logger.debug("%s Valid Sunset times not found, Using complete sunset band", log_prefix('SleepTime'))

        sunset_band_start, sunset_band_end = sunset_start_times_limit, sunset_end_times_limit

    logger.debug("%s Sunset Band: %s - %s", log_prefix('SleepTime'), str(sunset_band_start), str(sunset_band_end))

    # check if lighting tou array is of right size
    if not lighting_tou.shape[0] == Cgbdisagg.HRS_IN_DAY:
        logger.info("%s lighting time of usage reported incorrectly, unable to get sleep attributes", log_prefix('SleepTime'))

        return None

    # get lighting band start and end times based on tou

    lighting_tou[lighting_tou > 0] = 1

    lighting_start_times = np.where(lighting_tou[:-1] < lighting_tou[1:])[0]

    lighting_end_times = np.where(lighting_tou[:-1] > lighting_tou[1:])[0]

    # update start/end times if start/end hours included in lighting

    if lighting_tou[start_hour] > 0:
        lighting_start_times = np.insert(lighting_start_times, 0, 0)

    if lighting_tou[end_hour] > 0:
        lighting_end_times = np.insert(lighting_end_times, len(lighting_end_times), end_hour)

    logger.debug("%s Lighting Bands start/end times:%s - %s",
                 log_prefix('SleepTime'), str(lighting_start_times), str(lighting_end_times))

    # Get lighting bands which fall into sunset band

    sleep_buffer_hours = 3

    li_evening_idx = np.where(lighting_end_times > (sunset_band_end-sleep_buffer_hours))[0]

    if li_evening_idx.shape[0] == 0:
        logger.info("%s Unable to get sleep times, No lighting bands found during evening times", log_prefix('SleepTime'))

        return None

    # Get sleep band start time based on last li evening band

    sleep_band_start_time = max(lighting_start_times[li_evening_idx][-1], sunset_band_end)

    # Get sleep end time based on last evening li band

    sleep_band_end_time = lighting_end_times[li_evening_idx][-1]

    logger.debug("%s Sleep band start/end times:%s - %s",
                 log_prefix('SleepTime'), str(round(sleep_band_start_time, 2)), str(round(sleep_band_end_time, 2)))

    # get confidence in sleep time based on length of sleep band

    confidence_band_length_ratio = wakeup_sleep_config.get('confidence_band_length_ratio')

    samples_per_hour = int(Cgbdisagg.SEC_IN_HOUR / lifestyle_input_object.get('sampling_rate'))

    sleep_band_length = (sleep_band_end_time - sleep_band_start_time) / samples_per_hour

    sleep_confidence = confidence_band_length_ratio * sleep_band_length

    # get sleep time based on confidence value

    sleep_time = sleep_band_end_time - sleep_confidence

    logger.debug("%s Sleep time:%s +/- %s hrs",
                 log_prefix('SleepTime'), str(round(sleep_time, 2)), str(round(sleep_confidence, 2)))

    # fill sleep attributes in dictionary

    sleep_attributes.update({
        'sleep_time'      : lifestyle_input_object.get('behavioural_profile').get('sleep_time'),
        'sleep_confidence': sleep_confidence
    })

    t_sleep_attributes_end = datetime.now()

    logger.info("%s Got sleep attributes in | %.3f s", log_prefix('SleepTime'),
                get_time_diff(t_sleep_attributes_start,
                              t_sleep_attributes_end))

    return sleep_attributes
