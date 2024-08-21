
"""
Author - Nisha Agarwal
Date - 8th Oct 20
Post processing steps for calculation of final sleeping hours of the user
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff
from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import fill_array
from python3.itemization.aer.functions.itemization_utils import get_index_array
from python3.itemization.aer.functions.itemization_utils import rolling_func

from python3.itemization.init_itemization_config import init_itemization_params

from python3.itemization.aer.behavioural_analysis.activity_profile.postprocess_for_active_hours import extend_inactive_segments

from python3.itemization.aer.behavioural_analysis.activity_profile.config.init_active_hours_config import init_active_hours_config
from python3.itemization.aer.behavioural_analysis.activity_profile.config.init_postprocess_active_hours_config import init_postprocess_active_hours_config


def get_sleep_hours(activity_curve, samples_per_hour, activity_seq, active_hours, logger_pass):

    """
        Post processing steps for calculation of final sleeping hours of the user

        Parameters:
            activity_curve             (np.ndarray)     : living load activity profile
            samples_per_hour           (int)            : samples in an hour
            activity_seq               (np.ndarray)     : labels of activity sequences of the user
            active_hours               (np.ndarray)     : active/nonactive mask array
            logger_pass                (dict)           : Contains the logger and the logging dictionary to be passed on

        Returns:
            active_hours               (np.ndarray)     : Updated active/nonactive mask array
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_sleep_time')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_activity_profile_start = datetime.now()

    seq_config = init_itemization_params().get('seq_config')

    sleep_hours_config = init_active_hours_config(samples_per_hour).get('sleep_time_config')

    # post process decision steps to fill mid night hours as inactive hours

    active_hours = fill_mid_night(activity_curve, samples_per_hour, activity_seq, active_hours, sleep_hours_config)

    logger.debug("Post processing done for mid night hours")

    limit = samples_per_hour/2

    # replace short duration active/inactive hours

    old_seq = find_seq(active_hours, np.zeros(len(activity_curve)), np.zeros(len(activity_curve)))

    old_seq[np.logical_and(old_seq[:, seq_config.get('label')]==0 ,
                           old_seq[:, seq_config.get('length')] <= limit), 0] = 1

    for i in range(len(old_seq)):
        active_hours = fill_array(active_hours, old_seq[i, 1], old_seq[i, seq_config.get('end')], old_seq[i, 0])

    old_seq = find_seq(active_hours, np.zeros(len(activity_curve)), np.zeros(len(active_hours)))

    old_seq[np.logical_and(old_seq[:, seq_config.get('label')] == 1 ,
                           old_seq[:, seq_config.get('length')] <= limit), 0] = 0

    for i in range(len(old_seq)):
        active_hours = \
            fill_array(active_hours, old_seq[i, seq_config.get('start')], old_seq[i, seq_config.get('end')], old_seq[i, 0])

    logger.debug("Filtered short duration active/inactive hours")

    # final checks to achieve 5 hours of minimum sleep time and one hours of minimum morning lighting

    active_hours = \
        get_final_sleep_time(activity_curve, samples_per_hour, activity_seq, active_hours, sleep_hours_config, logger)

    logger.debug("Final post processing steps done to ensure max/min limits of sleeping/morning hours")

    t_activity_profile_end = datetime.now()

    logger.info("Calculation of sleep time took | %.3f s",
                get_time_diff(t_activity_profile_start, t_activity_profile_end))

    return active_hours


def fill_mid_night(activity_curve, samples_per_hour, activity_seq, active_hours, config):

    """
     Post process step to find inactive hours during midnight

        Parameters:
            activity_curve             (np.ndarray)     : living load activity profile
            samples_per_hour           (int)            : samples in an hour
            activity_seq               (np.ndarray)     : labels of activity sequences of the user
            active_hours               (np.ndarray)     : active/nonactive mask array
            config                     (dict)           : sleep hours config

        Returns:
            sleep_hours                (np.ndarray)     : Modified sleep hours
    """

    # post process step to find inactive hours during midnight

    activity_curve_derivative = activity_curve - np.roll(activity_curve, 1)

    seq = find_seq(activity_seq, np.zeros(len(activity_curve)), np.zeros(len(activity_curve)))

    mid_night_start = config.get('mid_night_start')
    mid_night_end = config.get('mid_night_end')

    start_arr, end_arr, previous_val = get_seq_start_end_idxs_arrays(seq, activity_seq)

    mid_night_hours = np.arange(mid_night_start * samples_per_hour, mid_night_end * samples_per_hour)

    for i in mid_night_hours:

        active_hours = fill_mid_night_at_ts(i, active_hours, start_arr, end_arr, previous_val, activity_seq, activity_curve_derivative,
                                            config)

    active_hours[3*samples_per_hour : 4*samples_per_hour] = 0

    return active_hours


def fill_mid_night_at_ts(index, active_hours, start_arr, end_arr, previous_val, activity_seq, activity_curve_derivative, config):

    """
    Post process step to find inactive hours during midnight given time stamp

        Parameters:
            index                           (int)               : Time stamp index
            active_hours:                   (np.ndarray)        : Original sleep hours
            start_arr:                      (np.ndarray)        : start of activity seq array
            end_arr:                        (np.ndarray)        : end of activity seq array
            previous_val:                   (int)               : previous value on activity seq array
            activity_seq:                   (np.ndarray)        : activity seq array
            activity_curve_derivative:      (np.ndarray)        : actvity curve derivative
            config:                         (dict)              : config dictionary

        Returns:
            active_hours                    (np.ndarray)        : Modified sleep hours

    """

    mid_night_start = config.get('mid_night_start')
    mid_night_end = config.get('mid_night_end')

    samples_per_hour = len(active_hours) / Cgbdisagg.HRS_IN_DAY

    inactive_bool = get_inactive_seq_bool(active_hours, index, start_arr, end_arr, activity_seq, mid_night_start, mid_night_end)

    # Check if the seq id decreasing type and derivative is less than threshold, then mark it as inactive

    bool1 = (active_hours[index] and activity_seq[index] == -1) and \
            (np.abs(activity_curve_derivative[index]) < config.get('mid_night_derivative_limit'))

    # Check if the seq id constant type and the seq lies in the mid night region then mark it as inactive

    bool2 = active_hours[index] and activity_seq[index] == 0 and \
            not (start_arr[index] < (mid_night_end - 1) * samples_per_hour and
                 end_arr[index] <= (mid_night_end - 1) * samples_per_hour)

    mark_seq_inactive = bool1 or bool2

    # Ignoring sonar issue since the sequence of if/else statements is important

    inactive_bool_output = 0
    mark_seq_inactive_output = 0

    if inactive_bool:
        active_hours[index] = inactive_bool_output

    elif active_hours[index] and activity_seq[index] == -1 and previous_val[index] == 1:
        active_hours[index] = active_hours[index - 1]

    elif mark_seq_inactive:
        active_hours[index] = mark_seq_inactive_output

    elif active_hours[index] and activity_seq[index] == 0 and previous_val[index] == 1 and \
            (start_arr[index] < (mid_night_end - 1) * samples_per_hour and
             end_arr[index] < (mid_night_end - 1) * samples_per_hour):
        active_hours[index] = copy.deepcopy(active_hours[index - 1])

    return active_hours


def get_inactive_seq_bool(active_hours, index, start_arr, end_arr, activity_seq, mid_night_start, mid_night_end):

    """
    Bool to check whether the time is to be marked as inactive

        Parameters:
            active_hours                   (np.ndarray)        : Original sleep hours
            index                          (int)               : Time stamp index
            start_arr                      (np.ndarray)        : start of activity seq array
            end_arr                        (np.ndarray)        : end of activity seq array
            mid_night_start                (int)               : mid night start ts
            mid_night_end                  (int)               : mid night end ts

        Returns:
            inactive_bool                   (bool)              : calculated inactive seq bool

    """

    samples_per_hour = len(active_hours) / Cgbdisagg.HRS_IN_DAY

    # check if seq is increasing but the sequence lies in the mid night hours

    inactive_bool = active_hours[index] and activity_seq[index] == 1 and \
                    start_arr[index] < (mid_night_end - 1) * samples_per_hour and \
                    end_arr[index] <= (mid_night_end - 1) * samples_per_hour

    # check if seq is decreasing but the sequence lies in the mid night hours

    inactive_bool = inactive_bool or (active_hours[index] and activity_seq[index] == -1 and
                                      start_arr[index] > (mid_night_start + 1) * samples_per_hour and
                                      start_arr[index] < (mid_night_end - 1) * samples_per_hour and
                                      end_arr[index] <= (mid_night_end - 1) * samples_per_hour)

    return inactive_bool


def get_seq_start_end_idxs_arrays(seq, activity_seq):

    """
    Utils function to calculate start, end, and start of previous seq for all the calculated sequences

        Parameters:
            seq                         (np.ndarray)     : activity sequences
            activity_seq                (np.ndarray)     : labels of activity sequences of the user

        Returns:
            start_arr                   (np.ndarray)     : array of start index of all the sequences
            end_arr                     (np.ndarray)     : array of end index of all the sequences
            previous_val                (np.ndarray)     : array of label of previous seq all the sequences
    """

    seq_config = init_itemization_params().get('seq_config')

    start_arr = np.zeros(len(activity_seq))
    end_arr = np.zeros(len(activity_seq))
    previous_val = np.zeros(len(activity_seq))

    for i in range(len(seq)):

        index_array = get_index_array(seq[i, seq_config.get('start')], seq[i, seq_config.get('end')], len(start_arr))

        start_arr[index_array % len(start_arr)] = seq[i, seq_config.get('start')] % len(start_arr)
        end_arr[index_array % len(start_arr)] = (seq[i, seq_config.get('end')] - 1) % len(end_arr)
        previous_val[index_array % len(start_arr)] = seq[(i - 1) % len(seq), 0]

    return start_arr, end_arr, previous_val


def get_final_sleep_time(activity_curve, samples_per_hour, activity_seq, active_hours, config, logger):

    """
     Final checks to maximum/minimum of morning active and sleeping hours

        Parameters:
            activity_curve             (np.ndarray)     : living load activity profile
            samples_per_hour           (int)            : samples in an hour
            activity_seq               (np.ndarray)     : labels of activity sequences of the user
            active_hours               (np.ndarray)     : active/nonactive mask array
            config                     (dict)           : dict containing active hours config values
            logger                     (logger)         : logger object

        Returns:
            active_hours               (np.ndarray)     : Updated active/nonactive mask array
    """

    active_hours = max_morning_hours_sanity_checks(active_hours, samples_per_hour, config, logger)

    active_hours = min_morning_hours_sanity_check(active_hours, activity_curve, activity_seq, samples_per_hour, config, logger)

    active_hours = max_sleeping_hours_sanity_check(active_hours, samples_per_hour, config, logger)

    active_hours = min_sleeping_hours_sanity_check(active_hours, activity_curve, activity_seq, samples_per_hour,
                                                   config, logger)

    return active_hours


def max_morning_hours_sanity_checks(active_hours, samples_per_hour, config, logger):

    """
    Final checks to maximum/minimum of morning active

        Parameters:
            activity_curve             (np.ndarray)     : living load activity profile
            samples_per_hour           (int)            : samples in an hour
            active_hours               (np.ndarray)     : active/nonactive mask array
            config                     (dict)           : dict containing active hours config values
            logger                     (logger)         : logger object

        Returns:
            active_hours               (np.ndarray)     : Updated active/nonactive mask array
    """

    seq_config = init_itemization_params().get('seq_config')

    # achieve maximum of 3.5 hours of morning activity

    morning_hours = config.get('before_sunrise_activity_hours')

    morning_active_hours = active_hours[morning_hours]

    seq = find_seq(morning_active_hours, np.zeros(len(morning_active_hours)), np.zeros(len(morning_active_hours)),
                   overnight=False)

    if np.any(seq[:, seq_config.get('label')] == 1):

        seq_1 = seq[seq[:, seq_config.get('label')] == 1]

        if np.max(seq_1[:, seq_config.get('length')]) > config.get('max_morning_activity_hours') * samples_per_hour:

            score = config.get('before_sunrise_activity_score')

            logger.info("Maximum length morning active hours is absent")

            index = np.argmax(seq_1[:, seq_config.get('length')])

            diff = np.max(seq_1[:, seq_config.get('length')]) - config.get('max_morning_activity_hours') * samples_per_hour

            # add before sunrise activity score

            length = len(morning_active_hours)

            start = int(seq_1[index, seq_config.get('start')])
            end = int(seq_1[index, seq_config.get('end')])

            score = score[morning_hours]

            # remove morning activity using time of day score

            while diff > 0:

                if score[start % length] > score[end % length]:
                    morning_active_hours[start % length] = 0
                    start = start + 1
                    diff = diff - 1
                else:
                    morning_active_hours[end % length] = 0
                    end = end - 1
                    diff = diff - 1

            active_hours[morning_hours] = morning_active_hours

    return active_hours



def min_morning_hours_sanity_check(active_hours, activity_curve, activity_seq, samples_per_hour, config, logger):

    """
    Final checks to maximum/minimum of morning active

        Parameters:
            active_hours               (np.ndarray)     : active/nonactive mask array
            activity_curve             (np.ndarray)     : living load activity profile
            activity_seq               (np.ndarray)     : labels of activity sequences of the user
            samples_per_hour           (int)            : samples in an hour
            config                     (dict)           : dict containing active hours config values
            logger                     (logger)         : logger object

        Returns:
            active_hours               (np.ndarray)     : Updated active/nonactive mask array

    """

    activity_curve_abs_deri = np.abs(activity_curve - np.roll(activity_curve, 1))

    # achieve minimum of 1 hour of morning lighting activity

    morning_active_hours = config.get("morning_activity_hours")

    if np.all(active_hours[morning_active_hours] == 0) and \
        not np.all(activity_curve_abs_deri[morning_active_hours] < 0.01):

        logger.info("Minimum length morning active hours is absent")

        score = np.zeros(len(activity_curve))

        weights = config.get('weights')

        morning_active_hours = morning_active_hours.astype(int)

        for i in morning_active_hours:
            score[i] = score[i] + weights[int(activity_seq[i])]
            score[i] = score[i] + activity_curve_abs_deri[i]

        target_length = config.get('min_morning_activity_hours') * samples_per_hour

        max_val = 0
        start = morning_active_hours[0]

        for i in morning_active_hours:
            if np.sum(score[i:i + target_length]) > max_val:
                start = i
                max_val = np.sum(score[i:i + target_length])

        active_hours[start: start+target_length] = 1

    return active_hours


def max_sleeping_hours_sanity_check(active_hours, samples_per_hour, config, logger):

    """
    Final checks to maximum/minimum of sleeping hours

        Parameters:
            active_hours               (np.ndarray)     : active/nonactive mask array
            samples_per_hour           (int)            : samples in an hour
            config                     (dict)           : dict containing active hours config values
            logger                     (logger)         : logger object

        Returns:
            active_hours               (np.ndarray)     : Updated active/nonactive mask array

    """

    # achieve maximum of 11 hours of sleeping time

    seq_config = init_itemization_params().get('seq_config')

    probable_sleeping_hours2 = np.arange(20*samples_per_hour, 33*samples_per_hour + 1) % len(active_hours)

    sleeping_hours = active_hours[probable_sleeping_hours2]

    seq = find_seq(sleeping_hours, np.zeros(len(sleeping_hours)), np.zeros(len(sleeping_hours)))

    seq_0 = seq[seq[:, 0] == 0]

    if not np.all(sleeping_hours == 0) and np.max(seq_0[:, seq_config.get('length')]) > config.get('max_sleeping_hours') * samples_per_hour:

        logger.info("Maximum length sleeping hours is absent")

        score = config.get('max_sleeping_hours_score')

        index = np.argmax(seq_0[:, seq_config.get('length')])

        diff = np.max(seq_0[:, seq_config.get('length')]) - config.get('max_sleeping_hours') * samples_per_hour

        length = len(sleeping_hours)

        start = int(seq_0[index, seq_config.get('start')] + 1)
        end = int(seq_0[index, seq_config.get('end')] - 1)

        score = score[probable_sleeping_hours2]

        # remove sleep times  using time of day score

        while diff > 0:

            if score[start % length] > score[end % length]:
                sleeping_hours[start % length] = 1
                start = start + 1
                diff = diff - 1
            else:
                sleeping_hours[end % length] = 1
                end = end - 1
                diff = diff - 1

    active_hours[probable_sleeping_hours2] = sleeping_hours

    return active_hours


def min_sleeping_hours_sanity_check(active_hours, activity_curve, activity_seq, samples_per_hour,
                                    config, logger):

    """
    Final checks to maximum/minimum of sleeping hours

        Parameters:
            active_hours               (np.ndarray)     : active/nonactive mask array
            activity_curve             (np.ndarray)     : living load activity profile
            activity_seq               (np.ndarray)     : labels of activity sequences of the user
            samples_per_hour           (int)            : samples in an hour
            config                     (dict)           : dict containing active hours config values
            logger                     (logger)         : logger object

        Returns:
            active_hours               (np.ndarray)     : Updated active/nonactive mask array

    """

    probable_sleeping_hours = config.get('probable_sleeping_hours2')

    seq_config = init_itemization_params().get('seq_config')

    # achieve minimum of 5 hours of sleeping time

    sleeping_hours = copy.deepcopy(active_hours[probable_sleeping_hours])
    sleeping_hours = copy.deepcopy(sleeping_hours)

    seq = find_seq(sleeping_hours, np.zeros(len(sleeping_hours)), np.zeros(len(sleeping_hours)), overnight=False)

    flag = 0

    if np.all(seq[:, seq_config.get('label')] == 1):
        flag = 1

    else:
        seq_0 = seq[seq[:, seq_config.get('label')] == 0]
        if np.max(seq_0[:, seq_config.get('length')]) < config.get('min_sleeping_hours') * samples_per_hour:
            flag = 1

    if flag:
        logger.info("Minimum length sleeping hours is absent")

        score = config.get('min_sleeping_hours_score')

        score[active_hours == 0] = 2 + score[active_hours == 0]

        score = score[probable_sleeping_hours]

        length = config.get('min_sleeping_hours') * samples_per_hour

        total_score = np.roll(rolling_func(score, length / 2, 0), int(-length / 2), 0)

        start = np.argmax(total_score)

        index_array = get_index_array(start, start + length, len(score))

        active_hours = 1 - active_hours
        max_length = config.get('min_sleeping_hours') * samples_per_hour

        active_hours = extend_inactive_segments(active_hours, index_array, activity_curve, activity_seq, max_length,
                                                samples_per_hour,
                                                init_postprocess_active_hours_config(activity_curve,
                                                                                     np.zeros(activity_curve.shape),
                                                                                     np.zeros(activity_curve.shape),
                                                                                     samples_per_hour).get("postprocess_active_hour_config"), flag=True)
        active_hours = 1 - active_hours

        sleeping_hours[index_array] = 0

        active_hours[probable_sleeping_hours] = sleeping_hours

    return active_hours
