
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Perform weekday/weekend analysis
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.itemization_utils import find_seq

from python3.itemization.aer.behavioural_analysis.home_profile.config.weekend_weekday_profile_config import get_weekday_weekend_profile_config


def weekend_analysis(item_input_object, item_output_object, logger_pass):

    """
    Perform weekday/weekend analysis

    Parameters:
        item_input_object         (dict)           : Dict containing all hybrid inputs
        item_output_object        (dict)           : Dict containing all hybrid outputs
        logger_pass                 (dict)           : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_input_object         (dict)           : Dict containing all hybrid inputs
        item_output_object        (dict)           : Dict containing all hybrid outputs
    """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('weekend_analysis')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    occupants_profile = item_output_object.get("occupants_profile")
    samples_per_hour = item_input_object.get("item_input_params").get('samples_per_hour')
    weekday_curve = item_input_object.get("weekday_activity_curve")
    weekend_curve = item_input_object.get("weekend_activity_curve")

    weekday_profile = dict()

    # Initialize default profile

    weekday_profile.update({
        'weekend_mid_day_present': 0,
        'weekend_mid_day_absent': 0,
        'weekend_morning_present': 0,
        'weekend_morning_absent': 0,
        'weekend_night_present': 0,
        'weekend_night_absent': 0,
        'weekend_eve_activity_present': 0,
        'weekend_eve_activity_absent': 0,
        'weekend_laundry': 0,
        "weekend_early_eve_present": 0,
        "weekend_early_eve_absent": 0,
        "weekend_breakfast_absent": 0,
        "weekend_lunch_absent": 0,
        "weekend_dinner_absent": 0,
        "weekend_hours": 0
    })

    weekday_weekend_diff = np.percentile(weekday_curve, 97) - np.percentile(weekday_curve, 3)

    config = get_weekday_weekend_profile_config(samples_per_hour, weekday_weekend_diff)

    profile_diff_thres = config.get('profile_diff_thres')
    late_night_hours = config.get('late_night_hours')
    morn_act_hours = config.get('morn_act_hours')

    seq_label = seq_config.SEQ_LABEL
    seq_len = seq_config.SEQ_LEN

    # checking diff between weekday and weekend activity curve, and whether the difference is greater than certain threshold

    hours = config.get("hours_bucket")

    weekend_weekday_diff = weekend_curve - weekday_curve
    weekend_weekday_diff[late_night_hours] = 0

    weekday_weekend_curve_thres = np.ones(len(weekend_weekday_diff)) * profile_diff_thres
    weekday_weekend_curve_thres[morn_act_hours] = profile_diff_thres / 2

    weekend_weekday_diff[np.abs(weekend_weekday_diff) < weekday_weekend_curve_thres] = 0

    weekend_weekday_diff = np.sign(weekend_weekday_diff)

    samples_thres = int(Cgbdisagg.SEC_IN_HOUR / Cgbdisagg.SEC_IN_15_MIN)

    # Smooth the difference for 15 min sampling rate data
    # incase weekend activity is detected just of a 15 min slot

    if samples_per_hour >= samples_thres:
        for i in range(len(weekend_weekday_diff)):

            single_epoch_weekend_act_inc_or_dec_detected = \
                weekend_weekday_diff[(i - 1) % len(weekend_weekday_diff)] == weekend_weekday_diff[(i + 1) % len(weekend_weekday_diff)]

            if single_epoch_weekend_act_inc_or_dec_detected:
                weekend_weekday_diff[i % len(weekend_weekday_diff)] = \
                    weekend_weekday_diff[(i - 1) % len(weekend_weekday_diff)]

    logger.debug("Calculation of weekend/weekday activity differnce done")

    # Find chunks of weekday - weekend activity difference

    weekday_weekend_diff_seq = find_seq(weekend_weekday_diff, np.zeros(len(weekend_weekday_diff)), np.zeros(len(weekend_weekday_diff)))

    no_extra_weekend_act_present = np.all(weekday_weekend_diff_seq[:, 0] == 0)

    if no_extra_weekend_act_present:

        # No significant difference found between weekday and weekend profile

        logger.info("No significant difference found between weekday and weekend profile")

        item_output_object.update({
            "occupants_profile": occupants_profile,
            "weekday_profile": weekday_profile,
            "weekend_weekday_diff": weekend_weekday_diff
        })

        return item_input_object, item_output_object

    else:

        t0 = datetime.now()

        weekday_weekend_diff_seq = weekday_weekend_diff_seq[weekday_weekend_diff_seq[:, seq_label] != 0]

        weekend_hours = 0

        if np.any(weekday_weekend_diff_seq[:, seq_label] == 1):
            weekend_hours = np.max(weekday_weekend_diff_seq[weekday_weekend_diff_seq[:, seq_label] == 1, seq_len])

        weekday_profile["weekend_hours"] = weekend_hours

        # Preparing weekday-weekend diff profile for all chunks

        weekend_act_present_flag = np.logical_and(weekday_weekend_diff_seq[:, seq_label] == 1,
                                                  weekday_weekend_diff_seq[:, seq_len] >= samples_per_hour * 0.5)
        weekend_act_absent_flag = np.logical_and(weekday_weekend_diff_seq[:, seq_label] == -1,
                                                 weekday_weekend_diff_seq[:, seq_len] >= samples_per_hour * 0.5)

        # preparing dictionary that will have weekday/weekend activity behaviour for each time slot of the day

        for i in range(len(weekday_weekend_diff_seq)):
            weekday_profile = \
                preparing_weekday_weekend_profile_for_each_time_slots(i, weekend_act_present_flag[i], weekend_act_absent_flag[i],
                                                                      weekday_weekend_diff_seq, weekday_profile, samples_per_hour, hours)

        weekday_weekend_diff = weekend_curve - weekday_curve
        weekday_weekend_diff_label = np.zeros(len(weekday_weekend_diff))
        weekday_weekend_diff_label[weekday_weekend_diff < -profile_diff_thres] = -1
        weekday_weekend_diff_label[weekday_weekend_diff > profile_diff_thres] = 1

        if np.all(weekday_weekend_diff_label[hours[0].astype(int)] < 1):
            weekday_profile["weekend_breakfast_absent"] = 1
        if np.all(weekday_weekend_diff_label[hours[2].astype(int)] < 1):
            weekday_profile["weekend_dinner_absent"] = 1
        if np.all(weekday_weekend_diff_label[hours[1].astype(int)] < 1):
            weekday_profile["weekend_lunch_absent"] = 1

        t1 = datetime.now()

        logger.info("Preparing weekday-weekend diff profile took | %.3f s", get_time_diff(t0, t1))

        # Updating occupancy profile based on information derived from weekday/weekend profile
        # for example office goer score is slightly increased for extra weekend activity user

        weekday_profile, occupants_profile = update_occupancy_profile(weekday_profile, occupants_profile, hours, weekday_weekend_diff_label)

        t2 = datetime.now()

        logger.info(" Updating occupancy profile based on weekday/weekend profile took | %.3f s", get_time_diff(t1, t2))

    item_output_object.update({
        "occupants_profile": occupants_profile,
        "weekday_profile": weekday_profile,
        "weekend_weekday_diff": weekend_weekday_diff
    })

    return item_input_object, item_output_object


def update_occupancy_profile(weekday_profile, occupants_profile, time_slots, weekday_weekend_diff_label):

    """
    update occupancy profile of the user based on weekday weekend behavior of the user

    Parameters:
        weekday_profile             (dict)           : weekday profile
        occupants_profile           (dict)           : occupancy profile
        time_slots                  (list)           : all time slots of the day
        weekday_weekend_diff_label  (np.ndarray)     : array containing information of extra weekend activity

    Returns:
        weekday_profile             (dict)           : weekday profile
        occupants_profile           (dict)           : occupancy profile
    """

    office_goer = occupants_profile.get("occupants_prob")[0]
    early_arrival = occupants_profile.get("occupants_prob")[1]
    home_stayer = occupants_profile.get("occupants_prob")[2]

    config = get_weekday_weekend_profile_config()

    early_arrival_score_thres = config.get('early_arrival_score_thres')
    office_goer_score_thres = config.get('office_goer_score_thres')
    home_stayer_score_thres = config.get('home_stayer_score_thres')
    morn_weekend_act_score_offset = config.get('morn_weekend_act_score_offset')
    eve_weekend_act_score_offset = config.get('eve_weekend_act_score_offset')
    len_thres_for_morn_weekend_act = config.get('len_thres_for_morn_weekend_act')
    len_thres_for_eve_weekend_act = config.get('len_thres_for_eve_weekend_act')

    # Increasing office going prob for users, having higher weekday morning activity
    if (np.sum(weekday_weekend_diff_label[time_slots[3].astype(int)] > 0) > len(time_slots[3]) * len_thres_for_morn_weekend_act) and \
            office_goer < office_goer_score_thres:
        office_goer = office_goer + morn_weekend_act_score_offset

    # Increasing office going prob for users, having higher day time activity on weekends
    if (np.sum(weekday_weekend_diff_label[time_slots[5].astype(int)] < 0) > len(time_slots[5]) * len_thres_for_eve_weekend_act) and \
            office_goer < office_goer_score_thres:
        office_goer = office_goer + eve_weekend_act_score_offset

    # Increasing early arrival for users, having higher morning time activity on weekdays
    if (np.sum(weekday_weekend_diff_label[time_slots[4].astype(int)] < 0) > len(time_slots[4]) * len_thres_for_morn_weekend_act) and \
            early_arrival < early_arrival_score_thres:
        early_arrival = early_arrival + morn_weekend_act_score_offset

    # Increasing early arrival for users, having higher evening time activity on weekdays
    if (np.sum(weekday_weekend_diff_label[time_slots[6].astype(int)] < 0) > len(time_slots[6]) * len_thres_for_eve_weekend_act) and \
            early_arrival < early_arrival_score_thres:
        early_arrival = early_arrival + eve_weekend_act_score_offset

    # Increasing stay at home prob for users, having no additional weekday/weekend acivtivity
    if (np.sum(weekday_weekend_diff_label[time_slots[3].astype(int)] == 0) > len(time_slots[3]) * len_thres_for_eve_weekend_act) and \
            home_stayer > home_stayer_score_thres:
        home_stayer = home_stayer + morn_weekend_act_score_offset

    # Increasing office going count for users, having higher day time activity on weekends
    if np.sum(weekday_weekend_diff_label[time_slots[3].astype(int)] > 1) > len(time_slots[3]) * len_thres_for_morn_weekend_act:
        weekday_profile["extra_office_goer"] = 1

    occupants_profile["occupants_prob"][0] = office_goer
    occupants_profile["occupants_prob"][1] = early_arrival
    occupants_profile["occupants_prob"][2] = home_stayer

    return weekday_profile, occupants_profile


def preparing_weekday_weekend_profile_for_each_time_slots(time_idx, weekend_act_present_flag, weekend_act_absent_flag,
                                                          weekday_weekend_diff_seq, weekday_profile, samples_per_hour, time_slots):

    """
    update weekday profile dict of the user based on weekday weekend behavior of the user

    Parameters:
        time_idx                    (int)            : index of current time slot
        weekend_act_present_flag    (bool)           : true if extra weekend activity is present
        weekend_act_present_flag    (bool)           : true if extra weekday activity is present
        weekday_profile             (dict)           : weekday profile
        samples_per_hour            (int)            : samples in an hour
        time_slots                  (list)           : all time slots of the day

    Returns:
        weekday_profile             (dict)           : weekday profile
    """

    weekday_weekend_diff_seq_start = weekday_weekend_diff_seq[time_idx, 1]
    weekday_weekend_diff_seq_end = weekday_weekend_diff_seq[time_idx, 2]

    thres = 0.5 * samples_per_hour

    weekend_act_time = np.arange(weekday_weekend_diff_seq_start, weekday_weekend_diff_seq_end + 1)

    if weekend_act_present_flag and (len(np.intersect1d(weekend_act_time, time_slots[0])) > thres):
        weekday_profile["weekend_morning_present"] = 1
    if weekend_act_present_flag and (len(np.intersect1d(weekend_act_time, time_slots[1])) > thres):
        weekday_profile["weekend_mid_day_present"] = 1
    if weekend_act_present_flag and (len(np.intersect1d(weekend_act_time, time_slots[2])) > thres):
        weekday_profile["weekend_night_present"] = 1
    if weekend_act_present_flag and (len(np.intersect1d(weekend_act_time, time_slots[4])) > thres):
        weekday_profile["weekend_early_eve_present"] = 1

    weekday_profile["weekend_laundry"] = weekend_act_present_flag and (len(np.intersect1d(weekend_act_time, time_slots[3])) > 1.5 * samples_per_hour)

    if not weekend_act_absent_flag:
        return weekday_profile

    if len(np.intersect1d(weekend_act_time, time_slots[0])) > thres:
        weekday_profile["weekend_morning_absent"] = 1
    if len(np.intersect1d(weekend_act_time, time_slots[1])) > thres:
        weekday_profile["weekend_mid_day_absent"] = 1
    if len(np.intersect1d(weekend_act_time, time_slots[2])) > thres:
        weekday_profile["weekend_night_absent"] = 1
    if len(np.intersect1d(weekend_act_time, time_slots[4])) > thres:
        weekday_profile["weekend_early_eve_absent"] = 1

    return weekday_profile
