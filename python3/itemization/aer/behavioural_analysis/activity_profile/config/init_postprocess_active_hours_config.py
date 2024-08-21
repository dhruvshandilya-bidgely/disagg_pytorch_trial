"""
Author - Nisha Agarwal
Date - 3rd Jan 21
This file contain config values for calculating active hours
"""

# Import python packages

import numpy as np


def init_postprocess_active_hours_config(activity_curve, activity_curve_derivative, activity_curve_diff, samples_per_hour):

    """
    Initialize config used for sleep hours calculation

    Parameters:
        activity_curve              (np.ndarray)    : Activity curve of the user
        activity_curve_derivative   (np.ndarray)    : derivative of activity curve of the user
        activity_curve_diff         (np.ndarray)    : Range of activity curve of the user
        samples_per_hour            (int)           : number of samples in an hour
    Returns:
        config                      (dict)          : active hours config dict
    """

    config = dict()

    activity_curve_buckets = [np.min(activity_curve),
                              np.min(activity_curve) + 0.3 * activity_curve_diff,
                              np.min(activity_curve) + 0.6 * activity_curve_diff]

    derivative_buckets = [np.min(activity_curve_derivative),
                          np.min(activity_curve_derivative)+0.2*activity_curve_diff,
                          np.min(activity_curve_derivative)+0.4*activity_curve_diff,
                          np.min(activity_curve_derivative)+0.6*activity_curve_diff,
                          np.min(activity_curve_derivative)+0.8*activity_curve_diff]

    active_hours_score = np.zeros(24*samples_per_hour)
    active_hours_score[np.arange(0 * samples_per_hour, 1 * samples_per_hour + 1)] = 0.2
    active_hours_score[np.arange(1 * samples_per_hour, 4 * samples_per_hour + 1)] = 0.5
    active_hours_score[np.arange(4 * samples_per_hour, 6 * samples_per_hour + 1)] = 0.1
    active_hours_score[np.arange(6 * samples_per_hour, 8 * samples_per_hour + 1)] = 0.05
    active_hours_score[np.arange(8 * samples_per_hour, 9 * samples_per_hour + 1)] = 0

    morning_hours = np.arange(6 * samples_per_hour, 9 * samples_per_hour)
    night_hours = np.arange(21 * samples_per_hour, 22 * samples_per_hour)

    total_samples = 24 * samples_per_hour
    probable_sleeping_hours = np.arange(total_samples-2*samples_per_hour, total_samples + 10*samples_per_hour) % total_samples

    late_evening_hours = np.arange(19 * samples_per_hour, 22 * samples_per_hour)

    # "deri_scoring_limit": derivative threshold for activity score based on derivative,
    # 'derivative_buckets': derivative buckets further used for activity scoring,
    #  "activity_curve_buckets": activity curve buckets further used for activity scoring ,
    #  'non_active_tod_score': activity score given to inactive segments,
    #  'zigzag_score': score given to zigzag pattern segment ,
    #  'increasing_score': activity score for increasing sequence ,
    #  'decreasing_score': activity score for decreasing sequences ,
    #  'constant_score': active score for constant seq,
    #  'min_sleeping_hours': min sleeping hours,
    #  'derivative_limit': active sequence derivative threshold,
    #  "zigzag_hour_interval": duration used for identifying zigzag pattern in activity curve ,
    #  'morning_hours': morning_hours,
    #  'night_hours': night_hours,
    #  'sleeping_hours': probable_sleeping_hours,
    #  'active_hours_score': initialized active_hours_scores based on time of day
    #  'late_evening_hours': late_evening_hours

    postprocess_active_hours_config = {
        "deri_scoring_limit": 0.02,
        'derivative_buckets': derivative_buckets,
        "activity_curve_buckets": activity_curve_buckets,
        'non_active_tod_score': 0.7,
        'zigzag_score': 15,
        'increasing_score': 0.15,
        'decreasing_score': 0.4,
        'constant_score': 0.6,
        'min_sleeping_hours': 5,
        'derivative_limit': 0.05,
        "zigzag_hour_interval": 2,
        'morning_hours': morning_hours,
        'night_hours': night_hours,
        'sleeping_hours': probable_sleeping_hours,
        'active_hours_score': active_hours_score,
        'late_evening_hours': late_evening_hours,
        'thres_for_inc_seq': 0.03,
        'thres_for_dec_seq': 0.05,
        'net_thres_for_inc_seq': 0.04,
        'net_thres_for_dec_seq': 0.06,
        'thres_for_morn_hours': 0.01,
        'thres_for_night_hours': 0.02
    }

    config.update({
        "postprocess_active_hour_config": postprocess_active_hours_config
    })

    return config
