"""
Author - Nisha Agarwal
Date - 3rd Jan 21
This file contain config values for calculating active hours
"""

# Import python packages

import numpy as np


def init_active_hours_config(samples_per_hour):

    """
    Initialize config used for active hours calculation

    Parameters:
        samples_per_hour         (int)          : number of samples in an hour
    Returns:
        config                   (dict)         : active hours config dict
    """

    config = dict()

    # 'weightage': weightage to the 4 factors used to tag segment at active/inactive ,
    # These 4 factors include tha variation, pattern, net slope, and the level
    # 'var_limit': variation factor threshold ,
    #  'slope_limit': slope factor threshold ,
    #  'filter_active_hours_limit': threshold for derivative of each activity sequence
    #  'small_diff_decreament': threshold adjustment of users with lower activity range
    #  'diff_limit': threshold adjustment of users with lower activity range
    #  'score_limit': activity score threshold
    #  'diff_factor': threshold used to filter out top levels activity segments
    #  'derivative_limit': threshold used to filter out activity segments with high derivative

    active_hours_config = {
        'weightage': [0.25, 0.25, 0.25, 0.25],
        'var_limit': 0.05,
        'slope_limit': 0.05,
        'filter_active_hours_limit': 0.05,
        'small_diff_decreament': 0.03,
        'diff_limit': 0.3,
        'score_limit': 0.75,
        'diff_factor': 0.3,
        'derivative_limit': 0.06,
        'hour_limit': 4
    }

    config.update({

        "active_hours_config": active_hours_config

    })

    active_hours_score = np.zeros(24*samples_per_hour)
    active_hours_score[np.arange(0 * samples_per_hour, 1 * samples_per_hour + 1)] = 0.2
    active_hours_score[np.arange(1 * samples_per_hour, 4 * samples_per_hour + 1)] = 0.5
    active_hours_score[np.arange(4 * samples_per_hour, 6 * samples_per_hour + 1)] = 0.1
    active_hours_score[np.arange(6 * samples_per_hour, 8 * samples_per_hour + 1)] = 0.05
    active_hours_score[np.arange(8 * samples_per_hour, 9 * samples_per_hour + 1)] = 0

    morning_hours = np.arange(6 * samples_per_hour, 9 * samples_per_hour)
    night_hours = np.arange(21 * samples_per_hour, 22 * samples_per_hour)

    total_samples = 24 * samples_per_hour
    probable_sleeping_hours = np.arange(total_samples-3*samples_per_hour, total_samples + 8*samples_per_hour) % total_samples
    morning_activity_hours = np.arange(5 * samples_per_hour, 11 * samples_per_hour)
    late_evening_hours = np.arange(19 * samples_per_hour, 22 * samples_per_hour).astype(int)

    #   'non_active_tod_score': inactivity score for already tagged inactive segment
    #   'zigzag_score': inactivity score for zigzag pattern segment
    #   'increasing_score': inactivity score for active increasing segment
    #   'decreasing_score': inactivity score for active decreasing segment
    #   'constant_score': inactivity score for active constant segment
    #   'min_sleeping_hours': minimum sleeping hours
    #   'derivative_limit': threshold used to detect zigzag pattern
    #   'morning_hours': assumed morning hours
    #   'night_hours': assumed night hours
    #   'sleeping_hours': most probable sleeping hours
    #   'active_hours_score': score based on time of the day
    #   'late_evening_hours': assumed late evening hours

    extend_inactive_hours_config = {
        'non_active_tod_score': 0.7,
        'zigzag_score': 15,
        'increasing_score': 0.15,
        'decreasing_score': 0.4,
        'constant_score': 0.6,
        'min_sleeping_hours': 5,
        'derivative_limit': 0.05,
        'morning_hours': morning_hours,
        'night_hours': night_hours,
        'sleeping_hours': probable_sleeping_hours,
        'active_hours_score': active_hours_score,
        'late_evening_hours': late_evening_hours
    }

    config.update({

        "extend_inactive_hours_config": extend_inactive_hours_config

    })

    # inactivity score for maintaining a minimum of 5 hours of sleeping hours
    # high score is given to the mid night hours
    # the score decreases as we move away from the mid night hours

    min_sleeping_hours_score = np.zeros(24 * samples_per_hour)
    min_sleeping_hours_score[np.arange(21 * samples_per_hour, 23 * samples_per_hour)] = 1
    min_sleeping_hours_score[np.arange(23 * samples_per_hour, 24 * samples_per_hour)] = 3
    min_sleeping_hours_score[np.arange(0 * samples_per_hour, 3 * samples_per_hour)] = 5
    min_sleeping_hours_score[np.arange(3 * samples_per_hour, 4 * samples_per_hour)] = 2
    min_sleeping_hours_score[np.arange(4 * samples_per_hour, 6 * samples_per_hour)] = 1
    min_sleeping_hours_score[np.arange(6 * samples_per_hour, 8 * samples_per_hour)] = 0.5

    # inactivity score for maintaining a minimum of 1 hours of morning lighting hours
    # high score is given to the mid night hours
    # the score decreases as we move away from the mid night hours
    # lowest score is given in 6-9 time frame, since this time of the day is most suitable for morning activity

    before_sunrise_activity_score = np.zeros(24 * samples_per_hour)
    before_sunrise_activity_score[np.arange(0 * samples_per_hour, 3 * samples_per_hour)] = 4
    before_sunrise_activity_score[np.arange(3 * samples_per_hour, 4 * samples_per_hour)] = 2
    before_sunrise_activity_score[np.arange(4 * samples_per_hour, 6 * samples_per_hour)] = 1
    before_sunrise_activity_score[np.arange(6 * samples_per_hour, 8 * samples_per_hour)] = 0.5
    before_sunrise_activity_score[np.arange(8 * samples_per_hour, 9 * samples_per_hour)] = 0.2

    # inactivity score for maintaining a maximum of 11 hours of sleeping time
    # high score is given to the 8pm-10pm, and 6-10am window
    # This is because if required to cutoff the sleeping hours of the user,
    # we will start with reducing window with the highest score

    max_sleeping_hours_score = np.zeros(24 * samples_per_hour)
    max_sleeping_hours_score[np.arange(20 * samples_per_hour, 22 * samples_per_hour)] = 4
    max_sleeping_hours_score[np.arange(22 * samples_per_hour, 24 * samples_per_hour)] = 2
    max_sleeping_hours_score[np.arange(0 * samples_per_hour, 3 * samples_per_hour)] = 0.5
    max_sleeping_hours_score[np.arange(3 * samples_per_hour, 4 * samples_per_hour)] = 1
    max_sleeping_hours_score[np.arange(4 * samples_per_hour, 6 * samples_per_hour)] = 2
    max_sleeping_hours_score[np.arange(6 * samples_per_hour, 8 * samples_per_hour)] = 3
    max_sleeping_hours_score[np.arange(8 * samples_per_hour, 10 * samples_per_hour)] = 5

    # 'mid_night_start': mid night duration starting hour
    # 'mid_night_end': mid night duration ending hours
    #  'probable_sleeping_hours': probable_sleeping_hours,
    #  'morning_activity_hours': morning_activity_hours,
    #  'min_morning_activity_hours': minimum morning active hours
    #  'max_morning_activity_hours': maximum morning active hours
    #  'min_sleeping_hours': minimum sleeping hours
    #  'max_sleeping_hours':  minimum sleeping hours
    #  'morning_start': probable start time of morning hours
    #  'min_sleeping_hours_score': min_sleeping_hours_score,
    #  'max_sleeping_hours_score': max_sleeping_hours_score,
    #  'before_sunrise_activity_score': before_sunrise_activity_score,
    #  'before_sunrise_activity_hours': before sunrise active hours

    sleep_time_config = {
        'mid_night_start': 0,
        'mid_night_end': 5,
        'mid_night_derivative_limit': 0.07,
        'probable_sleeping_hours2': probable_sleeping_hours,
        'morning_activity_hours': morning_activity_hours,
        'min_morning_activity_hours': 1,
        'max_morning_activity_hours': 3.5,
        'min_sleeping_hours': 5,
        'max_sleeping_hours': 9,
        'morning_start': 6,
        'weights': [0.05, 0.2, 0.1],
        'min_sleeping_hours_score': min_sleeping_hours_score,
        'max_sleeping_hours_score': max_sleeping_hours_score,
        'before_sunrise_activity_score': before_sunrise_activity_score,
        'before_sunrise_activity_hours': np.arange(0, 9 * samples_per_hour).astype(int),
    }

    config.update({
        "sleep_time_config": sleep_time_config
    })


    postprocess_active_hours_config = {
        'mid_night_start': 0
    }

    config.update({
        "postprocess_active_hour_config": postprocess_active_hours_config
    })

    return config
