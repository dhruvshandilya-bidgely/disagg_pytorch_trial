"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Calculated user attributes used for statistical appliance estimation
"""

# Import python packages

import copy
import numpy as np

# import functions from within the project

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import rolling_func
from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.behavioural_analysis.home_profile.config.occupancy_profile_config import get_occupancy_profile_config


def get_users_parameters(activity_curve, samples_per_hour, activity_seq, sleep_hours, occupancy_features):

    """
       Calculated user attributes used for statistical appliance estimation

       Parameters:
           activity_curve        (np.ndarray)         : activity profile of the user
           samples_per_hour      (int)                : samples in an hour
           activity_seq          (np.ndarray)         : activity sequences
           sleep_hours           (np.ndarray)         : sleep hours boolean array
           occupancy_features    (np.ndarray)         : occupancy profile of the user

       Returns:
           user_parameters       (dict)               : dict of user attributes used for statistical appliance estimation
       """

    activity_curve_derivative = activity_curve - np.roll(activity_curve, 1)

    config = get_occupancy_profile_config(samples_per_hour).get('user_attributes')

    start_list = config.get('start_list')
    end_list = config.get('end_list')
    morning_hours = config.get('morning_hours')
    sleeping_hours = config.get('sleeping_hours')
    default_sleep_time = config.get('default_sleep_time')
    default_wake_time = config.get('default_wake_time')

    user_parameters = {
        "morning_start": -1,
        "late_morning_activity_end_time": -1,
        "morning_end": -1,
        "early_activity_start_time": -1,
        "office_coming_time": -1,
        "sleep_time": -1,
        "lunch_start": -1,
        "lunch_end": -1,
        "dinner_start": -1,
        "dinner_end": -1,
        "lunch_present": 1
    }

    # check validity of wakeup time
    if np.any(sleep_hours[morning_hours] == 1):
        wakeup_time = np.where(sleep_hours[morning_hours] == 1)[0][0] / samples_per_hour + 5
    else:
        wakeup_time = -1

    if wakeup_time * samples_per_hour not in default_wake_time:
        wakeup_time = -1

    # check validity of sleep time
    if np.any(sleep_hours[sleeping_hours][::-1] == 1):
        sleep_time = ((len(sleeping_hours) - np.where(sleep_hours[sleeping_hours][::-1] == 1)[0][0] + 1) / samples_per_hour + 20) % 24
    else:
        sleep_time = -1

    if sleep_time * samples_per_hour not in (default_sleep_time) % (samples_per_hour * 24):
        sleep_time = -1

    # preparing all time slots

    hours_list = [[]] * len(start_list)

    for i in range(len(start_list)):
        hours_list[i] = np.arange(start_list[i], end_list[i] + 1)

    # checking validity of early activity start time

    early_activity_start_time = -1

    if occupancy_features[1] and np.any(activity_curve_derivative[hours_list[4]] > 0.02):
        early_activity_start_time = np.where(activity_curve_derivative[hours_list[4]] > 0.02)[0][0] / samples_per_hour + \
                                    (hours_list[4][0] / samples_per_hour)
    elif occupancy_features[1]:
        early_activity_start_time = -1

    # updating wakeup time/sleep time and early activity start time of the user
    user_parameters["morning_start"] = max(-1, int(wakeup_time * samples_per_hour))
    user_parameters["sleep_time"] = max(-1, int(sleep_time * samples_per_hour))
    user_parameters["early_activity_start_time"] = early_activity_start_time

    user_parameters = calculating_office_going_time(user_parameters, activity_curve, samples_per_hour, hours_list, activity_seq)

    rolling_avg_derivative = copy.deepcopy(activity_curve_derivative)

    if samples_per_hour > 2:
        rolling_avg_derivative = rolling_func(activity_curve_derivative, int(0.25 * samples_per_hour))

    # Calculating the time of days when morning activity of the user ends
    if user_parameters["morning_end"] == -1:
        if np.any(rolling_avg_derivative[hours_list[2]] < -0.02):
            morning_end = ((np.where(rolling_avg_derivative[hours_list[2]] < -0.02)[0][-1]) / samples_per_hour +
                           hours_list[2][0] / samples_per_hour) % 24
        else:
            morning_end = 10
        user_parameters["morning_end"] = morning_end * samples_per_hour

    if user_parameters["morning_start"] > user_parameters["morning_end"]:
        user_parameters["morning_end"] = user_parameters["morning_start"] + samples_per_hour

    if user_parameters["late_morning_activity_end_time"] == -1 and \
            np.any(rolling_avg_derivative[hours_list[1]] < -0.02):
        late_morning_activity_end_time = ((np.where(rolling_avg_derivative[hours_list[1]] < -0.02)[0][-1]) / samples_per_hour
                                          + hours_list[1][0] / samples_per_hour) % 24
        user_parameters["late_morning_activity_end_time"] = late_morning_activity_end_time * samples_per_hour

    user_parameters = prepare_cooking_activity_time(user_parameters, activity_curve, samples_per_hour, occupancy_features)

    return user_parameters


def calculating_office_going_time(user_parameters, activity_curve, samples_per_hour, hours_list, activity_seq):

    """
       Calculated user attributes - office going time of the user

       Parameters:
           user_parameters       (dict)               : dict of user attributes used for statistical appliance estimation
           activity_curve        (np.ndarray)         : activity profile of the user
           samples_per_hour      (int)                : samples in an hour
           hours_list            (list)               : list of all activity time slots
           activity_seq          (np.ndarray)         : activity sequences

       Returns:
           user_parameters       (dict)               : dict of user attributes used for statistical appliance estimation
       """

    config = get_occupancy_profile_config(samples_per_hour).get('user_attributes')

    overlap_thres = config.get('overlap_thres')

    activity_curve_derivative = activity_curve - np.roll(activity_curve, 1)

    segments_data = find_seq(activity_seq, activity_curve_derivative, activity_curve)

    tmp_segments_data = segments_data[segments_data[:, 0] == -1]

    # checking the decreasing activity segments of the user

    for i in range(len(tmp_segments_data)):

        start = tmp_segments_data[i, 1]
        end = tmp_segments_data[i, 2]
        index_array = get_index_array(start, end, len(activity_curve))

        # checking whether any decreasing activity was found during morning hours to check presence of morning activity end time

        if (len(np.intersect1d(index_array, hours_list[2]))) > overlap_thres:
            temp_index_array = np.intersect1d(index_array, hours_list[2])
            temp_derivative = np.where(activity_curve_derivative[temp_index_array] < -0.01)[0]

            if len(temp_derivative):
                user_parameters["morning_end"] = temp_index_array[temp_derivative[-1]]

        if (len(np.intersect1d(index_array, hours_list[1]))) > overlap_thres:
            temp_index_array = np.intersect1d(index_array, hours_list[1])
            temp_derivative = np.where(activity_curve_derivative[temp_index_array] < -0.01)[0]

            if len(temp_derivative):
                user_parameters["late_morning_activity_end_time"] = temp_index_array[temp_derivative[-1]]

    tmp_segments_data = segments_data[segments_data[:, 0] == 1]

    for i in range(len(tmp_segments_data)):

        start = tmp_segments_data[i, 1]
        end = tmp_segments_data[i, 2]
        index_array = get_index_array(start, end, len(activity_curve))

        # checking whether any increasing activity was found during evening hours
        # to check presence of start of activity or office arrival time of the user

        if user_parameters["office_coming_time"] == -1 and (len(np.intersect1d(index_array, hours_list[5]))) > overlap_thres:
            temp_index_array = np.intersect1d(index_array, hours_list[5])
            temp_derivative = np.where(activity_curve_derivative[temp_index_array] > 0.01)[0]

            if len(temp_derivative):
                user_parameters["office_coming_time"] = temp_index_array[temp_derivative[0]]

    return user_parameters


def prepare_cooking_activity_time(user_parameters, activity_curve, samples_per_hour, occupancy_features):


    """
       Calculated user attributes used for statistical appliance estimation - probable cooking usage slot

       Parameters:
           user_parameters       (dict)               : dict of user attributes used for statistical appliance estimation
           activity_curve        (np.ndarray)         : activity profile of the user
           samples_per_hour      (int)                : samples in an hour
           occupancy_features    (np.ndarray)         : occupancy profile of the user

       Returns:
           user_parameters       (dict)               : dict of user attributes used for statistical appliance estimation
       """

    activity_curve_derivative = activity_curve - np.roll(activity_curve, 1)

    lunch_score = 0

    config = get_occupancy_profile_config(samples_per_hour).get('user_attributes')

    lunch_probable_hours = config.get('lunch_probable_hours')
    dinner_hours = config.get('dinner_hours')
    lunch_hours = config.get('lunch_hours')
    default_score = config.get('default_score')
    default_lunch_slot = config.get('default_lunch_slot')

    lunch_hours_score = np.zeros(len(activity_curve))

    cooking_hours = 1 * samples_per_hour

    lunch_hours_score[lunch_hours] = 1

    lunch_time_act_range = np.abs(activity_curve_derivative)

    # picking the time where activity range is highest in lunch time slots

    lunch_time_act_range = np.multiply(lunch_time_act_range, lunch_hours_score)

    score = -default_score

    lunch_start = -1

    for i in np.arange(12 * samples_per_hour, 13 * samples_per_hour + 1):

        temp_score = np.sum(lunch_time_act_range[i: i + cooking_hours])

        if temp_score > score:
            score = temp_score
            lunch_start = i

    dinner_hours_score = np.zeros(len(activity_curve))

    cooking_hours = 1 * samples_per_hour

    dinner_hours_score[dinner_hours] = 1

    # picking the time where activity range is highest in dinner time slots

    dinner_time_act_range = np.abs(activity_curve_derivative)

    dinner_time_act_range = np.multiply(dinner_time_act_range, dinner_hours_score)

    score = -default_score

    dinner_start = -1

    for i in dinner_hours:

        temp_score = np.sum(dinner_time_act_range[i: i + cooking_hours])

        if temp_score > score:
            score = temp_score
            dinner_start = i

    user_parameters["lunch_start"] = lunch_start
    user_parameters["lunch_end"] = lunch_start + cooking_hours
    user_parameters["dinner_start"] = dinner_start
    user_parameters["dinner_end"] = dinner_start + cooking_hours

    # checking if calculated lunch score is valid based on activity profile in the given region
    # else default times are alloted

    activity_curve_range = np.percentile(activity_curve, 97) - np.percentile(activity_curve, 5)

    lunch_score = lunch_score + int(
        np.mean(activity_curve[lunch_probable_hours]) > np.min(activity_curve) + 0.2 * activity_curve_range)
    lunch_score = lunch_score + int((np.max(activity_curve[lunch_probable_hours]) -
                                     np.min(activity_curve[lunch_probable_hours])) > 0.35 * activity_curve_range)
    lunch_score = lunch_score + int(np.max(activity_curve_derivative[lunch_probable_hours]) > 0.03)
    lunch_score = lunch_score + int(
        np.sum(activity_curve_derivative[np.arange(default_lunch_slot, default_lunch_slot + cooking_hours + 1)]) > 0.05)

    if (not occupancy_features[2]) or lunch_score < 2 or user_parameters["lunch_start"] == -1:
        user_parameters["lunch_present"] = 0

    if user_parameters["lunch_present"]:
        user_parameters["lunch_start"] = default_lunch_slot
        user_parameters["lunch_end"] = default_lunch_slot + cooking_hours

    return user_parameters
