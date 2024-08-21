
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Calculates count of early arrival - if someone arrives at home around 2-4pm,
 i.e. a consistent increase in activity is present in user raw data
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.itemization.aer.behavioural_analysis.home_profile.config.occupancy_profile_config import get_occupancy_profile_config


def get_early_arrival_prob(item_input_object, item_output_object):

    """
       Calculates count of people probably arriving before default office hours

       Parameters:
            item_input_object            (dict)               : Dict containing all inputs
            item_output_object           (dict)               : Dict containing all outputs

       Returns:
            early_arrivals               (int)                : Count of early arrival present in the house
            early_arr_prob               (float)              : Probability of early arrival being present in the house
    """

    activity_curve = item_input_object.get("weekday_activity_curve")
    activity_seq = item_output_object.get("profile_attributes").get('activity_sequences')
    samples_per_hour = item_input_object.get("item_input_params").get('samples_per_hour')

    config = get_occupancy_profile_config(samples_per_hour).get('early_arrivals_config')

    noon_hours = config.get('noon_hours')

    morning_drop_activity = check_morning_activity_drop(item_input_object)

    morning_rise_activity, morn_rise_thres = check_morning_activity_rise(item_input_object, item_output_object)

    # Calculate whether there is significant rise in activity in early evening hours
    # The time to be verified lies between 2pm to 4pm

    max_derivative = 0

    early_arr_count = 0

    early_arrival_activity = np.where(activity_seq == 1)[0]

    early_arrival_activity = np.intersect1d(early_arrival_activity, noon_hours)

    activity_curve_derivative = activity_curve - np.roll(activity_curve, 1)

    i = 0

    # Next section, estimates whether there is any continuous sequences of activity rise in early evening hours,
    # If not found, early arrival count is considered to be 0

    morning_and_noon_activity_present_flag = len(early_arrival_activity) and (morning_drop_activity or morning_rise_activity)

    if morning_and_noon_activity_present_flag:

        val = early_arrival_activity[0]

        max_derivative = activity_curve_derivative[val]

        net_derivative = activity_curve_derivative[val]

        while i < len(early_arrival_activity)-1:

            consequtive_activity_rise_flag = early_arrival_activity[i] == early_arrival_activity[i + 1] - 1

            if consequtive_activity_rise_flag:
                net_derivative = activity_curve_derivative[int(early_arrival_activity[i] + 1)] + net_derivative
                max_derivative = max(net_derivative, max_derivative)

            else:
                max_derivative = max(activity_curve_derivative[int(early_arrival_activity[i])], max_derivative)
                net_derivative = 0

            i = i + 1

        if (morning_drop_activity or morning_rise_activity) and max_derivative > morn_rise_thres:
            early_arr_count = 1

    # early arrival probability is calculated using max increase in active in the target horus

    early_arr_prob = config.get('offset') + config.get('deri_weight')*max_derivative

    early_arr_prob = min(1, early_arr_prob)

    return early_arr_count, early_arr_prob


def check_morning_activity_drop(item_input_object):

    """
       Calculates count of people probably arriving before default office hours

       Parameters:
            item_input_object            (dict)               : Dict containing all inputs

       Returns:
           morning_drop_activity         (int)                : 1 if significant drop in morning activity is found
    """

    activity_curve = item_input_object.get("weekday_activity_curve")
    samples_per_hour = item_input_object.get("item_input_params").get('samples_per_hour')

    config = get_occupancy_profile_config(samples_per_hour).get('early_arrivals_config')

    morning_hours = config.get('morning_hours')

    # Calculate activity curve derivative to be used to get the amplitude of change in activity

    activity_curve_derivative = activity_curve - np.roll(activity_curve, 1)
    activity_curve_range = np.percentile(activity_curve, 97) - np.percentile(activity_curve, 5)

    morning_drop_activity = 1

    morn_dip_thres = config.get('morn_dip_thres')

    morn_dip_thres = (morn_dip_thres + 0.01) * (activity_curve_range < config.get('act_thres')) + \
                     morn_dip_thres * (activity_curve_range >= config.get('act_thres'))

    # Calculate whether there is significant drop in activity in early morning hours

    if np.any(activity_curve_derivative[morning_hours] > 0):

        morning_drop_activity = 1

        morning_rise = morning_hours[np.where(activity_curve_derivative[morning_hours] > 0)[0][0]]

        # Since school going hours are generally around 8am, the morning active hours are between wakeup and 8 am

        morning_active_hours = np.arange(morning_rise, 8 * samples_per_hour + 1)

        # Possible hours of morning activity drop

        early_arrival_activity = np.where(activity_curve_derivative < 0.01)[0]
        early_arrival_activity = np.intersect1d(early_arrival_activity, morning_active_hours)

        i = 1

        # Next section, estimates whether there is any continuous sequences of activity drop,
        # if no morning drop is considered

        if len(early_arrival_activity):

            val = int(early_arrival_activity[0])

            max_derivative = activity_curve_derivative[val]
            net_derivative = activity_curve_derivative[val]

            while i < len(early_arrival_activity)-1:

                consequtive_activity_rise_flag = early_arrival_activity[i] == early_arrival_activity[i + 1] - 1

                if consequtive_activity_rise_flag:
                    net_derivative = activity_curve_derivative[int(early_arrival_activity[i])] + net_derivative
                    max_derivative = min(net_derivative, max_derivative)

                else:
                    max_derivative = min(activity_curve_derivative[int(early_arrival_activity[i])], max_derivative)
                    net_derivative = 0

                i = i + 1

            if max_derivative > -1 * morn_dip_thres:
                morning_drop_activity = 0

        else:
            morning_drop_activity = 0

    return morning_drop_activity


def check_morning_activity_rise(item_input_object, item_output_object):

    """
       Calculates count of people probably arriving before default office hours

       Parameters:
            item_input_object            (dict)               : Dict containing all inputs
            item_output_object           (dict)               : Dict containing all outputs

       Returns:
           morning_rise_activity         (int)                : 1 if significant rise in morning activity is found
           morn_rise_thres               (float)              : activity profile threshold used to determine change in morn activity

    """

    activity_curve = item_input_object.get("weekday_activity_curve")
    activity_seq = item_output_object.get("profile_attributes").get('activity_sequences')
    samples_per_hour = item_input_object.get("item_input_params").get('samples_per_hour')

    config = get_occupancy_profile_config(samples_per_hour).get('early_arrivals_config')

    # Calculate activity curve derivative to be used to get the amplitude of change in activity

    activity_curve_range = np.percentile(activity_curve, 97) - np.percentile(activity_curve, 5)

    morn_rise_thres = config.get('morn_rise_thres')

    morn_rise_thres = (morn_rise_thres - 0.01) * (activity_curve_range < config.get('act_thres')) + \
                      morn_rise_thres * (activity_curve_range >= config.get('act_thres'))

    # Calculate whether there is significant drop in activity in early morning hours

    # Calculate whether there is significant rise in activity in early morning hours
    # This activity rise is only verified in the early morning hours

    morning_rise_activity = 0

    early_eve_hours = config.get('early_eve_hours')

    early_arrival_activity = np.where(activity_seq == 1)[0]
    early_arrival_activity = np.intersect1d(early_arrival_activity, early_eve_hours)

    activity_curve_derivative = activity_curve - np.roll(activity_curve, 1)

    i = 0

    # Next section, estimates whether there is any continuous sequences of activity rise,
    # if no, morning rise is considered is considered to be 0

    if len(early_arrival_activity):

        val = early_arrival_activity[0]

        max_derivative = activity_curve_derivative[val]
        net_derivative = activity_curve_derivative[val]

        while i < len(early_arrival_activity) - 1:

            consecutive_activity_rise_flag = early_arrival_activity[i] == early_arrival_activity[i + 1] - 1

            if consecutive_activity_rise_flag:
                net_derivative = activity_curve_derivative[int(early_arrival_activity[i] + 1)] + net_derivative
                max_derivative = max(net_derivative, max_derivative)

            else:
                max_derivative = max(activity_curve_derivative[int(early_arrival_activity[i])], max_derivative)
                net_derivative = 0

            i = i + 1

        if max_derivative > morn_rise_thres:
            morning_rise_activity = 1

    return morning_rise_activity, morn_rise_thres
