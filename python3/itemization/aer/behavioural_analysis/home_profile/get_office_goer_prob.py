
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Calculates count of Office goers
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.behavioural_analysis.home_profile.config.occupancy_profile_config import get_occupancy_profile_config


def get_office_goer_prob(item_input_object, item_output_object, activity_levels_mapping,
                         home_stayer_count, stay_at_home_prob):

    """
       Calculates count of Office goers

       Parameters:
           item_input_object             (dict)               : Dict containing all hybrid inputs
           item_output_object            (dict)               : Dict containing all hybrid outputs
           activity_levels_mapping       (np.ndarray)         : timestamp level activity levels mapping
           home_stayer_count             (int)                : calculated count of stay at home people
           stay_at_home_prob             (float)              : calculated probability of stay at home people

       Returns:
           office_goer_count             (int)                : Count of office goers present in the house
           office_going_prob             (float)              : Probability of office goers present in the house
           multi_office_goers            (int)                : True if multiple office goers are present in the house
           morning_activity              (int)                : True if morning activity present in the house
    """

    office_going_prob = 0

    activity_curve = item_input_object.get("weekday_activity_curve")
    activity_seq = item_output_object.get("profile_attributes").get('activity_sequences')
    activity_segments = item_output_object.get("profile_attributes").get('activity_segments')
    activity_levels = item_output_object.get("profile_attributes").get('activity_levels')
    samples_per_hour = item_input_object.get("item_input_params").get('samples_per_hour')

    # initialilizing config and required inputs

    config = get_occupancy_profile_config(samples_per_hour, home_stayer_count).get('office_goer_config')

    activity_curve_range = np.percentile(activity_curve, 98) - np.percentile(activity_curve, 2)

    evening_hours = config.get('evening_hours')

    morning_activity_present_bool, morn_act_score = get_morning_activity_params(activity_curve, activity_seq, config)

    min_office_goer_score = 0
    min_office_goer_score_weights = config.get('min_office_goer_score_weights')
    eve_act_levels_thres =  config.get('eve_act_levels_thres')
    office_score_thres = config.get('office_score_thres')
    office_score_weights = config.get('office_score_weights')
    office_score_offset = config.get('office_score_offset')
    zero_office_score = config.get('zero_office_score')
    office_goer_prob_thres = config.get('office_goer_prob_thres')

    activity_levels = np.append(0, activity_levels)

    act_level_list = activity_levels[activity_levels_mapping.astype(int)]

    act_level_list[act_level_list == 0] = np.min(activity_curve)

    # fetching activity levels of the user

    evening_levels = np.diff(activity_levels_mapping[evening_hours])
    levels_diff = np.diff(act_level_list[evening_hours])

    # preparing max office goer score if atleast some evening activity is present

    if np.any(evening_levels > 0) and np.any(activity_seq[evening_hours] == 1):

        eve_act_levels = np.max(levels_diff[evening_levels > 0])
        min_office_goer_score = min_office_goer_score_weights[0] + morn_act_score + \
                                min_office_goer_score_weights[1] * eve_act_levels

        # in case of insufficient evening activity, min office goer is removed

        if eve_act_levels < eve_act_levels_thres[0] or (morn_act_score == 0 and eve_act_levels < eve_act_levels_thres[1]):
            min_office_goer_score = 0

    office_going_prob, multi_office_goers_bool, multi_office_goers_score, valid_act_segments = \
        calculate_office_goer_score(activity_segments, activity_curve_range, activity_curve, config, evening_hours,
                                    morn_act_score, activity_seq, stay_at_home_prob, office_going_prob)

    # checking if office goer tag can be given to the user

    if office_going_prob >= office_score_thres[0] or multi_office_goers_score > office_score_thres[1]:
        office_going_prob = office_score_weights[0] * office_going_prob + office_score_weights[1] * multi_office_goers_score + office_score_offset
    else:
        office_going_prob = min(zero_office_score, office_going_prob)

    # updating office goer score to handle cases where evening activity was continuously increasing instead of sudden increase

    eve_time_slot_1 = config.get('eve_time_slot_1')
    eve_time_slot_2 = config.get('eve_time_slot_2')
    eve_time_slot_0 = config.get('eve_time_slot_0')
    eve_time_slot_3 = config.get('eve_time_slot_3')
    thres_for_inc_diff = config.get('thres_for_inc_diff')
    office_goer_offset_for_inc_diff = config.get('office_goer_offset_for_inc_diff')

    if samples_per_hour > 1:

        increasing_diff_1 = (np.sum(activity_curve[eve_time_slot_1]) - np.sum(activity_curve[eve_time_slot_0])) / samples_per_hour

        increasing_diff_2 = (np.sum(activity_curve[eve_time_slot_2]) - np.sum(activity_curve[eve_time_slot_1])) / samples_per_hour

        increasing_diff_3 = (np.sum(activity_curve[eve_time_slot_3]) - np.sum(activity_curve[eve_time_slot_2])) / samples_per_hour

        increasing_diff = max(increasing_diff_1, increasing_diff_2)
        increasing_diff = max(increasing_diff, increasing_diff_3)

        # If increasing trend is present in evening activity, extra offset is given to office goer score

        if increasing_diff > thres_for_inc_diff and home_stayer_count == 2:
            office_going_prob = office_going_prob + office_goer_offset_for_inc_diff[0]

        elif increasing_diff > thres_for_inc_diff:
            office_going_prob = office_going_prob + office_goer_offset_for_inc_diff[1]

        elif increasing_diff < thres_for_inc_diff:
            office_going_prob = office_going_prob - office_goer_offset_for_inc_diff[1]

    office_going_prob = max(office_going_prob, min_office_goer_score)

    # assigning user to office goer tag if score is greater than 0.5
    # assigning user to multiple office goer tag if score is greater than 0.7, based on the quatity of activity rise during evening
    # these multiple office goer users are given higher stat app consumption during evening hours

    office_goers = office_going_prob > office_goer_prob_thres[0]

    office_goer_count = 0

    office_going_prob = np.round(office_going_prob, 2)

    office_goer_count, multi_office_goers_bool = \
        update_office_goer_tag(valid_act_segments, office_goers, office_going_prob, office_goer_prob_thres,
                               office_goer_count, multi_office_goers_bool)

    office_going_prob = min(1, office_going_prob)

    if stay_at_home_prob > 0.9:
        office_goer_count = office_goer_count - 1
        office_going_prob = max(0, office_going_prob - 0.2)
        office_goer_count = max(office_goer_count, 0)
        multi_office_goers_bool = 0

    return office_goer_count, office_going_prob, multi_office_goers_bool, morning_activity_present_bool


def update_office_goer_tag(valid_act_segments, office_goers, office_going_prob, office_goer_prob_thres, office_goer_count, multi_office_goers_bool):

    """
       Calculates count of Office goers

       Parameters:
           item_input_object             (dict)               : Dict containing all hybrid inputs
           item_output_object            (dict)               : Dict containing all hybrid outputs
           activity_levels_mapping       (np.ndarray)         : timestamp level activity levels mapping
           home_stayer_count             (int)                : calculated count of stay at home people
           stay_at_home_prob             (float)              : calculated probability of stay at home people

       Returns:
           office_goer_count             (int)                : Count of office goers present in the house
           office_going_prob             (float)              : Probability of office goers present in the house
           multi_office_goers            (int)                : True if multiple office goers are present in the house
           morning_activity              (int)                : True if morning activity present in the house
    """

    if office_goers:
        office_goer_count = 1
        multi_office_goers_bool = int(office_going_prob >= office_goer_prob_thres[1])

        if valid_act_segments >= 3:
            multi_office_goers_bool = 1

        if multi_office_goers_bool:
            office_goer_count = 2

    return office_goer_count, multi_office_goers_bool


def get_morning_activity_params(activity_curve, activity_seq, config):

    """
     This function checks presence of morning living load activity

       Parameters:
           activity_curve                (np.ndarray)         : activity profile of the user
           activity_seq                  (np.ndarray)         : activity sequences
           config                        (dict)               : config dict
       Returns:
           morning_activity_present_bool (bool)               : flag that represents presence of morning actvity
           morn_act_score                (float)              : score associated with morning activity
    """

    morning_hours = config.get('morning_hours')
    morning_end_time = config.get('morning_end_time')
    max_morn_act_der_thres = config.get('max_morn_act_der_thres')
    morn_act_score_thres = config.get('morn_act_score_thres')

    # checking if morning activity is present in activity profile

    morn_act_score = 0
    morning_activity_present_bool = 0

    activity_curve_derivative = activity_curve - np.roll(activity_curve, 1)

    morning_labels = activity_seq[morning_hours][:-1]

    if np.any(morning_labels == 1):

        morning_activity_present_bool = 1

        morning_rise = morning_hours[np.where(morning_labels)[0][0]]

        time_wind = np.arange(morning_rise, morning_end_time + 1)

        early_arrival_activity = np.where(activity_curve_derivative < 0.01)[0]

        early_arrival_activity = np.intersect1d(early_arrival_activity, time_wind)

        i = 1

        # checking each activity segment during morning hours

        morning_activity_present_bool = 0

        if len(early_arrival_activity):

            val = int(early_arrival_activity[0])

            max_morn_act_der = activity_curve_derivative[val]
            net_morn_act_der = activity_curve_derivative[val]

            while i < len(early_arrival_activity)-1:

                if early_arrival_activity[i] == early_arrival_activity[i+1]-1:
                    net_morn_act_der = activity_curve_derivative[int(early_arrival_activity[i])] + net_morn_act_der
                    max_morn_act_der = min(net_morn_act_der, max_morn_act_der)

                else:
                    max_morn_act_der = min(activity_curve_derivative[int(early_arrival_activity[i])], max_morn_act_der)
                    net_morn_act_der = 0

                i = i + 1

            # if activity rise is less than threshold, morning_activity_present_bool is made 0

            if max_morn_act_der > -max_morn_act_der_thres:
                morning_activity_present_bool = 0

            # else morning activity score is calculated

            if max_morn_act_der < -morn_act_score_thres:
                morn_act_score = morn_act_score_thres
            else:
                morn_act_score = (max_morn_act_der + max_morn_act_der_thres) / (-0.9)

    return morning_activity_present_bool, morn_act_score


def calculate_office_goer_score(activity_segments, activity_curve_range, activity_curve, config, evening_hours,
                                morn_act_score, activity_seq, stay_at_home_prob, office_going_prob):

    """
    Calculates office goer score

       Parameters:
           activity_segments             (np.ndarray)         : array containing information for individual segments
           activity_curve_range          (float)              : range of activity profile
           activity_curve                (np.ndarray)         : activity profile of the user
           config                        (dict)               : config dict
           evening_hours                 (np.ndarray)         : evening hours
           morn_act_score                (float)              : score associated with morning activity
           activity_seq                  (np.ndarray)         : activity sequences
           stay_at_home_prob             (float)              : calculated probability of stay at home people
           office_going_prob             (float)              : initialized office goer score


       Returns:
           office_going_prob             (float)              : updated office goer score
           multi_office_goers_bool       (int)                : True if multiple office goers are present in the house
           multi_office_goers_score      (float)              : updated multiple office goer score
    """

    multi_office_goers_bool = 0
    valid_act_segments = 0
    multi_office_goers_score = 0

    len_threshold = config.get('len_threshold')
    act_diff_thres = config.get('diff_threshold')
    valid_segments = config.get('valid_segments')
    late_night_hours = config.get('late_night_hours')
    morning_end_time = config.get('morning_end_time')
    max_score_for_act_rise = config.get('max_score_for_act_rise')
    act_rise_score_thres = config.get('act_rise_score_thres')
    stat_at_home_thres = config.get('stat_at_home_thres')
    multi_office_goers_score_thres = config.get('multi_office_goers_score_thres')

    office_going_prob_offset = config.get('office_going_prob_offset')
    min_overlapping_frac = config.get('min_overlapping_frac')

    samples_per_hour = int(len(activity_curve) / 24)

    # checking user activity profile for each segment of activity

    for i in range(len(activity_segments)):

        if not (activity_segments[i, 0] in valid_segments):
            continue

        start = activity_segments[i, 2]
        end = activity_segments[i, 3]

        index_array = get_index_array(start, end, len(activity_curve))

        # removing late night hours while removing the activity segment

        index_array = np.setdiff1d(index_array, late_night_hours)

        # This variable is to check whether the current activity segment is present in evening hours

        check_eve_time_presence = len(np.intersect1d(index_array, evening_hours)) > samples_per_hour

        # This variable is to check whether 30% of the segment lies in evening hours

        check_ev_time_presence_frac = False

        if start >= morning_end_time:
            check_ev_time_presence_frac = len(np.intersect1d(index_array, evening_hours)) / len(index_array) >= min_overlapping_frac

        activity_curve_range = max(activity_curve_range, 0.15)

        # picking only evening hours for further calculation

        index_array = np.intersect1d(index_array, evening_hours)

        if check_ev_time_presence_frac and check_eve_time_presence and np.any(activity_seq[index_array] == 1):

            # user is assigned to office goer category since consistent increase in evening load detected

            office_going_prob = office_going_prob_offset + max(min(morn_act_score, 0.1), 0)

            valid_act_segments = valid_act_segments + 1

            hours_count = int(len(index_array) / samples_per_hour)

            # This variable is to allot score to calculate chances of having higher occupants based on length of consistent eve activity

            check_based_on_hour_count = \
                ((np.exp(hours_count / len_threshold) - np.exp(-hours_count / len_threshold)) / (np.exp(-hours_count / len_threshold) +
                                                                                                 np.exp(hours_count / len_threshold))) / 5

            if stay_at_home_prob > stat_at_home_thres:
                activity_rise = np.max(activity_curve[index_array]) - np.min(
                    activity_curve[index_array][: np.argmax(activity_curve[index_array]) + 1])
            else:
                activity_rise = np.max(activity_curve[index_array]) - np.min(activity_curve[index_array])

            # This variable is to allot score to calculate chances of having higher occupants based on amount of rise in evening activity

            check_based_on_act_rise = activity_rise / (act_diff_thres * activity_curve_range)
            check_based_on_act_rise = min(check_based_on_act_rise, max_score_for_act_rise)

            # This variable is to allot score to calculate chances of having higher occupants based on max activity profile of the user

            check6 = np.max(activity_curve[index_array] * 0.1 / np.percentile(activity_curve, 95))

            if check_based_on_act_rise > act_rise_score_thres:
                multi_office_goers_score = check_based_on_hour_count + check_based_on_act_rise + check6

                multi_office_goers_bool = multi_office_goers_score >= multi_office_goers_score_thres

    return office_going_prob, multi_office_goers_bool, multi_office_goers_score, valid_act_segments
