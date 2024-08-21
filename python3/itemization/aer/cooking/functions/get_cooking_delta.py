
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Estimate energy delta for cooking appliances
"""

# Import python packages

import numpy as np


def get_cooking_delta(clean_days_consumption, samples_per_hour, cooking_config):

    """
       Prepare tou estimate of cooking appliances for weekday and weekends individually

        Parameters:
           clean_days_consumption     (np.ndarray)         : Consumption profile for clean days
           samples_per_hour           (int)                : samples in an hour
           cooking_config             (dict)               : cooking config dictionary

       Returns:
           cooking_delta              (np.ndarray)        : cooking delta
       """

    # fetch cooking hours

    breakfast_hours = cooking_config.get("breakfast_hours")
    lunch_hours = cooking_config.get("lunch_hours")
    dinner_hours = cooking_config.get("dinner_hours")

    cooking_hours = [breakfast_hours, lunch_hours, dinner_hours]

    cooking_delta = np.zeros(len(cooking_hours))

    total_cooking_slots = 3

    duration = [max(1, 0.5 * samples_per_hour), samples_per_hour, samples_per_hour]

    length = len(clean_days_consumption)

    scaling_factor_for_energy_delta = cooking_config.get('scaling_factor_for_energy_delta')

    # Calculate delta for breakfast, lunch and dinner

    for index in range(total_cooking_slots):

        activity_inc = 0

        cooking_hours_for_given_slot = np.arange(cooking_hours[index][0], cooking_hours[index][-1]-duration[index]+1, 1)

        for j in cooking_hours_for_given_slot:

            diff_in_living_load_activity = \
                np.max(clean_days_consumption[(np.arange(j, j + duration[index] + 1).astype(int)) % length]) - \
                np.min(clean_days_consumption[(np.arange(j - duration[index], j + 1).astype(int)) % length])

            activity_inc = max(activity_inc, diff_in_living_load_activity)

        cooking_delta[index] = activity_inc*scaling_factor_for_energy_delta

    # Modify cooking delta in case of small values

    for index in range(total_cooking_slots):

        inc_energy_delta_to_be_used_for_cooking_estimate1 = \
            (np.max(clean_days_consumption) - np.min(clean_days_consumption)) > \
                cooking_config.get("energy_profile_limit")[0]/samples_per_hour and\
                cooking_delta[index] < cooking_config.get("delta_limit")/samples_per_hour

        inc_energy_delta_to_be_used_for_cooking_estimate2 = \
            (np.max(clean_days_consumption) - np.min(clean_days_consumption)) > \
            cooking_config.get("energy_profile_limit")[1] / samples_per_hour and \
            cooking_delta[index] < cooking_config.get("delta_limit") / samples_per_hour

        if inc_energy_delta_to_be_used_for_cooking_estimate1:
            cooking_delta[index] = cooking_delta[index] * cooking_config.get("delta_multiplier")[0]

        elif inc_energy_delta_to_be_used_for_cooking_estimate2:
            cooking_delta[index] = cooking_delta[index] * cooking_config.get("delta_multiplier")[1]

    return cooking_delta
