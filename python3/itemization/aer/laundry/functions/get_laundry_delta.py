"""
Author - Nisha Agarwal
Date - 9th Feb 20
Prepare timestamp level energy delta for laundry estimation
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.config.pilot_constants import PilotConstants

from python3.itemization.aer.laundry.config.get_estimation_config import get_estimation_config


def get_laundry_delta(item_input_object, pilot, weekday_consumption, weekend_consumption, samples_per_hour, activity_curve):

    """
       Prepare change in user living load activity (at each hour of the day) to calculate for laundry estimates

       Parameters:
           pilot                        (int)                 : pilot id
           weekday_consumption          (np.ndarray)          : weekday energy profile
           weekend_consumption          (np.ndarray)          : Weekend energy profile
           samples_per_hour             (int)                 : samples in an hour
           activity_curve               (np.ndarray)          : Activity curve of the user

       Returns:
           weekday_delta                (np.ndarray)          : Weekday energy delta
           weekend_delta                (np.ndarray)          : Weekend energy delta
    """

    samples_per_hour = int(samples_per_hour)

    config = get_estimation_config(pilot_level_config=item_input_object.get('pilot_level_config'), samples_per_hour=samples_per_hour)

    breakfast_hours = config.get('breakfast_hours')
    lunch_hours = config.get('lunch_hours')
    dinner_hours = config.get('dinner_hours')
    default_max_val = config.get('default_max_val')
    default_min_val = config.get('default_min_val')
    cooking_offset = config.get('cooking_offset')
    non_laundry_hours = config.get('non_laundry_hours')
    zero_laundry_hours = config.get('zero_laundry_hours')
    zero_laundry_hours_low_cons_pilots = config.get('zero_laundry_hours_low_cons_pilots')

    weekday_delta = np.zeros(len(weekday_consumption))
    weekend_delta = np.zeros(len(weekday_consumption))

    length = len(activity_curve)

    # using living load activity curve of the user,
    # the energy rise in each time slots of the day is calculated
    # Based on these energy increase/decrease values, laundry consumption will be initialized

    for idx1 in range(len(activity_curve)):

        max_val = default_max_val
        min_val = default_min_val

        for idx2 in range(samples_per_hour + 1):

            max_val = max(np.sum(weekday_consumption[np.arange(idx1 + idx2, idx1 + idx2 + samples_per_hour).astype(int) % length]), max_val)
            min_val = min(min_val, np.sum(weekday_consumption[np.arange(idx1 - samples_per_hour - idx2, idx1 - idx2).astype(int) % length]))

        weekday_delta[idx1] = max_val - min_val

        max_val = default_max_val
        min_val = default_min_val

        for idx2 in range(samples_per_hour + 1):
            max_val = max(np.sum(weekend_consumption[np.arange(idx1 + idx2, idx1 + idx2 + samples_per_hour).astype(int) % length]), max_val)
            min_val = min(min_val, np.sum(weekend_consumption[np.arange(idx1 - samples_per_hour - idx2, idx1 - idx2).astype(int) % length]))

        weekend_delta[idx1] = max_val - min_val

    # energy delta is made 0 in inactive hours

    weekday_delta[non_laundry_hours] = 0
    weekend_delta[non_laundry_hours] = 0

    if pilot in PilotConstants.INDIAN_PILOTS:
        weekday_delta[zero_laundry_hours_low_cons_pilots] = 0
        weekend_delta[zero_laundry_hours_low_cons_pilots] = 0
    else:
        weekday_delta[zero_laundry_hours] = 0
        weekend_delta[zero_laundry_hours] = 0

    # slight energy delta is reduced during cooking hours, to avoid overlap with cooking initialized consumption

    weekday_delta[np.argmax(weekday_delta[breakfast_hours]) + breakfast_hours[0]] = \
        weekday_delta[np.argmax(weekday_delta[breakfast_hours]) + breakfast_hours[0]] - cooking_offset[0]
    weekday_delta[np.argmax(weekday_delta[lunch_hours]) + lunch_hours[0]] = \
        weekday_delta[np.argmax(weekday_delta[lunch_hours]) + lunch_hours[0]] - cooking_offset[1]
    weekday_delta[np.argmax(weekday_delta[dinner_hours]) + dinner_hours[0]] = \
        weekday_delta[np.argmax(weekday_delta[dinner_hours]) + dinner_hours[0]] - cooking_offset[2]

    weekend_delta[np.argmax(weekend_delta[breakfast_hours]) + breakfast_hours[0]] =\
        weekend_delta[np.argmax(weekend_delta[breakfast_hours]) + breakfast_hours[0]] - cooking_offset[0]
    weekend_delta[np.argmax(weekend_delta[lunch_hours]) + lunch_hours[0]] = \
        weekend_delta[np.argmax(weekend_delta[lunch_hours]) + lunch_hours[0]] - cooking_offset[1]
    weekend_delta[np.argmax(weekend_delta[dinner_hours]) + dinner_hours[0]] = \
        weekend_delta[np.argmax(weekend_delta[dinner_hours]) + dinner_hours[0]] - cooking_offset[2]

    return weekday_delta, weekend_delta
