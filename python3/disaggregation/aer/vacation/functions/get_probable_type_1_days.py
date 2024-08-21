"""
Author - Mayank Sharan
Date - 11/12/19
Identify probable type 1 vacation days
"""

# Import python packages

import numpy as np


def get_probable_type_1_days(day_wise_baseload, day_wise_power, vacation_config):

    """
    Identify probable type 1 vacation days

    Parameters:
        day_wise_power      (np.ndarray)        : Power computed corresponding to each day
        day_wise_baseload   (np.ndarray)        : Baseload computed corresponding to each day
        vacation_config     (dict)              : Contains all configuration variables needed for vacation

    Returns:
        probable_vac_bool   (np.ndarray)        : Boolean array marking days which are selected as probable
        probable_threshold  (np.ndarray)        : Day by day threshold to mark day as probable
    """

    # Initialize variables from config

    probable_day_config = vacation_config.get('probable_day')

    # Initialize probable threshold array

    probable_threshold = np.full(shape=day_wise_baseload.shape, fill_value=np.nan)

    # Identify days falling in different ranges of baseload values

    lv_1 = day_wise_baseload < probable_day_config.get('bl_lv_1')

    lv_2 = np.logical_and(day_wise_baseload >= probable_day_config.get('bl_lv_1'),
                          day_wise_baseload < probable_day_config.get('bl_lv_2'))

    lv_3 = np.logical_and(day_wise_baseload >= probable_day_config.get('bl_lv_2'),
                          day_wise_baseload < probable_day_config.get('bl_lv_3'))

    lv_4 = day_wise_baseload >= probable_day_config.get('bl_lv_3')

    # Assign probable thresholds as per classification

    probable_threshold[lv_1] = probable_day_config.get('bl_lv_1_thr')
    probable_threshold[lv_2] = probable_day_config.get('bl_lv_2_thr')
    probable_threshold[lv_4] = probable_day_config.get('bl_lv_3_thr')

    # Linearly assign thresholds for values in between level 2 to level 3

    slope = round((probable_day_config.get('bl_lv_3_thr') - probable_day_config.get('bl_lv_2_thr')) /
                  (probable_day_config.get('bl_lv_3') - probable_day_config.get('bl_lv_2')), 2)

    probable_threshold[lv_3] = \
        slope * (day_wise_baseload[lv_3] - probable_day_config.get('bl_lv_2')) + probable_day_config.get('bl_lv_2_thr')

    # Add the additional threshold assigned to the baseload ot get the final threshold

    probable_threshold = probable_threshold + day_wise_baseload

    # Decide if a day is a probable vacation day

    is_valid_power = np.logical_not(np.logical_or(np.isnan(day_wise_power), day_wise_power == 0))
    probable_vac_bool = np.logical_and(day_wise_power < probable_threshold, is_valid_power)

    return probable_vac_bool, probable_threshold
