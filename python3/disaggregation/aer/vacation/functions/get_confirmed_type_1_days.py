"""
Author - Mayank Sharan
Date - 12/12/19
Identify confirmed type 1 vacation days
"""

# Import python packages

import numpy as np


def get_confirmed_type_1_days(day_wise_baseload, power_check_passed_bool, sliding_power, confirmed_vac_bool,
                              vacation_config):

    """
    Identify days from the candidates available as confirmed type 1 vacation days

    Parameters:
        day_wise_baseload       (np.ndarray)        : Baseload computed corresponding to each day
        power_check_passed_bool (np.ndarray)        : Boolean array denoting days that passed the power check
        sliding_power           (np.ndarray)        : Array containing window wise power till the day was in the loop
        confirmed_vac_bool      (np.ndarray)        : Boolean array marking days that are confirmed as type 1 vacation
        vacation_config         (dict)              : Contains all configuration variables needed for vacation

    Returns:
        confirmed_vac_bool      (np.ndarray)        : Boolean array marking days that are confirmed as type 1 vacation
        sliding_power_mean      (np.ndarray)        : Array containing power mean value for all candidate days
        std_dev_arr             (np.ndarray)        : Array containing std dev values for all candidate days
        max_3_dev_arr           (np.ndarray)        : Array containing top 3 dev values for all candidate days
        confidence_params       (dict)              : Contains variables needed to compute confidence values
    """

    # Initialize variable containing sliding power data only for candidate days

    cand_sliding_power = sliding_power[power_check_passed_bool, :]

    # Compute number of valid values for each day

    num_values = cand_sliding_power.shape[1] - np.sum(np.isnan(cand_sliding_power), axis=1)

    # Compute standard deviation values for each day. We modify the stddev to conform to MATLAB method

    std_dev_arr = np.multiply(np.nanstd(cand_sliding_power, axis=1), np.sqrt(np.divide(num_values, num_values - 1)))
    std_dev_arr = np.round(std_dev_arr, 2)

    # Compute sliding power mean

    sliding_power_mean = np.round(np.nanmean(cand_sliding_power, axis=1), 2)

    # Compute all the deviation values for each candidate days

    dev_arr = cand_sliding_power - np.tile(sliding_power_mean, (cand_sliding_power.shape[1], 1)).transpose()
    dev_arr = np.round(np.power(dev_arr, 2), 2)

    # Replace all nan deviation values with negative infinity to ensure sorting works as desired then sort

    dev_arr[np.isnan(dev_arr)] = -np.inf
    sorted_dev_arr = np.sort(dev_arr, axis=1)

    # Using the sorted array get the top 3 deviation values arrange them such that they decrease in increasing columns

    num_top_dev = 3

    max_3_dev_arr = np.power(sorted_dev_arr[:, -num_top_dev:], 0.5)
    max_3_dev_arr = np.round(np.flip(max_3_dev_arr, axis=1), 2)

    # Initialize variables to compute the sliding power mean thresholds

    confirm_power_mean_config = vacation_config.get('confirm_power_mean')
    sliding_mean_thr = np.full(shape=day_wise_baseload.shape, fill_value=np.nan)

    # Identify days falling in different ranges of baseload values

    lv_1 = day_wise_baseload < confirm_power_mean_config.get('bl_lv_1')

    lv_2 = np.logical_and(day_wise_baseload >= confirm_power_mean_config.get('bl_lv_1'),
                          day_wise_baseload < confirm_power_mean_config.get('bl_lv_2'))

    lv_3 = np.logical_and(day_wise_baseload >= confirm_power_mean_config.get('bl_lv_2'),
                          day_wise_baseload < confirm_power_mean_config.get('bl_lv_3'))

    lv_4 = np.logical_and(day_wise_baseload >= confirm_power_mean_config.get('bl_lv_3'),
                          day_wise_baseload < confirm_power_mean_config.get('bl_lv_4'))

    lv_5 = np.logical_and(day_wise_baseload >= confirm_power_mean_config.get('bl_lv_4'),
                          day_wise_baseload < confirm_power_mean_config.get('bl_lv_5'))

    lv_6 = np.logical_and(day_wise_baseload >= confirm_power_mean_config.get('bl_lv_5'),
                          day_wise_baseload < confirm_power_mean_config.get('bl_lv_6'))

    lv_7 = day_wise_baseload >= confirm_power_mean_config.get('bl_lv_6')

    # Assign sliding power mean thresholds as per classification

    sliding_mean_thr[lv_1] = confirm_power_mean_config.get('bl_lv_1_thr')
    sliding_mean_thr[lv_2] = confirm_power_mean_config.get('bl_lv_2_thr')
    sliding_mean_thr[lv_3] = confirm_power_mean_config.get('bl_lv_3_thr')
    sliding_mean_thr[lv_4] = confirm_power_mean_config.get('bl_lv_4_thr')
    sliding_mean_thr[lv_5] = confirm_power_mean_config.get('bl_lv_5_thr')
    sliding_mean_thr[lv_6] = confirm_power_mean_config.get('bl_lv_6_thr')
    sliding_mean_thr[lv_7] = confirm_power_mean_config.get('bl_lv_7_thr')

    # Identify days that are selected based on the sliding power mean threshold

    sliding_mean_thr = day_wise_baseload + sliding_mean_thr
    sliding_mean_pass_bool = sliding_power_mean < sliding_mean_thr[power_check_passed_bool]

    # Initialize variables to compute the deviation thresholds

    confirm_dev_config = vacation_config.get('confirm_deviation')
    std_dev_thr = np.full(shape=day_wise_baseload.shape, fill_value=confirm_dev_config.get('std_dev_thr'))

    if vacation_config.get('user_info').get('is_europe'):

        # Identify days in different baseload range values for assigning stddev thresholds

        lv_1 = day_wise_baseload >= confirm_dev_config.get('bl_lv_1')
        lv_2 = day_wise_baseload >= confirm_dev_config.get('bl_lv_2')

        # Assign stddev threshold as per classification

        std_dev_thr[lv_1] = confirm_dev_config.get('bl_lv_1_thr')
        std_dev_thr[lv_2] = confirm_dev_config.get('bl_lv_2_thr')

        # Get days that comply to the std dev threshold

        std_dev_low_bool = std_dev_arr <= std_dev_thr[power_check_passed_bool]

        # Check that the top 3 deviations comply to the deviation threshold

        top_3_check_count = np.sum(max_3_dev_arr > confirm_dev_config.get('top_3_dev_thr'), axis=1)
        top_3_dev_low_bool = top_3_check_count == 0

        # Additional check to see if the top 2 deviations are not much larger than stddev for stddev above a threshold
        # Eliminates concentrated consumption days

        # Initialize column indices to be used for the max deviations array

        md_col = 0
        md3_col = 2

        top_2_dev_sum = np.sum(max_3_dev_arr[:, :md3_col], axis=1)

        dev_reject = np.logical_or(top_2_dev_sum > confirm_dev_config.get('top_2_dev_max_ratio') * std_dev_arr,
                                   max_3_dev_arr[:, md_col] > confirm_dev_config.get('top_dev_max_ratio') * std_dev_arr)

        dev_reject = np.logical_and(std_dev_arr >= confirm_dev_config.get('min_stddev_for_dev_reject'), dev_reject)

        # Select days that qualify either of the std dev and top 3 deviation check and also clear the high dev check

        dev_pass_bool = np.logical_and(np.logical_or(std_dev_low_bool, top_3_dev_low_bool), np.logical_not(dev_reject))

    else:

        # Get days that comply to the std dev threshold

        std_dev_low_bool = std_dev_arr <= std_dev_thr[power_check_passed_bool]

        # Check that the top 3 deviations comply to the deviation threshold

        top_3_check_count = np.sum(max_3_dev_arr > confirm_dev_config.get('top_3_dev_thr'), axis=1)
        top_3_dev_low_bool = top_3_check_count == 0

        # The days that either have top 3 deviations under threshold or std dev under threshold are selected

        dev_pass_bool = np.logical_or(std_dev_low_bool, top_3_dev_low_bool)

    # Populate the array marking days that are confirmed as type 1 vacation

    cand_days_idx = np.where(power_check_passed_bool)[0]
    cand_sel_bool = np.logical_and(dev_pass_bool, sliding_mean_pass_bool)

    confirmed_vac_idx_arr = cand_days_idx[cand_sel_bool]
    confirmed_vac_bool[confirmed_vac_idx_arr] = True

    # Populate dictionary with return values that will be used for confidence calculation

    confidence_params = {
        'dev_arr': dev_arr,
        'std_dev_arr': std_dev_arr,
        'std_dev_thr': std_dev_thr,
        'cand_sel_bool': cand_sel_bool,
        'max_3_dev_arr': max_3_dev_arr,
        'sliding_mean_thr': sliding_mean_thr,
        'sliding_power_mean': sliding_power_mean,
    }

    return confirmed_vac_bool, sliding_power_mean, std_dev_arr, max_3_dev_arr, confidence_params
