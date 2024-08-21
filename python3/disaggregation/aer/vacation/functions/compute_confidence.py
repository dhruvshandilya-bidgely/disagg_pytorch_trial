"""
Author - Mayank Sharan
Date - 12/12/19
Compute confidence for days as vacation in different stages of vacation detection
"""

# Import python packages

import numpy as np


def compute_power_chk_conf(rejected_in_power_chk, loop_break_power, power_chk_thr, vac_confidence, vacation_config):

    """
    Compute confidence values associated with days that were rejected as a part of the window power check

    Parameters:
        rejected_in_power_chk   (np.ndarray)        : Boolean array marking days that were rejected in power check
        loop_break_power        (np.ndarray)        : Array containing the power values that caused the loop break
        power_chk_thr           (np.ndarray)        : Array containing the thresholds used to validate windows
        vac_confidence          (np.ndarray)        : Confidence values for each day being vacation
        vacation_config         (dict)              : Contains all configuration variables needed for vacation

    Returns:
        vac_confidence          (np.ndarray)        : Confidence values for each day being vacation
    """

    # Initialize the config to compute confidence

    confidence_config = vacation_config.get('power_check_confidence')

    # Assign initial confidence value to all rejected days

    vac_confidence[rejected_in_power_chk] = confidence_config.get('max_conf')

    # Compute the margin by which the breaking power values exceeded the threshold

    break_pow_delta = loop_break_power - np.tile(power_chk_thr, (2, 1)).transpose()

    # Compute how much the confidence is to be reduced based on how far the break power is from threshold.
    # Use x / (c + x) to change this to lie in 0 to 1

    scaled_delta = np.sum(np.divide(break_pow_delta, confidence_config.get('scale_const') + break_pow_delta),
                          axis=1)

    # Scale the range further down so that it can modify the confidence to lie in a small range

    conf_delta = confidence_config.get('delta_wt') * scaled_delta

    # Fill removal value such that the confidence is 0 for days which get rejected with nan in one of the power values

    conf_delta[np.isnan(conf_delta)] = confidence_config.get('max_conf')

    # Update the confidence values for rejected days by subtracting the computed delta

    vac_confidence[rejected_in_power_chk] = vac_confidence[rejected_in_power_chk] - conf_delta[rejected_in_power_chk]
    vac_confidence = np.round(vac_confidence, 2)

    return vac_confidence


def compute_final_chk_conf(power_check_passed_bool, confirmed_vac_bool, confidence_params, vac_confidence,
                           vacation_config):

    """
    Compute and populate confidence values for days selected and not selected as a part of the final set of checks

    Parameters:
        power_check_passed_bool (np.ndarray)        : Boolean array denoting days that passed the power check
        confirmed_vac_bool      (np.ndarray)        : Boolean array marking days that are confirmed as type 1 vacation
        confidence_params       (dict)              : Contains variables needed to compute confidence values
        vac_confidence          (np.ndarray)        : Confidence values for each day being vacation
        vacation_config         (dict)              : Contains all configuration variables needed for vacation

    Returns:
        vac_confidence          (np.ndarray)        : Confidence values for each day being vacation
        perc_arr                (np.ndarray)        : Array containing what percentage of all deviation are the top 3
    """

    # Extract variables from confidence params dictionary

    dev_arr = confidence_params.get('dev_arr')
    std_dev_arr = confidence_params.get('std_dev_arr')
    std_dev_thr = confidence_params.get('std_dev_thr')
    cand_sel_bool = confidence_params.get('cand_sel_bool')
    max_3_dev_arr = confidence_params.get('max_3_dev_arr')
    sliding_mean_thr = confidence_params.get('sliding_mean_thr')
    sliding_power_mean = confidence_params.get('sliding_power_mean')

    # Initialize config variables needed for calculations

    uns_conf_config = vacation_config.get('final_uns_confidence')
    sel_conf_config = vacation_config.get('final_sel_confidence')

    # Initialize confidence for days rejected using final checks

    reject_cand_bool = np.logical_and(power_check_passed_bool, np.logical_not(confirmed_vac_bool))
    vac_confidence[reject_cand_bool] = uns_conf_config.get('max_conf')

    # Compute the variation in confidence based on rejection criteria

    # Metric 1 : Sliding Power Mean

    sliding_pow_mean_diff = sliding_power_mean - sliding_mean_thr[power_check_passed_bool]
    sliding_pow_mean_diff[sliding_pow_mean_diff < 0] = 0

    conf_uns_delta_1 = np.divide(sliding_pow_mean_diff, uns_conf_config.get('pwr_mean_const') + sliding_pow_mean_diff)
    conf_uns_delta_1_scaled = uns_conf_config.get('pwr_mean_wt') * conf_uns_delta_1

    # Metric 2 : Overall Std Dev

    overall_std_dev_diff = std_dev_arr - std_dev_thr[power_check_passed_bool]
    overall_std_dev_diff[overall_std_dev_diff < 0] = 0

    conf_uns_delta_2 = np.divide(overall_std_dev_diff, uns_conf_config.get('std_dev_const') + overall_std_dev_diff)
    conf_uns_delta_2_scaled = uns_conf_config.get('std_dev_wt') * conf_uns_delta_2

    # Metric 3 : Max Dev Variation

    max_dev_col = 0

    max_dev_diff = max_3_dev_arr[:, max_dev_col] - vacation_config.get('confirm_deviation').get('top_3_dev_thr')
    max_dev_diff[max_dev_diff < 0] = 0

    conf_uns_delta_3 = np.divide(max_dev_diff, uns_conf_config.get('max_dev_const') + max_dev_diff)
    conf_uns_delta_3_scaled = uns_conf_config.get('max_dev_wt') * conf_uns_delta_3

    # Metric 4 : Max 3 dev as percentage of total

    perc_arr = np.round(np.divide(np.sum(max_3_dev_arr, axis=1) * 100, np.nansum(np.sqrt(dev_arr), axis=1)), 2)

    conf_uns_delta_4 = np.divide(perc_arr, uns_conf_config.get('max_3_dev_perc_const') + perc_arr)
    conf_uns_delta_4_scaled = uns_conf_config.get('max_3_dev_perc_wt') * conf_uns_delta_4

    # Scale the sum of confidence deltas to the range you want to manipulate the actual confidence values in

    conf_uns_delta = uns_conf_config.get('delta_wt') * (conf_uns_delta_1_scaled + conf_uns_delta_2_scaled +
                                                        conf_uns_delta_3_scaled + conf_uns_delta_4_scaled)

    # Subtract the scaled delta from the confidence values to finalize for days rejected as a part of final checks

    unconfirmed_vac_bool = np.logical_not(cand_sel_bool)
    vac_confidence[reject_cand_bool] = vac_confidence[reject_cand_bool] - conf_uns_delta[unconfirmed_vac_bool]

    # Initialize confidence for days selected using final checks

    vac_confidence[confirmed_vac_bool] = sel_conf_config.get('min_conf')

    # Compute the variation in confidence based on selection criteria

    # Metric 1 : Sliding Power Mean

    sliding_pow_mean_diff = sliding_mean_thr[power_check_passed_bool] - sliding_power_mean
    sliding_pow_mean_diff[sliding_pow_mean_diff < 0] = 0

    conf_sel_delta_1 = np.divide(sliding_pow_mean_diff, sel_conf_config.get('pwr_mean_const') + sliding_pow_mean_diff)
    conf_sel_delta_1_scaled = sel_conf_config.get('pwr_mean_wt') * conf_sel_delta_1

    # Metric 2 : Overall Std Dev

    overall_std_dev_diff = std_dev_thr[power_check_passed_bool] - std_dev_arr
    overall_std_dev_diff[overall_std_dev_diff < 0] = 0

    conf_sel_delta_2 = np.divide(overall_std_dev_diff, sel_conf_config.get('std_dev_const') + overall_std_dev_diff)
    conf_sel_delta_2_scaled = sel_conf_config.get('std_dev_wt') * conf_sel_delta_2

    # Metric 3 : Percentage of Max 3 dev from total. We flip this as calculated before since we want additive effect

    conf_sel_delta_3_scaled = sel_conf_config.get('max_dev_wt') * (1 - conf_uns_delta_3)

    # Scale the sum of confidence deltas to the range you want to manipulate the actual confidence values in

    conf_3_delta = sel_conf_config.get('delta_wt') * (conf_sel_delta_1_scaled + conf_sel_delta_2_scaled +
                                                      conf_sel_delta_3_scaled)

    # Add the scaled delta to the confidence values to finalize for days selected as part of final checks

    vac_confidence[confirmed_vac_bool] = vac_confidence[confirmed_vac_bool] + conf_3_delta[cand_sel_bool]

    return vac_confidence, perc_arr
