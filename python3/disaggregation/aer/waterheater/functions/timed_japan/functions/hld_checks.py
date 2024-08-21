"""
Author - Sahana M
Date - 20/07/2021
The module performs HLD checks
"""

# Import python packages
import numpy as np
from copy import deepcopy


def false_positive_check(twh_confidence, wh_config, debug, twh_estimation):
    """
    Function used to check false positive cases
    Parameters:
        twh_confidence          (float)         : Timed wh confidence
        wh_config               (dict)          : Water heater config
        debug                   (dict)          : Debug dictionary
        twh_estimation          (np.ndarray)    : Timed wh final estimation array
    Returns:
        twh_confidence          (float)         : Timed wh confidence
    """

    penalty_2 = wh_config.get('hld_checks').get('penalty_2')
    detection_threshold = wh_config.get('detection_thr')
    amplitude_variation = wh_config.get('hld_checks').get('amplitude_variation')

    # Check for detection in the old run and perform amplitude variability check

    if twh_confidence >= detection_threshold and debug.get('hsm_in') is not None:
        old_hld = debug.get('hsm_in').get('timed_hld')[0]
        if old_hld == 0 and wh_config.get('pilot_id') in wh_config.get('hvac_pilots') and (
                np.std(np.diff(np.nansum(twh_estimation, axis=1))) >= amplitude_variation):
            twh_confidence -= penalty_2 * twh_confidence

    return twh_confidence


def false_negative_check(twh_confidence, original_confidence, time_bands_bool, wh_config, debug):
    """
    Function to make sure continuity is maintained and avoiding false negatives
    Parameters:
        twh_confidence          (float)         : Timed wh confidence
        original_confidence     (float)         : Original timed wh confidence from the model
        time_bands_bool         (np.ndarray)    : Timed band Boolean
        wh_config               (dict)          : Water heater config
        debug                   (dict)          : Debug dictionary
    Returns:
        twh_confidence          (float)         : Timed wh confidence
    """

    detection_threshold = wh_config.get('detection_thr')

    if twh_confidence < detection_threshold and debug.get('hsm_in') is not None:

        historical_time_bands = debug.get('hsm_in').get('twh_time_bands')

        if (debug.get('disagg_mode') in ['incremental', 'mtd']) and \
                ((type(historical_time_bands) is np.ndarray) or
                 (type(historical_time_bands) is list)) and \
                (debug.get('hsm_in').get('timed_confidence_score') is not None):
            update_score_bool = True
        else:
            update_score_bool = False

        # If score is below threshold but was detected in the historical run, update the score

        if update_score_bool and np.sum(np.logical_and(historical_time_bands, time_bands_bool)) and \
                (debug.get('hsm_in').get('timed_confidence_score')[0] >= detection_threshold):
            twh_confidence = original_confidence

    return twh_confidence


def hld_checks(debug, wh_config):
    """
    This function is used to perform HLD checks after detection
    Parameters:
        debug               (dict)          : Debug dictionary
        wh_config           (dict)          : Wh configurations dictionary
    Returns:
        hld                 (int)           : Final HLD
        debug               (dict)          : Debug dictionary
    """

    hld = bool(debug.get('timed_hld'))
    twh_confidence = debug.get('timed_confidence')

    if debug.get('disagg_mode') != 'mtd':

        # Extract the required variables
        # penalty_1 is the penalty value applied for violation in runs twh allowed and long duration (mostly Pool pump)

        factor = debug.get('factor')
        twh_estimation = deepcopy(debug.get('final_twh_matrix'))
        twh_estimation_bool = twh_estimation > 0
        penalty_1 = wh_config.get('hld_checks').get('penalty_1')
        max_runs_allowed = wh_config.get('hld_checks').get('max_runs_allowed')
        max_hours_allowed = wh_config.get('hld_checks').get('max_hours_allowed')
        duration_variation_thr = wh_config.get('hld_checks').get('duration_variation')

        if twh_confidence >= wh_config.get('detection_thr'):

            original_confidence = deepcopy(twh_confidence)

            # multiple start and end time check

            zero_array = np.full(shape=(twh_estimation_bool.shape[0], 1), fill_value=0)
            box_energy_idx_diff = np.diff(np.c_[zero_array,  twh_estimation_bool.astype(int), zero_array])
            corners_bool = box_energy_idx_diff[:, :-1]

            start_times = np.sum(corners_bool == 1, axis=1)
            end_times = np.sum(corners_bool == -1, axis=1)

            start_times_percentile = np.percentile(start_times, q=90)
            end_times_percentile = np.percentile(end_times, q=90)

            # More number of runs in a day check

            if start_times_percentile > max_runs_allowed or end_times_percentile > max_runs_allowed:
                twh_confidence = twh_confidence - penalty_1*twh_confidence

            # Long duration check

            duration_arr = np.sum(twh_estimation_bool, axis=1)
            if np.percentile(duration_arr, q=90) > (max_hours_allowed * factor):
                twh_confidence = twh_confidence - penalty_1*twh_confidence

            # Duration variation check

            duration_variation = np.std(abs(np.diff(duration_arr)))
            if duration_variation > duration_variation_thr:
                penalty = (duration_variation/duration_variation_thr)/10 + 0.1
                twh_confidence = twh_confidence - penalty * twh_confidence

            detection_threshold = wh_config.get('detection_thr')
            time_bands = np.sum(twh_estimation, axis=0)
            time_bands_bool = time_bands > 0

            # Check for False positives

            twh_confidence = false_positive_check(twh_confidence, wh_config, debug, twh_estimation)

            # Check to False Negatives and Avoiding discontinuity

            twh_confidence = false_negative_check(twh_confidence, original_confidence, time_bands_bool, wh_config, debug)

            if twh_confidence < detection_threshold:
                hld = False
            else:
                hld = True

            debug['twh_bands'] = time_bands_bool
            debug['timed_confidence'] = max(twh_confidence, 0)

    return hld, twh_confidence, debug
