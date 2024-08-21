"""
Author - Mayank Sharan
Date - 28/11/19
Initialize the config for running vacation
"""

# Import python packages

import numpy as np


def init_vacation_config(pilot_id, sampling_rate, timezone, uuid):

    """
    Extracts the continent of the user from the timezone and accordingly initializes config for EU or non EU user

    Parameters:
        pilot_id                (int)               : Id identifying the pilot user belongs to
        timezone                (str)               : String denoting the timezone the user belongs to
        sampling_rate           (int)               : The frequency at which the data is sampled
        uuid                    (str)               : Unique id associated with the user

    Returns:
        vacation_config         (dict)              : Dictionary containing all configuration parameters for vacation
    """

    # Initialize the config dictionary

    vacation_config = {}

    # Extract if the user belongs to Europe or not. Timezone format is <Continent>/<City>

    is_europe = timezone.split('/')[0] == 'Europe'

    # Initialize user info in the config

    vacation_config['user_info'] = {
        'uuid': uuid,
        'tz': timezone,
        'pilot_id': pilot_id,
        'is_europe': is_europe,
        'thin_pulse_rem': False,
        'sampling_rate': sampling_rate,
    }

    # Initialize thin pulse identification config
    # trunc_gap             : The window in hours from the edges of the LAP where peaks should not lie
    # lap_half_width        : Half of the size of 1 LAP configured in hours
    # thin_pulse_amp_std    : Limit to variation allowed (in Wh) around the thin pulse to call it a LAP
    # min_thin_pulse_amp    : Minimum required amplitude (in Wh) to be considered as thin pulse
    # max_thin_pulse_amp    : Maximum allowed amplitude (in Wh) to be considered as thin pulse
    # derivative_mask       : Convolution kernel to identify single peaks
    # amplitude_mask        : Convolution kernel to calculate the amplitude of single peaks

    vacation_config['thin_pulse_id'] = {
        'trunc_gap': 1,
        'lap_half_width': 2.5,
        'thin_pulse_amp_std': 80,
        'min_thin_pulse_amp': 120,
        'max_thin_pulse_amp': 1200,
        'derivative_mask': np.flipud([-1, 0, 1]),
        'amplitude_mask': np.flipud([-1 / 2, 1, -1 / 2]),
    }

    # Initialize thin pulse removal config
    # min_day_pts_for_removal   : Minimum peaks a day should have to be considered for removal
    # max_day_pts_for_removal   : Maximum peaks allowed in a day to be considered for removal
    # max_peak_diff_hrs         : Maximum difference between 2 consecutive peaks allowed on a day in hrs
    # med_peak_diff_low_thr     : Minimum median gap between peaks in hours needed in a day
    # med_peak_diff_high_thr    : Maximum median gap between peaks in hours allowed in a day
    # min_days_for_removal      : Minimum days to be candidates for removal so that it happens
    # min_peaks_for_removal     : Minimum peaks to be candidates for removal so that it happens

    vacation_config['thin_pulse_rem'] = {
        'min_day_pts_for_removal': 3,
        'max_day_pts_for_removal': 8,
        'max_peak_diff_hrs': 13,
        'med_peak_diff_low_thr': 3,
        'med_peak_diff_high_thr': 11,
        'min_days_for_removal': 2,
        'min_peaks_for_removal': 10,
    }

    # Initialize power computation config
    # max_nan_ratio_for_power   : Maximum ratio of nan values to total in the day allowed to compute power

    vacation_config['compute_power'] = {
        'max_nan_ratio_for_power': 0.5,
    }

    # Initialize baseload computation config
    # min_value_for_baseload    : Minimum value of consumption (in Wh) that will be considered for baseload computation
    # baseload_window_size      : The moving window in which the baseload is averaged
    # baseload_percentile       : The percentile to be taken on each day to be called baseload

    vacation_config['compute_baseload'] = {
        'min_value_for_baseload': 5,
        'baseload_window_size': 14,
        'baseload_percentile': 1,
    }

    if is_europe:

        # Config sections specific to the Europe geography

        # Initialize probable vacation day identification config
        # bl_lv_1       : Level 1 baseload value for assigning probable day thresholds (in Wh)
        # bl_lv_2       : Level 2 baseload value for assigning probable day thresholds (in Wh)
        # bl_lv_3       : Level 3 baseload value for assigning probable day thresholds (in Wh)
        # bl_lv_1_thr   : Probable day threshold value at Level 1 (in Wh)
        # bl_lv_2_thr   : Probable day threshold value at Level 2 (in Wh)
        # bl_lv_3_thr   : Probable day threshold value at Level 3 (in Wh)

        vacation_config['probable_day'] = {
            'bl_lv_1': 40,
            'bl_lv_2': 100,
            'bl_lv_3': 1200,
            'bl_lv_1_thr': 75,
            'bl_lv_2_thr': 125,
            'bl_lv_3_thr': 700,
        }

        # Initialize window power check config
        # bl_lv_1           : Level 1 baseload value for assigning power check thresholds (in Wh)
        # bl_lv_2           : Level 2 baseload value for assigning power check thresholds (in Wh)
        # bl_lv_3           : Level 3 baseload value for assigning power check thresholds (in Wh)
        # slide_size        : The period of time we shift the window by in hours
        # window_size       : The width of the window used to compute power for check in hours
        # bl_lv_1_thr       : Power check threshold value at Level 1 (in Wh)
        # bl_lv_2_thr       : Power check threshold value at Level 2 (in Wh)
        # bl_lv_3_thr       : Power check threshold value at Level 3 (in Wh)
        # bl_lv_1_thr_tp    : Power check threshold value at Level 1 for thin pulse removal users (in Wh)
        # bl_lv_2_thr_tp    : Power check threshold value at Level 2 for thin pulse removal users (in Wh)
        # bl_lv_3_thr_tp    : Power check threshold value at Level 3 for thin pulse removal users (in Wh)
        # bl_lv_4_slope     : Slope at which the threshold increases in level 4

        vacation_config['window_power_check'] = {
            'bl_lv_1': 40,
            'bl_lv_2': 100,
            'bl_lv_3': 300,
            'slide_size': 2,
            'window_size': 4,
            'bl_lv_1_thr': 100,
            'bl_lv_2_thr': 115,
            'bl_lv_3_thr': 130,
            'bl_lv_1_thr_tp': 110,
            'bl_lv_2_thr_tp': 125,
            'bl_lv_3_thr_tp': 140,
            'bl_lv_4_slope': 1.2,
        }

        # Initialize confirmed vacation day sliding power mean thresholds config
        # bl_lv_1       : Level 1 baseload value for assigning power check thresholds (in Wh)
        # bl_lv_2       : Level 2 baseload value for assigning power check thresholds (in Wh)
        # bl_lv_3       : Level 3 baseload value for assigning power check thresholds (in Wh)
        # bl_lv_4       : Level 4 baseload value for assigning power check thresholds (in Wh)
        # bl_lv_5       : Level 5 baseload value for assigning power check thresholds (in Wh)
        # bl_lv_6       : Level 6 baseload value for assigning power check thresholds (in Wh)
        # bl_lv_1_thr   : Power mean threshold value at Level 1 (in Wh)
        # bl_lv_2_thr   : Power mean threshold value at Level 2 (in Wh)
        # bl_lv_3_thr   : Power mean threshold value at Level 3 (in Wh)
        # bl_lv_4_thr   : Power mean threshold value at Level 4 (in Wh)
        # bl_lv_5_thr   : Power mean threshold value at Level 5 (in Wh)
        # bl_lv_6_thr   : Power mean threshold value at Level 6 (in Wh)
        # bl_lv_7_thr   : Power mean threshold value at Level 7 (in Wh)

        vacation_config['confirm_power_mean'] = {
            'bl_lv_1': 100,
            'bl_lv_2': 200,
            'bl_lv_3': 400,
            'bl_lv_4': 600,
            'bl_lv_5': 800,
            'bl_lv_6': 1200,
            'bl_lv_1_thr': 120,
            'bl_lv_2_thr': 140,
            'bl_lv_3_thr': 170,
            'bl_lv_4_thr': 300,
            'bl_lv_5_thr': 350,
            'bl_lv_6_thr': 400,
            'bl_lv_7_thr': 450,
        }

        # Initialize confirmed vacation day deviation thresholds config
        # bl_lv_1                   : Level 1 baseload value for assigning std dev thresholds (in Wh)
        # bl_lv_2                   : Level 2 baseload value for assigning std dev thresholds (in Wh)
        # bl_lv_1_thr               : Std dev threshold value at Level 1 (in Wh)
        # bl_lv_2_thr               : Std dev threshold value at Level 2 (in Wh)
        # std_dev_thr               : Std dev threshold values by default (in Wh)
        # top_3_dev_thr             : Deviation threshold for top 3 deviation values (in Wh)
        # top_dev_max_ratio         : Maximum ratio between highest deviation and std dev allowed
        # top_2_dev_max_ratio       : Maximum ratio between sum of top 2 deviations and std dev allowed
        # min_stddev_for_dev_reject : Minimum std dev value for ratios to apply (in Wh)

        vacation_config['confirm_deviation'] = {
            'bl_lv_1': 400,
            'bl_lv_2': 700,
            'bl_lv_1_thr': 80,
            'bl_lv_2_thr': 100,
            'std_dev_thr': 50,
            'top_3_dev_thr': 80,
            'top_dev_max_ratio': 3,
            'top_2_dev_max_ratio': 3.3,
            'min_stddev_for_dev_reject': 20,
        }

    else:

        # Config sections default used everywhere other than the Europe geography

        # Initialize probable vacation day identification config
        # bl_lv_1       : Level 1 baseload value for assigning probable day thresholds (in Wh)
        # bl_lv_2       : Level 2 baseload value for assigning probable day thresholds (in Wh)
        # bl_lv_3       : Level 3 baseload value for assigning probable day thresholds (in Wh)
        # bl_lv_1_thr   : Probable day threshold value at Level 1 (in Wh)
        # bl_lv_2_thr   : Probable day threshold value at Level 2 (in Wh)
        # bl_lv_3_thr   : Probable day threshold value at Level 3 (in Wh)

        vacation_config['probable_day'] = {
            'bl_lv_1': 40,
            'bl_lv_2': 100,
            'bl_lv_3': 1200,
            'bl_lv_1_thr': 100,
            'bl_lv_2_thr': 200,
            'bl_lv_3_thr': 700,
        }

        # Initialize window power check config
        # bl_lv_1           : Level 1 baseload value for assigning power check thresholds (in Wh)
        # bl_lv_2           : Level 2 baseload value for assigning power check thresholds (in Wh)
        # bl_lv_3           : Level 3 baseload value for assigning power check thresholds (in Wh)
        # slide_size        : The period of time we shift the window by in hours
        # window_size       : The width of the window used to compute power for check in hours
        # bl_lv_1_thr       : Power check threshold value at Level 1 (in Wh)
        # bl_lv_2_thr       : Power check threshold value at Level 2 (in Wh)
        # bl_lv_3_thr       : Power check threshold value at Level 3 (in Wh)
        # bl_lv_1_thr_tp    : Power check threshold value at Level 1 for thin pulse removal users (in Wh)
        # bl_lv_2_thr_tp    : Power check threshold value at Level 2 for thin pulse removal users (in Wh)
        # bl_lv_3_thr_tp    : Power check threshold value at Level 3 for thin pulse removal users (in Wh)
        # bl_lv_4_slope     : Slope at which the threshold increases in level 4

        vacation_config['window_power_check'] = {
            'bl_lv_1': 40,
            'bl_lv_2': 100,
            'bl_lv_3': 300,
            'slide_size': 2,
            'window_size': 4,
            'bl_lv_1_thr': 110,
            'bl_lv_2_thr': 120,
            'bl_lv_3_thr': 150,
            'bl_lv_1_thr_tp': 120,
            'bl_lv_2_thr_tp': 140,
            'bl_lv_3_thr_tp': 170,
            'bl_lv_4_slope': 0.5,
        }

        # Initialize confirmed vacation day sliding power mean thresholds config
        # bl_lv_1       : Level 1 baseload value for assigning power check thresholds (in Wh)
        # bl_lv_2       : Level 2 baseload value for assigning power check thresholds (in Wh)
        # bl_lv_3       : Level 3 baseload value for assigning power check thresholds (in Wh)
        # bl_lv_4       : Level 4 baseload value for assigning power check thresholds (in Wh)
        # bl_lv_5       : Level 5 baseload value for assigning power check thresholds (in Wh)
        # bl_lv_6       : Level 6 baseload value for assigning power check thresholds (in Wh)
        # bl_lv_1_thr   : Power mean threshold value at Level 1 (in Wh)
        # bl_lv_2_thr   : Power mean threshold value at Level 2 (in Wh)
        # bl_lv_3_thr   : Power mean threshold value at Level 3 (in Wh)
        # bl_lv_4_thr   : Power mean threshold value at Level 4 (in Wh)
        # bl_lv_5_thr   : Power mean threshold value at Level 5 (in Wh)
        # bl_lv_6_thr   : Power mean threshold value at Level 6 (in Wh)
        # bl_lv_7_thr   : Power mean threshold value at Level 7 (in Wh)

        vacation_config['confirm_power_mean'] = {
            'bl_lv_1': 100,
            'bl_lv_2': 200,
            'bl_lv_3': 400,
            'bl_lv_4': 600,
            'bl_lv_5': 800,
            'bl_lv_6': 1200,
            'bl_lv_1_thr': 120,
            'bl_lv_2_thr': 140,
            'bl_lv_3_thr': 170,
            'bl_lv_4_thr': 230,
            'bl_lv_5_thr': 300,
            'bl_lv_6_thr': 400,
            'bl_lv_7_thr': 450,
        }

        # Initialize confirmed vacation day deviation thresholds config
        # std_dev_thr       : Std dev threshold values by default (in Wh)
        # top_3_dev_thr     : Deviation threshold for top 3 deviation values (in Wh)

        vacation_config['confirm_deviation'] = {
            'std_dev_thr': 80,
            'top_3_dev_thr': 110,
        }

    # Initialize confidence config for static values that are used to directly assign or compare
    # not_probable_conf         : Confidence value for day marked as not probable type 1
    # wild_card_conf_thr        : Confidence value threshold for day to qualify for wild card entry
    # vacation_selection_thr    : Confidence value threshold for day to be marked as type 1 vacation

    vacation_config['static_confidence'] = {
        'not_probable_conf': 0,
        'wild_card_conf_thr': 0.26,
        'vacation_selection_thr': 0.5,
    }

    # Initialize confidence config for power check rejected days
    # max_conf      : Maximum confidence value allowed
    # delta_wt      : The weight to scale down delta from 0 to 1
    # scale_const   : The denominator constant to scale delta to 0 to 1

    vacation_config['power_check_confidence'] = {
        'max_conf': 0.3,
        'delta_wt': 0.15,
        'scale_const': 50,
    }

    # Initialize confidence config for final check rejected days
    # max_conf              : Maximum confidence value allowed
    # delta_wt              : The weight to scale down delta from 0 to 1
    # pwr_mean_wt           : The weight to scale down delta from 0 to 1
    # pwr_mean_const        : The denominator constant to scale delta to 0 to 1
    # std_dev_wt            : The weight to scale down delta from 0 to 1
    # std_dev_const         : The denominator constant to scale delta to 0 to 1
    # max_dev_wt            : The weight to scale down delta from 0 to 1
    # max_dev_const         : The denominator constant to scale delta to 0 to 1
    # max_3_dev_perc_wt     : The weight to scale down delta from 0 to 1
    # max_3_dev_perc_const  : The denominator constant to scale delta to 0 to 1

    vacation_config['final_uns_confidence'] = {
        'max_conf': 0.6,
        'delta_wt': 0.2,
        'pwr_mean_wt': 0.1,
        'pwr_mean_const': 20,
        'std_dev_wt': 0.1,
        'std_dev_const': 10,
        'max_dev_wt': 0.5,
        'max_dev_const': 10,
        'max_3_dev_perc_wt': 0.3,
        'max_3_dev_perc_const': 40,
    }

    # Initialize confidence config for final check selected days
    # min_conf          : Minimum confidence value allowed
    # delta_wt          : The weight to scale down delta from 0 to 1
    # pwr_mean_wt       : The weight to scale down delta from 0 to 1
    # pwr_mean_const    : The denominator constant to scale delta to 0 to 1
    # std_dev_wt        : The weight to scale down delta from 0 to 1
    # std_dev_const     : The denominator constant to scale delta to 0 to 1
    # max_dev_wt        : The weight to scale down delta from 0 to 1

    vacation_config['final_sel_confidence'] = {
        'min_conf': 0.7,
        'delta_wt': 0.3,
        'pwr_mean_wt': 0.2,
        'pwr_mean_const': 20,
        'std_dev_wt': 0.7,
        'std_dev_const': 30,
        'max_dev_wt': 0.1,
    }

    # Initialize config for type 2 vacation
    # conf_val          : Confidence value for type 2 vacation day
    # max_nan_ratio     : Maximum ration of points in day allowed to be nan from disconnection

    vacation_config['type_2'] = {
        'conf_val': 1,
        'max_nan_ratio': 0.25,
    }

    return vacation_config
