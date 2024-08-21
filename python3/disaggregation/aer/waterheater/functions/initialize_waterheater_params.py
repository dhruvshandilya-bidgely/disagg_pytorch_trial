"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Call the initialize_water_heater_params function to load the required parameters for the water heater algorithm
"""

# Import python packages

import pytz
import numpy as np
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_timezone(timezone_string):
    """
    Parameters:
        timezone_string     (int)               : Timezone of the user

    Returns:
        timezone            (int)               : Timezone of the user in hours wrt GMT
    """

    # Get the timezone offset number from string

    timezone_offset = datetime.now(pytz.timezone(timezone_string))

    # Convert offset to number of hours

    timezone = timezone_offset.utcoffset().total_seconds() / Cgbdisagg.SEC_IN_HOUR

    return timezone


def initialize_water_heater_params(disagg_input_object, global_config):
    """
    Parameters:
        disagg_input_object     (dict)          : Dictionary containing all inputs
        global_config           (dict)          : Dictionary containing all input configuration

    Returns:
        config                  (dict)          : Dictionary containing all required parameters for water heater
    """


    # Timed water heater config

    """
    "min_amplitude"                 : Minimum power rating for a timed water heater
    "max_amplitude"                 : Maximum power rating for a timed water heater
    "rounding_threshold"            : Threshold to check for rounding in data
    "rounding_delta"                : Allowed deviation for rounded data
    "minimum_box_points"            : Minimum number of data points for box fitting
    "baseload_window"               : Window size in hours to remove baseload
    "detection_threshold"           : Minimum edge proportion for initial detection
    "wtr_detection_threshold"       : Minimum edge proportion for initial detection (winter only)
    "start_end_ratio"               : The minimum fraction of start to end proportion for start type
    "std_thres"                     : Deviation of the edge proportion
    "std_thres_delta"               : Change in deviation of the edge proportion for rounded data
    "raw_roll_threshold"            : Raw edge proportion vs rolled edge proportion
    "raw_roll_threshold_wtr"        : Raw edge proportion vs rolled edge proportion (winter only)
    "raw_count_threshold"           : Minimum threshold for (raw * roll) proportion
    "raw_count_threshold_wtr"       : Minimum threshold for (raw * roll) proportion (winter only)
    "start_mean_threshold_major"    : Allowed number of water heater start/stop per day (high proportion)
    "start_mean_threshold_minor"    : Allowed number of water heater start/stop per day (low proportion)
    "minimum_fraction_idx"          : Minimum number of hours to compare the highest fraction
    "max_count_bar"                 : High/Low proportion boundary
    "elbow_threshold"               : Minimum elbow threshold for filtering out hours
    "max_duration_bound"            : Maximum allowed duration of run
    "lower_duration_bound"          : Lower bound for run duration compared to median
    "upper_duration_bound"          : Upper bound for run duration compared to median
    "bin_width"                     : Width of single bin (in Wh) for hourly proportion
    "min_rounding_bar"              : Minimum rounding bar
    "rounding_balance"              : Difference limit for rounding energy check
    "insignificant_run_threshold"   : Minor vs Major boundary proportion threshold
    "energy_fraction_threshold"     : Energy fraction threshold for adjacent hours of main run hours
    "energy_threshold"              : Max energy fraction threshold
    "minimum_hour_gap"              : Minimum gap in hours between two runs
    "box_hod_min_threshold"         : Minimum threshold for box energy proportion
    "wtr_box_hod_min_threshold"     : Minimum threshold for box energy proportion (winter only)
    "vicinity_days_bound"           : Days threshold to filter out noise days
    "max_limits"                    : The boundaries for various proportion range
    "adjacent_hours"                : Number of adjacent hours in a run to filter noise
    "fraction_limits"               : Proportion limits for High/Medium/Low
    "consumption_limits"            : Consumption limits for High/Medium/Low
    "amplitude_limits"              : Variation limits of timed water heater amplitude
    "required_bc_count"             : Minimum number of bill cycles required to filter odd ones
    "amplitude_bar"                 : Highest percentile allowed for timed water heater amplitude
    "valid_hours_range"             : Number of hours to consider for confidence boundary
    "set_point"                     : Default set point for the users
    "diff_set_point"                : Default set point deviation for the users
    "num_transition_months"         : Number of transition months allowed
    "seasons"                       : Default months for different seasons
            "wtr"                       : Default winter months
            "itr"                       : Default intermediate months
            "smr"                       : Default summer months

    "mid_temperature_range"         : Default Intermediate season bounds
    "pp_thresholds"                 : Pool pump thresholds
    "season_code"                   : Mapping of season string with integer
    """

    timed_waterheater_config = {
        "min_amplitude": 1000,
        "max_amplitude": 6000,
        "rounding_threshold": 100,
        "rounding_delta": 0.01,
        "minimum_box_points": 2,
        "baseload_window": 8,
        "detection_threshold": 0.40,
        "wtr_detection_threshold": 0.60,
        "start_end_ratio": 0.950,
        "std_thres": 0.090,
        "std_thres_delta": 0.030,
        "raw_roll_threshold": 0.350,
        "raw_roll_threshold_wtr": 0.50,
        "raw_count_threshold": 0.20,
        "raw_count_threshold_wtr": 0.30,
        "start_mean_threshold_major": 6.0,
        "start_mean_threshold_minor": 4.0,
        "minimum_fraction_idx": 4,
        "max_count_bar": 0.80,
        "elbow_threshold": 0.20,
        "max_duration_bound": 3,
        "lower_duration_bound": 0.50,
        "upper_duration_bound": 2,
        "bin_width": 50,
        "min_rounding_bar": 100,
        "rounding_balance": 0.010,
        "insignificant_run_threshold": 0.70,
        "energy_fraction_threshold": 0.80,
        "energy_threshold": 0.10,
        "minimum_hour_gap": 6,
        "box_hod_min_threshold": 0.40,
        "wtr_box_hod_min_threshold": 0.65,
        "vicinity_days_bound": 3,
        "max_limits": [0.90, 0.75, 0.50, 0.35],
        "adjacent_hours": [1, 2, 3, 4],
        "fraction_limits": [0.40, 0.60],
        "consumption_limits": [0.10, 0.25, 0.60],
        "required_bc_count": 3,
        "amplitude_limits": [0.50, 2.0],
        "amplitude_bar": 90,
        "valid_hours_range": 2,
        "set_point": 65,
        "diff_set_point": 12,
        "num_transition_months": 3,
        "seasons": {
            "wtr": [11, 12, 1, 2],
            "itr": [3, 4, 9, 10],
            "smr": [5, 6, 7, 8]
        },
        "mid_temperature_range": [56, 74],
        "pp_thresholds": {
            "pp_area_bar": 1500,
            "pp_duration_bar": 2.4,
            "pp_confidence_bar": 70
        },
        "season_code": {
            "wtr": 1,
            "itr": 2,
            "smr": 3
        }
    }

    # Non-timed waterheater detection config

    """
    "probability_thresholds"        : Lower and Upper detection probability bounds
    "min_thin_pulse_amp"            : Minimum energy per data point for thin pulse
    "max_thin_pulse_amp"            : Maximum energy per data point for thin pulse
    "thin_pulse_amp_std"            : Minimum energy difference of thin pulse from adjacent points
    "lap_half_width"                : Half duration of LAP(Low Activity Periods) in hours
    "end_gap"                       : Duration in hours at each LAP end that is checked for peaks
    "thin_pulse_half_gap"           : Half of the minimum gap expected between two consecutive peaks
    "minimum_box_size"              : The minimum duration (in seconds) for box fitting
    "night_bounds"                  : Start and end hours of night hours
    "allowed_days_gap"              : Gap in days allowed for consistency factor
    "amplitude_mask"                : Convolution filter to find every sharp peak
    "derivative_mask"               : Convolution filter to find peak energy difference with edges
    "peak_fraction"                 : Threshold for peaks every hour fraction
    "peak_lap_ratio"                : Peaks to lap ratio at each hour of day
    "wh_of_total_min"               : Minimum water heater consumption out of total (in percent)
    "average_consumption"           : Expected minimum water heater consumption per bill cycle (in Watts)
    "consumption_probability"       : Detection probability threshold for low consumption
    "timed_detection_bounds"        : Timed water heater confidence bounds
    "non_timed_detection_bounds"    : Non-timed water heater confidence bounds
    """

    non_timed_detection_config = {
        "probability_thresholds": [0.450, 0.60],
        "min_thin_pulse_amp": 250,
        "max_thin_pulse_amp": 1200,
        "thin_pulse_amp_std": 150,
        "lap_half_width": 2.50,
        "end_gap": 1.50,
        "peak_points": 2,
        "thin_pulse_half_gap": 1.5,
        "minimum_box_size": 1800.0,
        "night_bounds": [22, 7],
        "allowed_days_gap": 10,
        "amplitude_mask": np.flipud([-1 / 2, 1, -1 / 2]),
        "derivative_mask": np.flipud([-1, 0, 1]),
        "peak_fraction": 0.40,
        "peak_lap_ratio": 5,
        "wh_of_total_min": 5,
        "average_consumption": 50000,
        "consumption_probability": 0.65,
        "timed_detection_bounds": [0.75, 0.65, 0.50, 0.40, 0.30],
        "non_timed_detection_bounds": [0.95, 0.90, 0.85, 0.75, 0.65]
    }

    # Non-timed waterheater detection config

    """
    "max_runs"                  : Maximum number of allowed water heater usages per day
    "max_hours"                 : Maximum number of usage hours allowed per day
    "max_fat_pulse_duration"    : Maximum duration of single water heater usage (in hours)
    "min_inter_pulse_gap"       : Default gap in hours between consecutive thin pulses
    "min_thin_pulse_amp"        : Minimum energy per data point for thin pulse
    "max_thin_pulse_amp"        : Maximum energy per data point for thin pulse
    "min_duration_factor"       : Minimum duration of thin pulse factor
    "max_duration_factor"       : Maximum duration of thin pulse factor
    "min_fat_pulse_duration"    : Minimum duration of fat pulse (in minutes)
    "thin_pulse_dur_std"        : Variation of thin pulse duration (in minutes)
    "thin_pulse_amp_std"        : Minimum energy difference of thin pulse from adjacent points
    "allowed_peaks_count"       : Allowed number of peaks in a thin pulse filter window
    "peak_width_bounds"         : The allowed width of peaks for fat pulse usage
    "min_peaks_fraction"        : Minimum fraction of peaks required in a month to consider
    "peak_daily_proportion"     : Base fraction for valid daily counts
    "min_peak_distance"         : Minimum difference between fat usage bands (in hours)
    "peak_height"               : Fraction of peak height accepted adjacent to a fat pulse peak
    "fat_bound_limit"           : Min bound for fat pulse based on ideal fat pulse size
    "noise_window_size"         : Size of the window to filter noise fat pulses
    "num_compare_bc"            : Number of bill cycles check for filtering noise
    "peak_filter"               : Convolution filter to find every sharp peak
    "diff_filter"               : Convolution filter to find peak energy difference with edges
    "fat_duration_limit"        : Maximum duration in hours of single water heater usage
    "consumption_limit"         : Consumption check threshold for noise bill cycles
    "default_hour"              : Default fat pulse hour if no peaks detected
    "night_hours"               : Hours of the day designated as night
    "night_buffer"              : Number of nearest hours to peak night bounds
    "night_proportion"          : Proportion of night hours among detected fat hours
    "difference_days"           : Number of minimum days for within bill cycle comparison
    "default_scale_factor"      : Default scaling factor for thin and fat pulse
    "default_feature_value"     : Default feature value for seasons that are not present
    "thin_upscale_factor"       : Thin pulse upscale for high sampling rate
    "fat_upscale_factor"        : Fat pulse upscale for high sampling rate
    "wtr_consumption_threshold" : Percentile threshold to get the thin/fat pulse amplitude in winter
    "capping_buffer"            : Buffer allowed while capping noisy consumption
    "min_amp_cap"               : Minimum amplitude capping to calculate a lower range of amplitude allowed
    "max_amp_cap"               : Maximum amplitude capping to calculate a higher range of amplitude allowed
    "fat_min_cap"               : Minimum amplitude capping to calculate a lower range of fat pulse amplitude allowed
    "fat_max_cap"               : Maximum amplitude capping to calculate a higher range of fat pulse amplitude allowed
    "start_hod_min_thr"         : Minimum percentage of days with same start hour of the day required in a bill cycle
    "year_hod_thr"              : Minimum percentage of days with the same start hour of the day required thoughout the year
    "start_hod_thr"             : Percentage of days with the same start hour of the day required in a bill cycle
    "max_fat_runs"              : Number of fat pulse runs allowed in a day
    "fat_noise_hod_thr"         : Minimum percentage of days with the same start hour of day required for fat pulses
    "fat_noise_max_hours"       : Maximum number of hours allowed in a day for fat pulses
    "percentage_removal"        : Percentage removal of noise
    "avg_consumption_thr"       : Threshold to identify bill cycles with high consumption in comparision to the rest
    """

    non_timed_estimation_config = {
        "max_runs": 5,
        "max_hours": 7,
        "max_fat_pulse_duration": 4,
        "min_inter_pulse_gap": 4,
        "min_thin_pulse_amp": 250,
        "max_thin_pulse_amp": 1200,
        "min_duration_factor": 0.70,
        "max_duration_factor": 1.70,
        "min_fat_pulse_duration": 20,
        "thin_pulse_dur_std": 0.30,
        "thin_pulse_amp_std": 150,
        "allowed_peaks_count": 2,
        "peak_width_bounds": [0, 5],
        "min_peaks_fraction": 0.20,
        "peak_daily_proportion": 0.30,
        "min_peak_distance": 3,
        "peak_height": 0.50,
        "fat_bound_limit": 0.60,
        "noise_window_size": 3,
        "num_compare_bc": 3,
        "peak_filter": np.flipud([-1 / 2, 1, -1 / 2]),
        "diff_filter": np.flipud([-1, 0, 1]),
        "fat_duration_limit": 4,
        "consumption_limit": 0.10,
        "default_hour": 9,
        "night_hours": [23, 0, 1, 2, 3, 4, 5],
        "night_buffer": 1,
        "night_proportion": 0.40,
        "difference_days": 5,
        "default_scale_factor": 1,
        "default_feature_value": 0,
        "thin_upscale_factor": 1.50,
        "fat_upscale_factor": 2,
        "wtr_consumption_threshold": 90,
        "capping_buffer": 100,
        "min_amp_cap": 0.8,
        "max_amp_cap": 1.2,
        "fat_min_cap": 0.7,
        "fat_max_cap": 1.3,
        "start_hod_min_thr": 0.1,
        "year_hod_thr": 0.5,
        "start_hod_thr": 0.8,
        "max_fat_runs": 3,
        "fat_noise_hod_thr": 0.7,
        "fat_noise_max_hours": 4,
        "percentage_removal": 0.2,
        "avg_consumption_thr": 2
    }

    # Definition for all the config parameters

    """
    "min_sampling_rate"     : The minimum sampling rate to be used for algorithm
    "min_num_days"          : Minimum number of valid data days for historical/incremental run
    "timezone"              : The timezone of the user
    "timezone_hours"        : Hours offset of the timezone from GMT
    "num_transition_months" : Number of transition months allowed
    "seasons"               : The three possible seasons for every user
    "season_code"           : Numerical encoding for each season

    "thermostat_wh"         : Parameters for Non-timed storage Thermostat based water heater
        "baseload_window"       : Window size in hours to remove baseload
        "detection"             : Parameters for detection
        "estimation"            : Parameters for estimation
        "season_code"           : Mapping of season string with integer

    "timed_wh"              : Parameters for Timed storage water heater
    "block_wh_types"        : Types of WHs to block
    """

    # Combine the parameters for timed and non-timed water heater

    config = dict({
        "min_sampling_rate": 900,
        "min_num_days": 200,
        "timezone": disagg_input_object["home_meta_data"]["timezone"],
        "timezone_hours": get_timezone(disagg_input_object["home_meta_data"]["timezone"]),
        "num_transition_months": 3,
        "seasons": ["wtr", "itr", "smr"],
        "season_code":
            {
                1: "wtr",
                2: "itr",
                3: "smr"
            },

        "thermostat_wh": {
            "baseload_window": 4,
            "detection": non_timed_detection_config,
            "estimation": non_timed_estimation_config,
            "season_code": {
                "wtr": 1,
                "itr": 2,
                "smr": 3
            }
        },

        "timed_wh": timed_waterheater_config,

        "block_wh_types": ['GAS', 'Gas', 'gas', 'PROPANE', 'Propane', 'propane']
    })

    # Adding default info from global_config to wh_config for easy accessibility

    config["uuid"] = global_config.get("uuid")
    config["pilot_id"] = global_config.get("pilot_id")
    config["disagg_mode"] = global_config.get("disagg_mode")
    config["sampling_rate"] = global_config.get("sampling_rate")

    if global_config.get('sampling_rate') == Cgbdisagg.SEC_IN_HOUR:
        timed_waterheater_config['min_amplitude'] = 750
        timed_waterheater_config['max_amplitude'] = 5750
        timed_waterheater_config['wtr_detection_threshold'] = 0.70
        timed_waterheater_config['raw_count_threshold'] = 0.25
        timed_waterheater_config['raw_count_threshold_wtr'] = 0.35
        timed_waterheater_config['insignificant_run_threshold'] = 0.55

        non_timed_detection_config['thin_pulse_amp_std'] = 120
        non_timed_estimation_config['thin_pulse_amp_std'] = 120

    return config
