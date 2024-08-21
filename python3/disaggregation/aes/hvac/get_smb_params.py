"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to fetch SMB related constant parameters
"""


# No Imports here


def get_smb_params():
    """
    Function returns the static parameters used in hvac only
    Returns:
        static_param (dict) : Dictionary containing apll the static parameters
    """

    smb_params = {
        'utility': {
            'binary_search_cut_gap': 10,
            'large_number_constant': 123456789,
            'kilo': 1000,
            'residue_col': -2
        },
        'day_info': {
            'std_limit': 15,
            'std_arm': 0.6,
            'open_time_condition': -0.20,
            'close_time_condition': 0.25,
            'open_time_limit': 3
        },
        'month_info': {
            'low_limit_arm': 0.25,
            'open_arm': 1.5,
            'close_arm': 1.5,
            'open_low_limit': 134,
            'close_low_limit': 110,
            'merge_limit': 150,
            'net_conf_limit': 200,
            'high_conf_identifier': 200,
            'long_hour_limit': 21,
            'long_hour_id': 16,
            'false_open_hour': 2,
            'false_close_hour': 6,
            'consistency_measure': 2.6,
            'low_energy_std_arm': 2,
            'low_energy_smb_limit': 1200,
            'high_energy_smb_limit': 3600,
            'perc_work_hour_in_day': 0.95,
            'perc_work_hour_in_year': 0.05,
            'low_consumption_open_days': 15,
            'off_days_count': 2,
            'flatten_first_hours': 2,
            'median_strength': 2,
        },
        'valid_day': {
            'std_condition': 0.5,
            'outer_limit_arm': 0.4,
            'frac_limit_for_day': 0.1,
            'frac_of_work_in_day': 0.3
        },
        'operational': {
            'hour_median_condition': 0.01,
            'hour_median_low_lim': 0.5
        },
        'postprocess': {
            'min_residue_gap': 10,
            'underestimation_arm': 0.8,
            'overestimation_arm': 1.0,
            'cdd_reference_temp': 65,
            'hdd_reference_temp': 60,
            'work_hour_tolerance': 2,
            'ac_fp_low_limit': 40,
            'sh_fp_low_limit': 50
        },
        'extra-ao': {
            'min_factor_grey': 3,
            'grey_median_factor': 2
        },
        'input_df_col': {
            'external_light': -9,
            'bl': -8,
            'x-ao': -7,
            'op': -6,
            'ac_open': -5,
            'ac_close': -4,
            'sh_open': -3,
            'sh_close': -2,
            'oth': -1
        }
    }

    return smb_params
