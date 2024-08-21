"""
Author - Anand Kumar Singh
Date - 19th Feb 2020
Call the initialize_solar_params function to load the required parameters for the solar disaggregation algorithm
"""

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def init_solar_estimation_config(global_config):
    """
        Initialise parameters for solar config

        Parameters:
            global_config          (dict)                : Global configuration set in the pipeline

        Returns:
            config                 (dict)                : Dict with all the values to be used within solar module
    """
    config = dict({'uuid': global_config.get('uuid'),
                   'pilot_id': global_config.get('pilot_id'),
                   'timezone': None,
                   'latitude': None,
                   'longitude': None,
                   'disagg_mode': None,

                   # Thresholds to be used when negative data is available

                   'neg_cap_estimation': {'r_square_threshold': {'hi_r_square': 0.7,
                                                                 'mid_r_square': 0.6,
                                                                 'low_r_square': 0.5,
                                                                },
                                          'minimum_good_days': 14
                                         },

                   # Parameters used for normalising feature vector
                   'normalisation_params': {'max_irradiance': 1000,
                                            'upper_percentile': 0.98,
                                            'lower_percentile': 0.02},

                   'sampling_rate': global_config.get('sampling_rate'),

                   # Variable to enable type of CSR values
                   'estimation_data_type': 'neg_consumption',

                   # Columns to be used in feature vector
                   'features_from_input_data': [Cgbdisagg.INPUT_TEMPERATURE_IDX,
                                                Cgbdisagg.INPUT_SKYCOV_IDX,
                                                Cgbdisagg.INPUT_WIND_SPD_IDX,
                                                Cgbdisagg.INPUT_CONSUMPTION_IDX],

                   # Column index for solar potential and solar generation columns
                   'solar_potential_column': Cgbdisagg.INPUT_DIMENSION,
                   'solar_generation_column': (Cgbdisagg.INPUT_DIMENSION + 1),

                   # Normalisation thresholds, will be used in mtd to save time
                   'normalisation_thresholds': {'temp': {'lower': None,
                                                         'upper': None},
                                                'sky_cover': {'lower': None,
                                                              'upper': None},
                                                'wind': {'lower': None,
                                                         'upper': None},
                                                'consumption': {'lower': None,
                                                                'upper': None},
                                               },

                   # Baseload difference for valid solar day
                   'baseload_threshold': 400,

                   # XGB model for solar potential estimation
                   'estimation_model': None,

                   # Capacity stored in existing HSM
                   'previous_capacity': None,

                   # Upper and lower capacity coefficients
                   'capacity_params': {'lower_capacity_coefficient': 0.8408933218644109,
                                       'upper_capacity_coefficient': 0.21532688158991533,
                                       'upper_limit_for_capacity': 0.9},

                   # Residual Capacity from HSM
                   'residual_capacity_array': None,

                   # List of bill cycles for which results are to be written
                   'out_bill_cycles' : None,
                   'debug_mode': False
                   })
    return config


def init_solar_detection_config(global_config, disagg_input_object):
    """
    Initialize the parameters required for running solar detection

    Parameters:
        global_config       (dict)              : Dictionary containing global config
        disagg_input_object (dict)              : Dictionary containing information about disaggregation run

    Returns:
        solar_detection_config           (dict)              : Dictionary containing all configuration variables for solar
    """

    so_config = {
        'uuid': global_config.get('uuid'),
        'pilot': global_config.get('pilot_id'),
        'timezone': disagg_input_object.get('home_meta_data').get('timezone'),
        'latitude': None,
        'longitude': None,


        # Solar module universal config

        'so_min_data_req': 90,
        'max_instances': 11,

        # Indices of consumption and sunlight array in CNN input 3-d matrix

        'consumption_arr_idx': 0,
        'sunlight_arr_idx': 1,

        # Parameters of solar main solar detection function

        'solar_disagg': {
            # CNN threshold for detection of solar user
            'detection_threshold': 0.5,
            # LightGBM threshold for detection of solar user
            'lgbm_threshold':0.5,
            # If negative values during suntime beyond this ratio -> solar user
            'neg_suntime_thresh': 0.1,
            # Ratio of zero values for enabling/disabling lightgbm
            'zero_suntime_thresh': 0.05,
            # Checks for disconnections in data
            'non_disconn_ratio': 0.25
        },

        # Parameters of solar detection data preparation function

        'prep_solar_data': {
            'percentile_cap': 90,
            'sun_index': Cgbdisagg.INPUT_DIMENSION,
            'slide_len': 30,
            'instance_size': 90
        },

        # Parameters for solar panel presence

        'solar_panel_presence': {
            'present_throughout': 0,
            'installation': 1,
            'removal': 2
        },

        # Parameters of solar detection data preparation function

        'get_detection_probability': {
            'permute_sequence': [0, 3, 1, 2]
        },

        # CNN model paramaters

        'cnn_model': {
            'conv1_out_channels': 8,
            'conv1_kernel': 3,
            'conv1_stride': 1,
            'conv1_pad': 1,
            'bn1_feat': 8,
            'pool1_kernel': 3,
            'drop_prob': 0.2,
            'conv2_in_channels': 8,
            'conv2_out_channels': 16,
            'conv2_kernel': 3,
            'conv2_stride': 1,
            'conv2_pad': 1,
            'bn2_feat': 16,
            'pool2_kernel': 2,
            'fc1_in_feat': 16 * 15 * 4,
            'fc1_out_feat': 128,
            'fc2_out_feat': 8
        },

        # solar detection post-processing paramaters

        'solar_post_processing': {
            'window': 2,
            'lower_confl_prob': 0.3,
            'high_confl_prob': 0.85,
            'ns_zero_ratio_night': 0.25,
            's_zero_ratio_day': 0.5,
            's_zero_ratio_night': 0.2,
            's_edge_fluct':  50,
            's_non_zero_days': 30,
            'ns_convert_prob': 0.4,
            's_convert_prob': 0.6,
            'num_days_insta': 90,
            'normalised_zero_thresh': 0.05,
            'num_allowed_constant': 5,
            'constant_cons_perc_thresh': 0.4
        }
    }

    return so_config

