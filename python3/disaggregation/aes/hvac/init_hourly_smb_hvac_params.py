""" Init Hourly Hvac Params """
import numpy as np


def setpoint_list(start, stop, step=1):
    """ Float list generation """
    return np.array(list(range(start, stop + step, step)))


def get_found_sh(mu, diff_prob):
    """ Return found"""
    if diff_prob < 0.08:
        found = 0
    elif diff_prob > 0.2:
        found = 1
    else:
        if mu > 309:
            found = 1
        else:
            found = 0

    return found


def get_found_wh(diff_prob, dist, config):
    """ Return found = get_found_wh(mu,sigma,diffProb,dist,config)"""
    found = diff_prob > config['MIN_PROPORTION'] and (dist >= config['MIN_DETECTION_DIST'])
    return found


def get_found_ac(mu, diff_prob):
    """ Get Found AC """
    found = 1
    if diff_prob <= 0.1338:
        found = 0
    # end
    return found


class Struct:
    """MATLAB structure"""

    def __init__(self, **entries):
        self.__dict__.update(entries)


def hvac_static_params():
    """
    Function returns the static parameters used in hvac only
    Returns:
        static_param (dict) : Dictionary containing all the static parameters
    """

    static_param = {'inequality_handler': 0.001,
                    'length_of_cluster_info': 8,
                    'length_of_cluster_limits': 2,
                    'ao_degree_day_threshold': 200,
                    'min_consumption_for_setpoint': 10,
                    'fallback_hour_aggregate': 12,
                    'fallback_cdd_quantile': 0.4,
                    'dbscan_init_epsilon': 0.8,
                    'dbscan_epsilon_decrement': 0.1,
                    'dbscan_lowest_epsilon': 0.2,
                    'relax_cluster_density': 0.1,
                    'dbscan_min_points_in_eps': 3,
                    'dbscan_lowest_r2_for_cluster': 0.44,
                    'include_extra_points_upper': 80,
                    'include_extra_points_lower': 20,
                    'arm_65': 1.65,
                    'arm_3_std': 3,
                    'hour_aggregate_level': 8,
                    'hour_aggregate_level_default': 24,
                    'fpl_id': 10017,
                    'pivot_F': 65,
                    'deg_day_tolerance_ac': 5,
                    'deg_day_tolerance_sh': 5,
                    'deg_day_tolerance': 5,
                    'day_cdd_low_limit': 200,
                    'day_hdd_low_limit': 350,
                    'kilo': 1000,
                    'hsm_large_num': 999999999.9,

                    'gaussian_related': {'sigma_arm': 1.65,
                                         'points_to_fill': 10000,
                                         'tolerance': 1e-15, },

                    'cold_pilots': [20010, 20012],
                    'cold_setpoint_low_lim': 70,
                    'cold_setpoint_high_lim': 80,

                    'quantized_low_lim': 3,
                    'quantized_r2_min_months': 2,
                    'quantized_high_lim': 300,
                    'default_r_square_quantized': 0.05,
                    'residual_quantization_val': 250,
                    'len_residual_round': 2,
                    'default_residual_rsquare': 1.0,
                    'default_residual_stability': 0.05,

                    'month_low_kwh_ac': 40,
                    'month_low_kwh_sh': 50,
                    'std_concerned_residual': 7,
                    'under_estimate_arm': 0.5,
                    'over_estimate_arm': 1.4,
                    'post_process_cdd_ref': 65,
                    'post_process_hdd_ref': 50,
                    'month_reduction_lim': 61,
                    'bs_gap_between_residue': 10,

                    'ao': {'data_per_hour_30': 2,
                           'data_per_hour_15': 4,
                           'hour_bin_size': 20,
                           'min_hour_baseload': 5,
                           'zero_threshold': 0.3,
                           'max_days_to_remove': 100,
                           'failsafe_quantile': 0.03,
                           'day_zero_threshold': 0.8,
                           'min_len_valid_points': 100,
                           'suppress_fp_hvac': 20000, },
                    'smb': {'input_df_bl_col': 2,
                            'data_per_hour_15': 4,
                            'hour_bin_size': 20,
                            'min_hour_baseload': 5,
                            'zero_threshold': 0.3,
                            'max_days_to_remove': 100,
                            'failsafe_quantile': 0.03,
                            'day_zero_threshold': 0.8,
                            'suppress_fp_hvac': 20000, },
                    'ineff': {'max_abrupt_hour_len': 10,
                              'low_quant': 0.05,
                              'high_quant': 0.95,
                              'low_outlier': 0,
                              'high_outlier': 0,
                              'total_outlier': 0,
                              'minimum_hvac_days': 10,
                              'lower_limit_quantile': 0.20,
                              'hvac_outlier_limit_list': [1.5, 1.875, 1.875],
                              'low_limit_axis': 0.25,
                              'high_limit_axis': 0.75,
                              'hvac_cons_low_lim': 100,
                              'minimum_ao_hvac_hours': 20,
                              'count_potential_lim': 10,
                              'init_prob': -1.,
                              'init_pred': -1,
                              'degrade_high_pct': 0.90,
                              'degrade_app_weight': 0.3,
                              'degrade_low_cap_len': 10,
                              'degrade_up_cap_len': 30,
                              'hvac_change_wt': 0.5,
                              'low_duty_cycle_lim': 0.3,
                              'low_len_modify': 15,
                              'low_len_modify_factor': 0.9,
                              'high_len_modify': 20,
                              'high_len_modify_factor': 0.95,
                              'hvac_change_low_prob_limit': 0.70,
                              'count_allowed_30': 70,
                              'count_allowed_15': 120,
                              'count_allowed_other': 40,
                              'support_threshold': 10,
                              'minimum_dc_limit': 0.05,
                              'run_length_threshold': 2,
                              'upper_divergence_threshold': 8,
                              'lower_divergence_threshold': -3,
                              'slide_size': 7,
                              'window_size': 50,
                              'start_office_hour': 7,
                              'end_office_hour': 17,
                              'max_allowed_short_cycling_streaks': 4,
                              'abrupt_tou_step': 15,
                              'abrupt_tou_window': 60,
                              'local_outlier_lim': 4,
                              'outlier_det_med_pct': 35,
                              'norm_temp_epsilon': 1e-10,
                              'outlier_prob_pivot': 0.3,
                              'tou_outlier_low_prob': 0.2,
                              'tou_outlier_high_prob': 0.1,
                              'final_outlier_lim': 0.1,
                              'large_neg_value': -1000,
                              'hsm_high_cons_quant': 0.90,
                              'hsm_allowed_buffer_months': 4,
                              'outlier_window_size': 28,
                              'season_min_hvac_days': 45,
                              'season_min_hvac_days_enough_summer': 70,
                              'min_count_potential': 10,
                              'enable_plotting': False},
                    'path': {'epoch_hvac_dir': '/var/log/bidgely/pyamidisagg/pydisagg/epoch_od_ao_hvac',
                             'hvac_plots': '/var/log/bidgely/pyamidisagg/pydisagg/hvac_plots',
                             'open_close_data': '/var/log/bidgely/pyamidisagg/pydisagg/open_close_day'},
                    'local_path': {'epoch_hvac_dir': '../epoch_od_ao_hvac',
                                   'hvac_plots': '../hvac_plots',
                                   'open_close_data': '../open_close_day'},
                    'temp_interpolation': {'interpolation_window_days': 20,
                                           'required_data_days': 10,
                                           'max_hours_to_interpolate': 15},
                    'remove_operational_load': {'max_zero_data_days': 0.85,
                                                'min_good_days': 5,
                                                'min_epoch_cons': 5,
                                                'work_hour_prop_divisor': 6}
                    }

    return static_param


def get_sh_upper_limit(disagg_input_object, logger_hvac, logger_flag):
    """
        Function to get the upper limit of SH setpoint searchspace

        Parameters:
            disagg_input_object (dict)              :   Dictionary containing input related key information
            logger_hvac         (logging object)    :   Writes logs during code flow

        Returns :
            sh_high             (int)               :   The high limit of SH setpoint search
    """

    sh_high = 63
    ac_low = 67

    sh_degree_epoch = (disagg_input_object['input_data'][:, 7] - sh_high) * (-1)
    ac_degree_epoch = disagg_input_object['input_data'][:, 7] - ac_low

    ac_dominance = np.sum(ac_degree_epoch > 0) / len(ac_degree_epoch)
    sh_dominance = np.sum(sh_degree_epoch > 0) / len(sh_degree_epoch)

    if (ac_dominance > 0.75) & (sh_dominance < 0.20):
        sh_high = 60
        if logger_flag == True:
            logger_hvac.info('HVAC3P1 : Constrained sh setpoint search space to {}F |'.format(sh_high))

    return sh_high


def get_ac_lower_limit(disagg_input_object, logger_hvac, logger_flag):
    """
        Function to get the upper limit of AC setpoint searchspace

        Parameters:
            disagg_input_object (dict)              :   Dictionary containing input related key information
            logger_hvac         (logging object)    :   Writes logs during code flow
            logger_flag         (bool)              :   Boolean to identify if ac setpoint limit should be restrained
        Returns :
            ac_low              (int)               :   The high limit of AC setpoint search
    """

    sh_high = 63
    ac_low = 67

    sh_degree_epoch = (disagg_input_object['input_data'][:, 7] - sh_high) * (-1)
    ac_degree_epoch = disagg_input_object['input_data'][:, 7] - ac_low

    ac_dominance = np.sum(ac_degree_epoch > 0) / len(ac_degree_epoch)
    sh_dominance = np.sum(sh_degree_epoch > 0) / len(sh_degree_epoch)

    if (sh_dominance > 0.75) & (ac_dominance < 0.20):
        ac_low = 70
        if logger_flag == True:
            logger_hvac.info('HVAC3P1 : Constrained ac setpoint search space to {}F |'.format(ac_low))

    return ac_low


def init_hvac_params(sampling_rate, disagg_input_object, logger_hvac, logger_flag=False):
    """
    Initialize general HVAC Parameters

    Parameters:
        sampling_rate (int)  :  The sampling rate of user
    Returns :
        hvac_config (dict)  :  Dictionary containing all HVAC specific config parameters (AC and SH)
    """

    multiplier = sampling_rate / 3600.0

    sh_high = get_sh_upper_limit(disagg_input_object, logger_hvac, logger_flag)
    ac_low = get_ac_lower_limit(disagg_input_object, logger_hvac, logger_flag)

    hvac_config = {

        'preprocess': {
            'NET_MAX_THRESHOLD': 30000,
        },
        'detection': {
            'MIN_POINTS': 400. / multiplier,
            'NUM_BINS': 30,
            'MID_TEMPERATURE_RANGE': [55, 70],
            'MID_TEMPERATURE_QUANTILE': [0.3, 0.7],
            'NUM_BINS_TO_REMOVE': 2,
            'AC': {
                'SETPOINTS': setpoint_list(100, ac_low, -5),
                'MIN_AMPLITUDE': 300 * multiplier,
                'MIN_DETECTION_DIST': 0.25,
                'MIN_DETECTION_STD': 1.6,
                'MIN_PROPORTION': 0.02,
                'getFound': get_found_ac,
            },
            'SH': {
                'SETPOINTS': setpoint_list(30, sh_high, 5),
                'MIN_AMPLITUDE': 200. * multiplier,
                'MIN_DETECTION_DIST': 0.05,
                'MIN_DETECTION_STD': 1.6,
                'MIN_PROPORTION': 0.02,
                'getFound': get_found_sh,
            },
            'WH': {
                'MIN_AMPLITUDE': 100. * multiplier,
                'MAX_AMPLITUDE': 700. * multiplier,
                'MIN_DETECTION_DIST': 0.3,
                'MIN_DETECTION_STD': 2,
                'MIN_PROPORTION': 0.1,
                'getFound': get_found_wh
            },
        },
        'adjustment': {
            'AC': {
                'MIN_NUM_STD': 0.15,
                'MIN_AMPLITUDE': 100 * multiplier
            },
            'SH': {
                'MIN_NUM_STD': 0.15,
                'MIN_AMPLITUDE': 100 * multiplier
            },
            'HVAC': {
                'MIN_NUM_STD': 0.15,
                'MIN_AMPLITUDE': 100 * multiplier
            }
        },
        'setpoint': {
            'AC': {
                'IS_AC': True,
                'SETPOINTS': setpoint_list(80, ac_low, -1),
                'MIN_AMPLITUDE': 300 * multiplier,
                'ARM_OF_STANDARD_DEVIATION': 0.5,
                'MAX_DAYS_TO_REMOVE': 10,
                'MIN_NUM_STD': 1.5,
                'MIN_DUTY_CYCLE': 0.5,
                'CAP_NUM_STD': 2,
                'MAX_NUM_STD': np.inf,
                'RSQ_THRESHOLD': 0.08,
                'PVAL_THRESHOLD': 0.08
            },
            'SH': {
                'IS_AC': False,
                'SETPOINTS': setpoint_list(45, sh_high, 1),
                'MIN_AMPLITUDE': 100 * multiplier,
                'ARM_OF_STANDARD_DEVIATION': 0.5,
                'MAX_DAYS_TO_REMOVE': 10,
                'MIN_NUM_STD': 1.5,
                'MIN_DUTY_CYCLE': 0.3,
                'CAP_NUM_STD': 2,
                'MAX_NUM_STD': 3,
                'RSQ_THRESHOLD': 0.08,
                'PVAL_THRESHOLD': 0.08
            },
        },
        'estimation': {
            'AC': {
                'IS_AC': True,
                'MIN_AMPLITUDE': 300 * multiplier,
                'MIN_NUM_STD': 1.5,
                'MIN_DUTY_CYCLE': 0.3,
                'CAP_NUM_STD': 2,
                'MAX_NUM_STD': np.inf,
                'SETPOINT_DIFF': np.nan,
                'REGULARIZATION': 2,
                'MIN_HRS_PER_DAY': 1. / multiplier,
                'MIN_HRS_PER_MONTH': 20. / multiplier,
                'SETPOINT_BUFFER': 10,
                'CONSEC_RANGE': 5. / multiplier,
                'CONSEC_POINTS': 3. / multiplier
            },
            'SH': {
                'IS_AC': False,
                'MIN_AMPLITUDE': 200. * multiplier,
                'MIN_NUM_STD': 1.5,
                'MIN_DUTY_CYCLE': 0.3,
                'CAP_NUM_STD': 2,
                'MAX_NUM_STD': np.inf,
                'SETPOINT_DIFF': np.nan,
                'REGULARIZATION': 2,
                'MIN_HRS_PER_DAY': 3. / multiplier,
                'MIN_HRS_PER_MONTH': 20. / multiplier,
                'SETPOINT_BUFFER': 10,
                'CONSEC_RANGE': 5. / multiplier,
                'CONSEC_POINTS': 3. / multiplier
            }
        },
        'postprocess': {
            'AC': {
                'MIN_DAILY_KWH': 1.5,
                'MIN_DAILY_PERCENT': 0.05,
                'MAX_DAILY_PERCENT': 0.95
            },
            'SH': {
                'MIN_DAILY_KWH': 0.4,
                'MIN_DAILY_PERCENT': 0.02,
                'MAX_DAILY_PERCENT': 0.95
            },
        },
    }
    return hvac_config
