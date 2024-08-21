""" Init Hourly Hvac Params """
import numpy as np
from python3.config.Cgbdisagg import Cgbdisagg


def setpoint_list(start, stop, step=1):
    """ Float list generation """
    return np.array(list(range(start, stop + step, step)))


def get_found_sh_jp(mu, sigma, diff_prob, dist, config, debug_detection):
    """ Return found
        This model is currently only used for Japan
    """
    found = 1
    tstat_ma_max = debug_detection.get('hdd').get('debug_tstat').get('t_stat_ma_max')
    corr = debug_detection.get('hdd').get('SH_log_reg_features').get('SH').get('corr')
    max_minus_min_hddcdd = debug_detection.get('hdd').get('SH_log_reg_features').get('SH').get('max_minus_min_hddcdd')
    corr_month_selection = debug_detection.get('hdd').get('SH_log_reg_features').get('SH').get('corr_month_selection')
    max_min_min_ms = debug_detection.get('hdd').get('SH_log_reg_features').get('SH').get('max_minus_min_ms')

    max_corr = max(corr, corr_month_selection)
    max_minus_min = max(max_minus_min_hddcdd, max_min_min_ms)

    # run logistic regression
    feats = np.array([diff_prob, max_corr, max_minus_min, tstat_ma_max])
    means = np.array(config['LOG_REG_PARAMS']['MEAN'])
    std = np.array(config['LOG_REG_PARAMS']['STD'])
    coeffs = np.array(config['LOG_REG_PARAMS']['COEFF'])
    scaled_feats = (feats - means) / std
    z = config['LOG_REG_PARAMS']['INTERCEPT'] + np.dot(scaled_feats, coeffs)
    prob_sh = 1 / (1 + np.exp(- z))

    if prob_sh < 0.5:
        found = 0

    return found


def get_found_sh(mu, sigma, diff_prob, dist, config):
    """ Return found """
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


def get_found_wh(mu, sigma, diff_prob, dist, config):
    """ Return found = get_found_wh(mu,sigma,diffProb,dist,config)"""
    found = diff_prob > config['MIN_PROPORTION'] and (dist >= config['MIN_DETECTION_DIST'])
    return found


def get_found_ac(mu, sigma, diff_prob, dist, config):
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


def parse_opt(p, v, **entries):
    """ Parse dict """
    setflag = Struct()
    for i in range(len(p)):
        locals()[p[i]] = v[i]
        setflag.__dict__[p[i]] = 0
    if entries is not None:
        for k, v in entries.items():
            setflag.__dict__[k] = 1
            locals()[k] = v

    ret = ()
    for i in range(len(p)):
        ret = ret + (locals()[p[i]],)
    ret = ret + (setflag, None)
    return ret


def init_hvac_params_jp(sampling_rate):

    """ Init Hourly Hvac Params """

    multiplier = sampling_rate / Cgbdisagg.SEC_IN_HOUR

    config = {
        'preprocess': {
            'NET_MAX_THRESHOLD': 30000,
        },
        'detection': {
            'MIN_POINTS': 750. / multiplier,
            'NUM_BINS': 25,
            'MID_TEMPERATURE_RANGE': [63, 70],
            'MID_TEMPERATURE_QUANTILE': [0.2, 0.8],
            'NUM_BINS_TO_REMOVE': 0,
            'AC': {
                'SETPOINTS': setpoint_list(100, 65, -5),
                'MIN_AMPLITUDE': 100 * multiplier,
                'MIN_DETECTION_DIST': 0.25,
                'MIN_DETECTION_STD': 1.6,
                'MIN_PROPORTION': 0.02,
                'SETPOINT_FOR_DECISION_TREE': 70,
                'getFound': get_found_ac,
            },
            'SH': {
                'SETPOINTS': setpoint_list(30, 55, 5),
                'MIN_AMPLITUDE': 100. * multiplier,
                'MIN_DETECTION_DIST': 0.05,
                'MIN_DETECTION_STD': 1.6,
                'MIN_PROPORTION': 0.02,
                'SETPOINT_FOR_DECISION_TREE': 55,
                'getFound': get_found_sh_jp,
                'MIN_TEMP_BOUND_LOWER': 0,
                'MIN_TEMP_BOUND_UPPER': 50,
                'MIN_TEMP_BOUND_STEP': 2,
                'MIN_DATA_POINTS_CDD': 30,
                'MAX_TEMP_BOUND_LOWER': 70,
                'MAX_TEMP_BOUND_UPPER': 120,
                'MAX_TEMP_BOUND_STEP': -2,
                'MAX_DATA_POINTS_CDD': 30,
                'LOWER_TEMP_FILTER': 55,
                'UPPER_TEMP_FILTER': 75,
                'TTEST_THRESH': 100,
                'FILTER_DATA_MID_VAL_LOWER_THRESH': 10,
                'FILTER_DATA_MIN_MID_TEMP_UPPER_THRESH': 40,
                'FILTER_DATA_RECURSIVE_TEMP_INC': 3,
                'MONTH_VALID_NUM_DAYS_THRESH': 30,
                'MIN_DATA_POINTS_THRESH': 30,
                'LOG_REG_PARAMS': {
                    'FEATURE': ['diffProb-maxCorr-maxMinusMin-tstatMA'],
                    'STD': [0.2771, 0.4356, 1217951.09270, 7.6970],
                    'MEAN': [0.3326, 0.5934, 672254.4299, 6.9801],
                    'COEFF': [2.5128, 0.8980, 4.4492, 1.6388],
                    'INTERCEPT': 3.9571
                    }
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
            }
        },
        'setpoint': {
            'AC': {
                'IS_AC': True,
                'SETPOINTS': setpoint_list(85, 70, -1),
                'MIN_AMPLITUDE': 150 * multiplier,
                'ARM_OF_STANDARD_DEVIATION': 0.5,
                'MAX_DAYS_TO_REMOVE': 10,
                'MIN_NUM_STD': 2,
                'MIN_DUTY_CYCLE': 0.3,
                'CAP_NUM_STD': 0.5,
                'MAX_NUM_STD': np.inf,
                'RSQ_THRESHOLD': 0.1,
                'PVAL_THRESHOLD': 0.1
            },
            'SH': {
                'IS_AC': False,
                'SETPOINTS': setpoint_list(45, 65, 1),
                'MIN_AMPLITUDE': 250 * multiplier,
                'ARM_OF_STANDARD_DEVIATION': 0.5,
                'MAX_DAYS_TO_REMOVE': 10,
                'MIN_NUM_STD': 1.5,
                'MIN_DUTY_CYCLE': 0.5,
                'CAP_NUM_STD': 1.5,
                'MAX_NUM_STD': 3,
                'RSQ_THRESHOLD': 0.1,
                'PVAL_THRESHOLD': 0.1
            },
        },
        'estimation': {
            'AC': {
                'IS_AC': True,
                'MIN_AMPLITUDE': 150 * multiplier,
                'MIN_NUM_STD': 2,
                'MIN_DUTY_CYCLE': 0.3,
                'CAP_NUM_STD': 2.5,
                'MAX_NUM_STD': np.inf,
                'SETPOINT_DIFF': np.nan,
                'REGULARIZATION': 1.5,
                'MIN_HRS_PER_DAY': 5. / multiplier,
                'MIN_HRS_PER_MONTH': 10. / multiplier,
                'SETPOINT_BUFFER': 10,
                'CONSEC_RANGE': 3. / multiplier,
                'CONSEC_POINTS': 1. / multiplier
            },
            'SH': {
                'IS_AC': False,
                'MIN_AMPLITUDE': 200. * multiplier,
                'MIN_NUM_STD': 1.5,
                'MIN_DUTY_CYCLE': 0.3,
                'CAP_NUM_STD': 0.2,
                'MAX_NUM_STD': np.inf,
                'SETPOINT_DIFF': np.nan,
                'REGULARIZATION': 1.5,
                'MIN_HRS_PER_DAY': 5. / multiplier,
                'MIN_HRS_PER_MONTH': 20. / multiplier,
                'SETPOINT_BUFFER': 10,
                'CONSEC_RANGE': 5. / multiplier,
                'CONSEC_POINTS': 2. / multiplier
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
    return config


def init_hvac_params(sampling_rate):
    """ Init Hourly Hvac Params """

    multiplier = sampling_rate / Cgbdisagg.SEC_IN_HOUR

    config = {
        'preprocess': {
            'NET_MAX_THRESHOLD': 30000,
        },
        'detection': {
            'MIN_POINTS': 500. / multiplier,
            'NUM_BINS': 30,
            'MID_TEMPERATURE_RANGE': [55, 70],
            'MID_TEMPERATURE_QUANTILE': [0.3, 0.7],
            'NUM_BINS_TO_REMOVE': 2,
            'AC': {
                'SETPOINTS': setpoint_list(100, 65, -5),
                'MIN_AMPLITUDE': 100 * multiplier,
                'MIN_DETECTION_DIST': 0.25,
                'MIN_DETECTION_STD': 1.6,
                'MIN_PROPORTION': 0.02,
                'getFound': get_found_ac,
            },
            'SH': {
                'SETPOINTS': setpoint_list(30, 60, 5),
                'MIN_AMPLITUDE': 100. * multiplier,
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
            }
        },
        'setpoint': {
            'AC': {
                'IS_AC': True,
                'SETPOINTS': setpoint_list(80, 65, -1),
                'MIN_AMPLITUDE': 300 * multiplier,
                'MIN_NUM_STD': 1.5,
                'MIN_DUTY_CYCLE': 0.5,
                'CAP_NUM_STD': 2,
                'MAX_NUM_STD': np.inf,
                'RSQ_THRESHOLD': 0.1,
                'PVAL_THRESHOLD': 0.1
            },
            'SH': {
                'IS_AC': False,
                'SETPOINTS': setpoint_list(45, 65, 1),
                'MIN_AMPLITUDE': 100 * multiplier,
                'MIN_NUM_STD': 1.5,
                'MIN_DUTY_CYCLE': 0.3,
                'CAP_NUM_STD': 2,
                'MAX_NUM_STD': 3,
                'RSQ_THRESHOLD': 0.1,
                'PVAL_THRESHOLD': 0.1
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
    return config
