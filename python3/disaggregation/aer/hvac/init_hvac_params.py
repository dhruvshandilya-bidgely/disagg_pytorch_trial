"""
Author - Abhinav
Date - 10/10/2018
Initiate hvac parameters
"""

# Import python packages

import numpy as np


def get_found_sh(mu, sigma, diff_prob, dist, config):
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


def init_hvac_params(sampling_rate):

    """
    Function to initialize hvac algorithm related parameters

    Parameters:
        sampling_rate (int)                      : Input data sampling rate (e.g. 900 for 15 min sampling)
    Returns:
        hvac_params (dict)                       : Dictionary containing hvac algo related initialized parameters
    """

    # hvac_params.detection.AC.SETPOINT : upper limit excluded
    # hvac_params.detection.SH.SETPOINT : upper limit excluded

    # hvac_params.setpoint.AC.SETPOINT : upper limit excluded
    # hvac_params.setpoint.SH.SETPOINT : upper limit excluded

    # hvac_params.preprocess.NET_MAX_THRESHOLD : Disallow high consumption from skewing correlation

    multiplier = sampling_rate / 3600.0
    hvac_params = {
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
                'SETPOINTS': np.arange(100, 60, -5),
                'MIN_AMPLITUDE': 100 * multiplier,
                'MIN_DETECTION_DIST': 0.25,
                'MIN_DETECTION_STD': 1.6,
                'MIN_PROPORTION': 0.02,
                'getFound': get_found_ac,
            },
            'SH': {
                'SETPOINTS': np.arange(30, 65, 5),
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
                'SETPOINTS': np.arange(80, 64, -1),
                'MIN_AMPLITUDE': 300 * multiplier,
                'ARM_OF_STANDARD_DEVIATION': 0.5,
                'MAX_DAYS_TO_REMOVE': 10,
                'MIN_NUM_STD': 1.5,
                'MIN_DUTY_CYCLE': 0.5,
                'CAP_NUM_STD': 2,
                'MAX_NUM_STD': 3,
                'RSQ_THRESHOLD': 0.1,
                'PVAL_THRESHOLD': 0.1
            },
            'SH': {
                'IS_AC': False,
                'SETPOINTS': np.arange(45, 66, 1),
                'MIN_AMPLITUDE': 100 * multiplier,
                'ARM_OF_STANDARD_DEVIATION': 0.5,
                'MAX_DAYS_TO_REMOVE': 10,
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
    return hvac_params
