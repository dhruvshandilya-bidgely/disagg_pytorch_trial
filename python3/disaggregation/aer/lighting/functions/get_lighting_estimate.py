"""
Author - Mayank Sharan
Date - 26/11/2019
get lighting estimate uses all parameters generated to extract lighting consumption values
"""

# Import python packages

import copy
import numpy as np

# Import functions from within the project

from python3.disaggregation.aer.lighting.init_lighting_params import setpoint_list
from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile


def get_lighting_estimate(data_lighting, cap_m, cap_e, int_m, int_e, num_periods, pd_mult, config, debug, logger_pass):

    """
    Parameters:
        data_lighting       (np.ndarray)        : 21 column input data matrix
        cap_m               (float)             : Capacity for morning
        cap_e               (float)             : Capacity for evening
        int_m               (float)             : Interpolation value for morning in Wh
        int_e               (float)             : Interpolation value for evening in Wh
        num_periods         (int)               : Number of periods in 1 day
        pd_mult             (int)               : Number of period in 1 hour
        debug               (dict)              : Contains all variables needed to debug the lighting algorithm
        config              (dict)              : All parameters needed to run lighting

    Output:
        int_m               (float)             : Interpolation value for morning in Wh
        int_e               (float)             : Interpolation value for evening in Wh
        lighting            (np.ndarray)        : 2d matrix with lighting estimate
        debug               (dict)              : Contains all variables needed to debug the lighting algorithm
    """
    # Initialise data

    num_days = data_lighting.shape[0]
    morning_idx = setpoint_list(config.get('MORNING_TIME')[0] * pd_mult,
                                config.get('MORNING_TIME')[1] * pd_mult - 1) - 1

    evening_idx_1 = setpoint_list(1, config.get('MORNING_TIME')[0] * pd_mult - 1) - 1
    evening_idx_2 = setpoint_list(config.get('MORNING_TIME')[1] * pd_mult, num_periods) - 1

    m_light = copy.deepcopy(data_lighting[:, morning_idx])
    e_light = np.c_[data_lighting[:, evening_idx_1], data_lighting[:, evening_idx_2]]

    m_light = np.round(m_light, 5)
    e_light = np.round(e_light, 5)

    m_data_max = superfast_matlab_percentile(m_light, config.get('DAY_MAX_PERCENTILE'), 1)
    e_data_max = superfast_matlab_percentile(e_light, config.get('DAY_MAX_PERCENTILE'), 1)

    m_non_lighting_cap = np.round(m_data_max - cap_m, 5)
    e_non_lighting_cap = np.round(e_data_max - cap_e, 5)

    interpolation_m = np.array([])
    interpolation_e = np.array([])

    # Get Lighting Estimate for each day

    for i in range(num_days):

        # Get data for the day

        m_light_day = m_light[i, :]
        m_non_lighting_cap_day = m_non_lighting_cap[i]

        # Bring the values in the range 0 to cap

        m_light_day[m_light_day > cap_m] = m_light_day[m_light_day > cap_m] - m_non_lighting_cap_day
        m_light_day[m_light_day > cap_m] = cap_m

        # Add a new interpolation value if the day has pure lighting

        e_light_day = e_light[i, :]
        e_non_lighting_cap_day = e_non_lighting_cap[i]

        # Bring the values in the range 0 to cap

        e_light_day[e_light_day > cap_e] = e_light_day[e_light_day > cap_e] - e_non_lighting_cap_day
        e_light_day[e_light_day > cap_e] = cap_e

        # Add a new interpolation value if the day has pure lighting

        m_light[i, :] = m_light_day
        e_light[i, :] = e_light_day

    m_above_zero = np.sum(m_light > 0, axis=1)
    e_above_zero = np.sum(e_light > 0, axis=1)

    m_below_zero = np.sum(m_light < 0, axis=1)
    e_below_zero = np.sum(e_light < 0, axis=1)

    m_pure_lighting_idx = np.logical_and(m_above_zero > 0, m_below_zero == 0)
    e_pure_lighting_idx = np.logical_and(e_above_zero > 0, e_below_zero == 0)

    m_pure_lighting_days = m_light[m_pure_lighting_idx, :]
    e_pure_lighting_days = e_light[e_pure_lighting_idx, :]

    for i in range(len(m_pure_lighting_days)):
        m_light_day = m_pure_lighting_days[i, :]
        interpolation_m = np.r_[interpolation_m, np.median(m_light_day[m_light_day > 0])]

    for i in range(len(e_pure_lighting_days)):
        e_light_day = e_pure_lighting_days[i, :]
        interpolation_e = np.r_[interpolation_e, np.median(e_light_day[e_light_day > 0])]

    if int_e == -1:
        if len(interpolation_e) == 0:
            int_e = cap_e/2
        else:
            int_e = np.median(interpolation_e)

    if int_m == -1:
        if len(interpolation_m) == 0:
            int_m = cap_m/2
        else:
            int_m = np.median(interpolation_m)

    e_light[e_light < 0] = int_e
    m_light[m_light < 0] = int_m

    lighting = np.c_[e_light[:, evening_idx_1], m_light, e_light[:, np.max(evening_idx_1) + 1:]]

    # Populate debug object

    debug['lighting'] = {
        'non_lighting_cap_morning': m_non_lighting_cap,
        'non_lighting_cap_evening': e_non_lighting_cap,
        'morning_interpolation': int_m,
        'evening_interpolation': int_e,
        }

    return lighting, debug
