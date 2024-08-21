"""
Author - Mayank Sharan
Date - 26/11/2019
get lighting params extracts parameters needed to generate lighting estimate from the data with japan specific changes
"""

# Import python packages

import copy
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.lighting.init_lighting_params import setpoint_list

from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile


def cap_bound(cap_m, cap_e, pd_mult, config, debug):

    """Bound the capacity"""

    # Lower limit on morning capacity

    if cap_m < config.get('CAPACITY_MIN_BOUND') / pd_mult:
        cap_m = config.get('CAPACITY_MIN_BOUND') / pd_mult
        debug['params']['cap_catch'] = '| Low Morning cap < 40 %s' % debug['params']['cap_catch']

    # Upper limit on morning capacity

    if cap_m > config.get('CAPACITY_MAX_BOUND') / pd_mult:
        cap_m = config.get('CAPACITY_MAX_BOUND') / pd_mult
        debug['params']['cap_catch'] = '| High Morning cap > 1500 %s' % debug['params']['cap_catch']

    # Lower limit on evening capacity

    if cap_e < config.get('CAPACITY_MIN_BOUND') / pd_mult:
        cap_e = config.get('CAPACITY_MIN_BOUND') / pd_mult
        debug['params']['cap_catch'] = '| Low Evening cap < 40 %s' % debug['params']['cap_catch']

    # Upper limit on evening capacity

    if cap_e > config.get('CAPACITY_MAX_BOUND') / pd_mult:
        cap_e = config.get('CAPACITY_MAX_BOUND') / pd_mult
        debug['params']['cap_catch'] = '| High Evening cap > 1500 %s' % debug['params']['cap_catch']

    # Bound capacities specifically for TEPCO

    temp_cap_m = copy.deepcopy(cap_m)
    cap_delta_allowed = 150 / pd_mult

    if cap_m == config.get('CAPACITY_MIN_BOUND') / pd_mult:
        temp_cap_m = 0

    if cap_e - temp_cap_m > cap_delta_allowed:
        cap_e = max(cap_e - (75 / pd_mult), temp_cap_m + cap_delta_allowed)

        if cap_e > temp_cap_m + cap_delta_allowed:
            cap_e = cap_e - (0.25 * (cap_e - (temp_cap_m + cap_delta_allowed)))

    return cap_m, cap_e, debug


def fix_zero_m_capacity(cap_m, temp_perc_season, config, data_params, morning_idx, temp_cap_by_time):
    """Utility to fix zero capacity morning if we get it"""

    while cap_m == 0 and temp_perc_season > config.get('ZERO_CAPACITY_FIX_STEP') + 1:

        temp_perc_season = temp_perc_season - config.get('ZERO_CAPACITY_FIX_STEP')
        if data_params.shape[0] > 1:
            temp_cap_by_time = superfast_matlab_percentile(data_params, 100 - temp_perc_season)
            cap_m = max(temp_cap_by_time[morning_idx])
        elif data_params.shape[0] == 1:
            temp_cap_by_time = data_params
            cap_m = max(temp_cap_by_time[morning_idx])
        else:
            break

    return cap_m, temp_cap_by_time


def fix_zero_e_capacity(cap_e, temp_perc_season, config, data_params, evening_idx_1, evening_idx_2, temp_cap_by_time):
    """Utility to fix zero capacity evening if we get it"""

    while cap_e == 0 and temp_perc_season > config.get('ZERO_CAPACITY_FIX_STEP') + 1:

        temp_perc_season = temp_perc_season - config.get('ZERO_CAPACITY_FIX_STEP')
        if data_params.shape[0] > 1:
            temp_cap_by_time = superfast_matlab_percentile(data_params, 100 - temp_perc_season)
            cap_e = max(np.r_[temp_cap_by_time[evening_idx_1], temp_cap_by_time[evening_idx_2]])
        elif data_params.shape[0] == 1:
            temp_cap_by_time = data_params
            cap_e = max(np.r_[temp_cap_by_time[evening_idx_1], temp_cap_by_time[evening_idx_2]])
        else:
            break

    return cap_e, temp_cap_by_time


def get_lighting_params_jp(data_season_rem, smoothing_noise_bound, perc_season, num_periods, pd_mult, config, debug,
                           logger_pass):

    """ Return cap_m, cap_e, lighting_band, debug

    GETLIGHTINGPARAMS detects hours in the lighting band and calculates the
    capacity of lighting for morning and evening

    Parameters:
        data_season_rem (double matrix) : 2D matrix with seasonal data removed
        smoothing_noise_bound (double)  : Upper bound on smoothing noise
        perc_season (double)            : Percentage of seasonal non zero days
        debug (dict)                  : All variable values needed to debug
        config (dict)                 : Contains all parameters needed

    Returns:
        cap_m (double)                      : Morning lighting capacity in Wh
        cap_e (double)                      : Evening lighting capacity in Wh
        lighting_band (logical vector)      : For each time slice if lighting or not
        debug (dict)                      : All variable values needed to debug
    """

    # Initialise data
    data_params = copy.deepcopy(data_season_rem)
    daily_cons = np.nansum(data_params, 1)
    data_params = data_params[daily_cons > config.get('ZERO_DAY_CONSUMPTION'), :]

    num_non_zero_days = data_params.shape[0]

    if num_non_zero_days == 0:
        num_non_zero_days = -1
    elif config.get('MIN_NOISE_RATIO') * smoothing_noise_bound < np.max(data_params):
        data_params[data_params < smoothing_noise_bound] = 0

    # Detect Lighting Band

    debug['params'] = {
        'band_catch': '|',
        'capacity_by_time': None,
        'perc_by_time': None,
        'time_threshold': None,
        'lighting_band': None,
        'morning_capacity': None,
        'evening_capacity': None,
    }

    is_lighting = copy.deepcopy(data_params)
    is_lighting[is_lighting > 0] = 1

    perc_by_time = np.nansum(is_lighting, axis=0) * 100 / num_non_zero_days

    if perc_season < config.get('LOW_SEASON_THRESHOLD'):
        debug['params']['band_catch'] = '| Perc season < 10 starting at %f %s' \
                                        % (perc_season, debug['params']['band_catch'])
        time_selection_thresh = perc_season
    else:
        time_selection_thresh = config.get('TIME_SELECTION_THRESHOLD')

    lighting_band = perc_by_time > time_selection_thresh

    # Restrict Lighting Band to 12 hours
    temp_thresh = time_selection_thresh
    fractional_band_cap = config.get('LIGHTING_BAND_CAP') / Cgbdisagg.HRS_IN_DAY

    while (np.sum(lighting_band) / num_periods > fractional_band_cap and
           temp_thresh <= config.get('TIME_SELECTION_THRESHOLD_CAP')):

        temp_thresh = temp_thresh + config.get('TIME_SELECTION_THRESHOLD_STEP')
        lighting_band = perc_by_time > temp_thresh

    if temp_thresh > time_selection_thresh:
        debug['params']['band_catch'] = '| Band threshold increased to %d %s'\
                                        % (temp_thresh, debug['params']['band_catch'])
        time_selection_thresh = temp_thresh

    non_typical_lighting_1 = setpoint_list(config.get('NON_TYPICAL_LIGHTING_1')[0] * pd_mult - 1,
                                           config.get('NON_TYPICAL_LIGHTING_1')[1] * pd_mult - 1)
    non_typical_lighting_2 = setpoint_list(config.get('NON_TYPICAL_LIGHTING_2')[0] * pd_mult - 1,
                                           config.get('NON_TYPICAL_LIGHTING_2')[1] * pd_mult - 1)

    if np.sum(lighting_band) / num_periods > fractional_band_cap:

        lighting_band[non_typical_lighting_1] = 0
        debug['params']['band_catch'] = '| 1-5 AM rejected %s' % debug['params']['band_catch']

    if np.sum(lighting_band) / num_periods > fractional_band_cap:

        lighting_band[non_typical_lighting_2] = 0
        debug['params']['band_catch'] = '| 12-4 PM rejected %s' % debug['params']['band_catch']

    # If the band is still greater than 12 hours truncate the band to fit
    # into 12 hours
    if np.sum(lighting_band) / num_periods > fractional_band_cap:
        lighting_idx = np.nonzero((lighting_band == 1).astype(int))[0]
        lighting_band[lighting_idx[int(np.floor(num_periods * fractional_band_cap)):]] = 0
        debug['params']['band_catch'] = 'Lighting Band truncated %s' % debug['params']['band_catch']

    # Calculate capacities
    data_params[:, ~lighting_band] = 0
    morning_idx = setpoint_list(config.get('MORNING_TIME')[0] * pd_mult,
                                config.get('MORNING_TIME')[1] * pd_mult - 1) - 1

    evening_idx_1 = setpoint_list(1, config.get('MORNING_TIME')[0] * pd_mult - 1) - 1
    evening_idx_2 = setpoint_list(config.get('MORNING_TIME')[1] * pd_mult, num_periods) - 1

    cap_by_time = superfast_matlab_percentile(data_params, 100 - perc_season, 0)

    cap_e = superfast_matlab_percentile(np.r_[cap_by_time[evening_idx_1], cap_by_time[evening_idx_2]],
                                        config.get('CAPACITY_PERCENTILE'))
    cap_m = superfast_matlab_percentile(cap_by_time[morning_idx], config.get('CAPACITY_PERCENTILE'))

    # Correct for minor fault in season percentage

    debug['params']['cap_catch'] = '|'

    if perc_season < 99 - config.get('CAPACITY_CORRECTION_STEP'):

        # Compute alternate capacity values

        cap_by_time_2 = superfast_matlab_percentile(data_params, 100 - perc_season -
                                                    config.get('CAPACITY_CORRECTION_STEP'), 0)
        cap_e_2 = superfast_matlab_percentile(np.r_[cap_by_time_2[evening_idx_1], cap_by_time_2[evening_idx_2]],
                                              config.get('CAPACITY_PERCENTILE'))
        cap_m_2 = superfast_matlab_percentile(cap_by_time_2[morning_idx], config.get('CAPACITY_PERCENTILE'))

        # Compare with threshold difference and update values accordingly for evening cap

        if cap_e - cap_e_2 > config.get('CAPACITY_DIFF_THRESHOLD') / pd_mult:
            debug['params']['cap_catch'] = '| Faulty season evening cap fixed %s' % debug['params']['cap_catch']
            cap_e = cap_e_2
            cap_by_time[evening_idx_1] = cap_by_time_2[evening_idx_1]
            cap_by_time[evening_idx_2] = cap_by_time_2[evening_idx_2]

        # Compare with threshold difference and update values accordingly for morning cap

        if cap_m - cap_m_2 > config.get('CAPACITY_DIFF_THRESHOLD') / pd_mult:
            debug['params']['cap_catch'] = '| Faulty season morning cap fixed %s' % debug['params']['cap_catch']
            cap_m = cap_m_2
            cap_by_time[morning_idx] = cap_by_time_2[morning_idx]

    # Correct capacity if zero

    temp_perc_season = perc_season
    temp_cap_by_time = cap_by_time

    cap_m, temp_cap_by_time = fix_zero_m_capacity(cap_m, temp_perc_season, config, data_params,
                                                  morning_idx, temp_cap_by_time)

    cap_by_time[morning_idx] = temp_cap_by_time[morning_idx]

    if temp_perc_season < perc_season:
        debug['params']['cap_catch'] = '| Morning cap 0 season perc decreased %s' % debug['params']['cap_catch']

    temp_perc_season = perc_season
    temp_cap_by_time = cap_by_time

    cap_e, temp_cap_by_time = fix_zero_e_capacity(cap_e, temp_perc_season, config, data_params, evening_idx_1,
                                                  evening_idx_2, temp_cap_by_time)

    cap_by_time[evening_idx_1] = temp_cap_by_time[evening_idx_1]
    cap_by_time[evening_idx_2] = temp_cap_by_time[evening_idx_2]

    if temp_perc_season < perc_season:
        debug['params']['cap_catch'] = '| Evening cap 0 season perc decreased %s' % debug['params']['cap_catch']

    # Limit the values of the capacities within bounds
    cap_m, cap_e, debug = cap_bound(cap_m, cap_e, pd_mult, config, debug)

    # Populate debug object
    debug['params']['capacity_by_time'] = cap_by_time
    debug['params']['perc_by_time'] = perc_by_time
    debug['params']['time_threshold'] = time_selection_thresh
    debug['params']['lighting_band'] = lighting_band.astype(int)
    debug['params']['morning_capacity'] = cap_m
    debug['params']['evening_capacity'] = cap_e

    return cap_m, cap_e, lighting_band, debug
