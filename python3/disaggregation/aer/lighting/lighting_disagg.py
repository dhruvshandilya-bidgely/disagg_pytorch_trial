"""
Author - Mayank Sharan
Date - 26/11/2019
lighting disagg runs disaggregation for lighting in different modes
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.disaggregation.aer.lighting.functions.get_day_data import get_day_data
from python3.disaggregation.aer.lighting.functions.detect_seasonal import detect_seasonal
from python3.disaggregation.aer.lighting.functions.remove_vacation import remove_vacation
from python3.disaggregation.aer.lighting.functions.get_lighting_params import get_lighting_params
from python3.disaggregation.aer.lighting.functions.percentile_vertical import percentile_vertical
from python3.disaggregation.aer.lighting.functions.get_monthly_estimate import get_monthly_estimate
from python3.disaggregation.aer.lighting.functions.remove_daily_minimum import remove_daily_minimum
from python3.disaggregation.aer.lighting.functions.percentile_horizontal import percentile_horizontal
from python3.disaggregation.aer.lighting.functions.post_process_lighting import post_process_lighting
from python3.disaggregation.aer.lighting.functions.get_lighting_estimate import get_lighting_estimate
from python3.disaggregation.aer.lighting.functions.push_lighting_hsm_attributes import push_lighting_hsm_attributes


def extract_hsm(hsm_in):
    """Utility to pull out HSM attributes, Removed to reduce complexity"""

    # Extract attributes from hsm for hsm run

    is_hsm_empty = (hsm_in is None) or (len(hsm_in) == 0)

    # Extract attributes if possible

    if not is_hsm_empty:
        hsm_attr = hsm_in.get('attributes')
    else:
        hsm_attr = None

    # Extract parameters from within hsm

    hsm_temp = None

    if hsm_attr is not None and len(hsm_attr) > 0:

        # For python

        if type(hsm_attr.get('morning_capacity')) == list:

            hsm_temp = {
                'lighting_band': np.array(hsm_attr.get('lighting_band')).astype(bool),
                'morning_capacity': float(hsm_attr.get('morning_capacity')[0]),
                'evening_capacity': float(hsm_attr.get('evening_capacity')[0]),
                'morning_interpolation': float(hsm_attr.get('morning_interpolation')[0]),
                'evening_interpolation': float(hsm_attr.get('evening_interpolation')[0]),
                'smoothing_noise_bound': float(hsm_attr.get('smoothing_noise_bound')[0]),
                'period': float(hsm_attr.get('period')[0]),
                'maxLightingPerDay': float(hsm_attr.get('maxLightingPerDay')[0]),
                'scaling': hsm_attr.get('scaling'),
                'secondLightestMonth_nightHours': hsm_attr.get('secondLightestMonth_nightHours')[0],
                'DarkestMonth_nightHours': hsm_attr.get('DarkestMonth_nightHours')[0],
            }

        # For MATLAB

        else:
            hsm_temp = {
                'lighting_band': np.array(hsm_attr.get('lighting_band')).astype(bool),
                'morning_capacity': float(hsm_attr.get('morning_capacity')),
                'evening_capacity': float(hsm_attr.get('evening_capacity')),
                'morning_interpolation': float(hsm_attr.get('morning_interpolation')),
                'evening_interpolation': float(hsm_attr.get('evening_interpolation')),
                'smoothing_noise_bound': float(hsm_attr.get('smoothing_noise_bound')),
                'period': float(hsm_attr.get('period')),
                'maxLightingPerDay': float(hsm_attr.get('maxLightingPerDay')),
                'scaling': hsm_attr.get('scaling'),
                'secondLightestMonth_nightHours': hsm_attr.get('secondLightestMonth_nightHours'),
                'DarkestMonth_nightHours': hsm_attr.get('DarkestMonth_nightHours'),
            }

        if type(hsm_temp.get('scaling')) == list:
            if len(hsm_temp.get('scaling')) == 1:
                hsm_temp['scaling'] = hsm_temp.get('scaling')[0]
            else:
                hsm_temp['scaling'] = np.array(hsm_temp.get('scaling'))

    return hsm_temp


def lighting_disagg(input_data, config, vacation_output, hist_mode, hsm_in, bypass_hsm, logger_pass):

    """
    Parameters:
        input_data          (np.ndarray)        : 21 column matrix containing the data
        config              (dict)              : Contains all local parameters needed to run lighting
        vacation_output     (np.ndarray)        : Vacation output at timestamp level to reject days
        hist_mode           (bool)              : True -> hsm creation code runs
        hsm_in              (dict)              : Previously created hsm to be used
        bypass_hsm          (bool)              : True -> We do not run the hsm code
        logger_pass         (dict)              : Contains base logger and logging dictionary

    Output:
        monthly_estimate    (np.ndarray)        : Billing cycle level KWh estimation of lighting usage
        debug               (dict)              : All variable values needed to debug
        hsm                 (dict)              : Parameters for use in future iterations
    """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('lighting_disagg')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    # Initialize debug hsm

    debug_hsm = {
        'data': None,
        'results': None,
        'lighting': None,
        'season': None,
    }

    # Initialise other variables needed for the disagg

    data_lighting = None
    lighting_band = None
    cap_m = None
    cap_e = None
    smoothing_noise_bound = None

    # Remove vacation days from the input data

    input_data = remove_vacation(input_data, vacation_output)
    period = input_data[1, Cgbdisagg.INPUT_EPOCH_IDX] - input_data[0, Cgbdisagg.INPUT_EPOCH_IDX]

    # Convert input data into 2d matrix with each row as a day

    t_before_day_data = datetime.now()
    data, month_ts, daily_ts, epoch_ts = get_day_data(input_data)
    t_after_day_data = datetime.now()

    logger.debug('Get day data took | %.3f s', get_time_diff(t_before_day_data, t_after_day_data))

    # Remove daily minimum to eliminate continuously running appliances

    t_before_min_removal = datetime.now()
    data_min_removed = remove_daily_minimum(data, 1)
    t_after_min_removal = datetime.now()

    logger.debug('Daily min removal took | %.3f s', get_time_diff(t_before_min_removal, t_after_min_removal))

    # Apply vertical percentile to the data

    t_before_percentile_v = datetime.now()
    data_filtered = percentile_vertical(data_min_removed, config.get('VERTICAL_WINDOW'),
                                        config.get('VERTICAL_PERCENTILE'))
    t_after_percentile_v = datetime.now()

    logger.debug('Vertical Percentile took | %.3f s', get_time_diff(t_before_percentile_v, t_after_percentile_v))

    # Apply horizontal percentile to the data

    t_before_percentile_h = datetime.now()
    data_filtered = percentile_horizontal(data_filtered, period, config.get('HORIZONTAL_WINDOW'),
                                          config.get('HORIZONTAL_PERCENTILE'))
    t_after_percentile_h = datetime.now()

    logger.debug('Horizontal Percentile took | %.3f s', get_time_diff(t_before_percentile_h, t_after_percentile_h))

    # Remove daily min and round values to avoid float precision issues

    t_before_min_rem2 = datetime.now()
    data_filtered = remove_daily_minimum(data_filtered, 0)
    data_filtered = np.round(data_filtered, 5)
    t_after_min_rem2 = datetime.now()

    logger.debug('Daily Min Removal 2 took | %.3f s', get_time_diff(t_before_min_rem2, t_after_min_rem2))

    # Get utility variables number of periods in a day and hour to index multiplier

    num_periods = data_filtered.shape[1]
    pd_mult = num_periods / 24

    # Set values to run different modes

    hsm_temp = extract_hsm(hsm_in)

    bypass_hsm = bypass_hsm or (hsm_temp is None) or (len(hsm_temp) == 0)

    # Initialize variables for hsm output

    hsm_based_lighting = []

    # Run the hsm based output

    if not bypass_hsm:

        logger.info('Entering HSM usage mode |')

        # Extract parameters out of HSM for the run

        lighting_band = hsm_temp.get('lighting_band')
        logger.info('Extracted lighting band | %s', str(lighting_band).replace('\n', ' '))

        cap_m = hsm_temp.get('morning_capacity')
        logger.info('Extracted morning capacity | %.3f', cap_m)

        cap_e = hsm_temp.get('evening_capacity')
        logger.info('Extracted evening capacity | %.3f', cap_e)

        int_m = hsm_temp.get('morning_interpolation')
        logger.info('Extracted morning interpolation | %.3f', int_m)

        int_e = hsm_temp.get('evening_interpolation')
        logger.info('Extracted evening interpolation | %.3f', int_e)

        smoothing_noise_bound = hsm_temp.get('smoothing_noise_bound')
        logger.info('Extracted smoothing noise bound | %.3f', smoothing_noise_bound)

        # Prepare data lighting for estimate

        data_lighting_hsm = copy.deepcopy(data_filtered)
        lighting_band = lighting_band[:data_lighting_hsm.shape[1]]
        lighting_band = np.r_[lighting_band,
                              np.full(shape=(max(0, data_lighting_hsm.shape[1] - lighting_band.size),),
                                      fill_value=False)]

        data_lighting_hsm[:, ~lighting_band] = 0

        if config['MIN_NOISE_RATIO'] * smoothing_noise_bound < np.nanmax(data_lighting_hsm):
            logger.info('Using smoothing noise bound |')
            data_lighting_hsm[data_lighting_hsm < smoothing_noise_bound] = 0
        else:
            logger.info('Not using smoothing noise bound |')

        # Run estimate code to pull out lighting

        t_before_estimation = datetime.now()
        hsm_based_lighting, debug_hsm = get_lighting_estimate(data_lighting_hsm, cap_m, cap_e, int_m, int_e,
                                                              num_periods, pd_mult, config, debug_hsm, logger_pass)
        t_after_estimation = datetime.now()

        logger.debug('Extracting estimate took | %.3f s', get_time_diff(t_before_estimation, t_after_estimation))

        # Process lighting to remove nan values and apply minimum with original data

        hsm_based_lighting[np.isnan(hsm_based_lighting)] = 0
        hsm_based_lighting = np.minimum(hsm_based_lighting, data_min_removed)
        hsm_based_lighting[np.isnan(data_min_removed)] = 0

        logger.info('HSM usage mode completed |')

    # Now, run fresh lighting for historical mode

    if hist_mode or bypass_hsm:

        logger.info('Entering HSM creation mode |')

        # Detect season in data to reject HVAC

        t_before_season_d = datetime.now()
        seasonal_flag, perc_season, smoothing_noise_bound, debug = detect_seasonal(data_filtered, num_periods,
                                                                                   pd_mult, config, logger_pass)
        t_after_season_d = datetime.now()

        logger.debug('Season detection took | %.3f s', get_time_diff(t_before_season_d, t_after_season_d))

        logger.info('Percentage season calculated | %.3f', perc_season)
        logger.info('Smoothing noise bound calculated | %.3f', smoothing_noise_bound)
        logger.info('Season flag calculated | %s', str(seasonal_flag).replace('\n', ' '))

        data_season_removed = copy.deepcopy(data_filtered)
        data_season_removed[:, seasonal_flag[:, 0]] = 0

        # Extract lighting parameters like capacity and lighting band needed to compute estimate

        t_before_params = datetime.now()
        cap_m, cap_e, lighting_band, debug = get_lighting_params(data_season_removed, smoothing_noise_bound,
                                                                 perc_season, num_periods, pd_mult, config, debug,
                                                                 logger_pass)
        t_after_params = datetime.now()

        logger.debug('Lighting parameters calculation took | %.3f s', get_time_diff(t_before_params, t_after_params))

        cap_m = round(cap_m, 5)
        cap_e = round(cap_e, 5)

        logger.info('Morning capacity calculated | %.3f', cap_m)
        logger.info('Evening capacity calculated | %.3f', cap_e)
        logger.info('Lighting band calculated | %s', str(lighting_band).replace('\n', ' '))

        data_lighting = data_season_removed
        data_lighting[:, ~lighting_band] = 0

        # For a check later in the getLightingEstimate

        int_m = -1
        int_e = -1

        if config['MIN_NOISE_RATIO'] * smoothing_noise_bound < np.nanmax(data_lighting):
            logger.info('Using smoothing noise bound |')
            data_lighting[data_lighting < smoothing_noise_bound] = 0
        else:
            logger.info('Not using smoothing noise bound |')

        # Get Lighting Estimate

        t_before_estimation = datetime.now()
        lighting, debug = get_lighting_estimate(data_lighting, cap_m, cap_e, int_m, int_e,
                                                num_periods, pd_mult, config, debug, logger_pass)
        t_after_estimation = datetime.now()

        logger.debug('Extracting estimate took | %.3f s', get_time_diff(t_before_estimation, t_after_estimation))

        lighting[np.isnan(lighting)] = 0
        lighting = np.minimum(lighting, data_min_removed)
        lighting[np.isnan(data_min_removed)] = 0

        logger.info('HSM creation mode completed |')

    else:
        logger.info('HSM lighting used to populate historical lighting |')
        lighting = copy.deepcopy(hsm_based_lighting)
        debug = copy.deepcopy(debug_hsm)

    if len(hsm_based_lighting) == 0:
        logger.info('HSM lighting populated using historical lighting |')
        hsm_based_lighting = copy.deepcopy(lighting)
        debug_hsm = copy.deepcopy(debug)

    # Calculating usage data
    daily_lighting = np.nansum(lighting, 1)
    daily_consumption = np.nansum(data, 1)
    daily_percentage = daily_lighting * 100. / daily_consumption

    daily_hours = copy.deepcopy(lighting)
    daily_hours[daily_hours > 0] = period / 3600

    # Populating all outputs

    # Populating hsm based lighting estimate

    t_before_post_proc = datetime.now()
    monthly_lighting_estimate_basedonhsm = get_monthly_estimate(month_ts, hsm_based_lighting)
    monthly_lighting_estimate_basedonhsm, scalar_hsm = post_process_lighting(input_data, config,
                                                                             monthly_lighting_estimate_basedonhsm,
                                                                             not (hist_mode or bypass_hsm),
                                                                             hsm_temp, logger_pass)
    t_after_post_proc = datetime.now()

    logger.debug('Post processing for hsm data took | %.3f s', get_time_diff(t_before_post_proc, t_after_post_proc))

    # Populating lighting estimate
    # Comment out later since we do not use this anywhere

    t_before_post_proc = datetime.now()
    monthly_lighting_estimate = get_monthly_estimate(month_ts, lighting)
    monthly_lighting_estimate, scalar = post_process_lighting(input_data, config, monthly_lighting_estimate,
                                                              not (hist_mode or bypass_hsm), hsm_temp, logger_pass)
    t_after_post_proc = datetime.now()

    logger.debug('Post processing for hist data took | %.3f s', get_time_diff(t_before_post_proc, t_after_post_proc))

    # Populating the debug object
    debug['data'] = {
        'data': data,
        'period': period,
        'minremoved': data_min_removed,
        'filtered': data_filtered,
        'in_lighting_band': data_lighting,
        'lighting': lighting,
        'month_ts': month_ts,
        'daily_ts': daily_ts,
        'epoch_ts': epoch_ts,
    }

    debug['results'] = {
        'daily_lighting': daily_lighting,
        'daily_hours': daily_hours,
        'daily_consumption': daily_consumption,
        'daily_percentage': daily_percentage,
        'monthly_lighting': monthly_lighting_estimate,
    }

    debug_hsm['data'] = {
        'lighting': hsm_based_lighting,
        'epoch_ts': epoch_ts,
    }

    # Populate the hsm

    if hist_mode:
        results = {
            'lighting_band': lighting_band,
            'morning_capacity': cap_m,
            'evening_capacity': cap_e,
            'morning_interpolation': debug['lighting']['morning_interpolation'],
            'evening_interpolation': debug['lighting']['evening_interpolation'],
            'smoothing_noise_bound': smoothing_noise_bound,
            'period': period,
            'DarkestMonth_nightHours': scalar['DarkestMonth_nightHours'],
            'secondLightestMonth_nightHours': scalar['secondLightestMonth_nightHours'],
            'maxLightingPerDay': scalar['maxLightingPerDay'],
            'scaling': scalar['scaling'],
        }
        hsm = {
            'timestamp': input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': push_lighting_hsm_attributes(results),
        }
        logger.info('Created lighting HSM is | %s', str(hsm).replace('\n', ' '))
    else:
        hsm = hsm_in

    exit_status = {
        'exit_code': 1,
        'error_list': [],
    }

    return monthly_lighting_estimate_basedonhsm, epoch_ts, debug, debug_hsm, hsm, exit_status
