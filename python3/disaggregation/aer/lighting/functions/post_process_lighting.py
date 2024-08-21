"""
Author - Mayank Sharan
Date - 26/11/2019
post processing lighting applies artificial seasonality to billing cycle level estimates
"""

# Import python packages

import copy
import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.lighting.init_lighting_params import setpoint_list


def historic_post_processing(dark_hours, estimate, scalar, monthly_lighting_estimate, config, logger):

    """Utility to post process for historical mode"""

    idx_ts = 0
    idx_value = 1
    idx_numdays = 2
    idx_dark_hours = 3

    # Find if postprocess is needed or not. Select months with 15+ days and
    # compare 2 hottest months to 2 coldest months lighting estimation per
    # day

    num_months = estimate.shape[0]

    if int(num_months / 2) % 2 == 0:
        spring_months = setpoint_list(np.round(num_months / 2), np.round(num_months / 2) + 1)
    else:
        spring_months = setpoint_list(np.round(num_months / 2), np.round(num_months / 2) + 1) - 1

    estimate = estimate[estimate[:, idx_numdays] >= config['SEASONALITY']['minDays'], :]
    estimate = estimate[estimate[:, idx_dark_hours].argsort()]

    buffer = config['SEASONALITY']['buffer'] * 1000

    try:
        compare_summer_winter = (np.sum(estimate[0:2, idx_value]) / np.sum(estimate[0:2, idx_numdays])) <= \
                                (np.sum(estimate[-2:, idx_value]) / np.sum(estimate[-2:, idx_numdays]) + buffer / 30)
        compare_summer_spring = (np.sum(estimate[0:2, idx_value]) / np.sum(estimate[0:2, idx_numdays])) <= \
                                (sum(estimate[spring_months, idx_value]) / np.sum(estimate[spring_months, idx_numdays])
                                 + buffer / 30)
    except IndexError:
        scalar['scaling'] = 1
        scalar['DarkestMonth_nightHours'] = dark_hours[0] + 0.5
        scalar['secondLightestMonth_nightHours'] = dark_hours[0] - 0.5
        scalar['maxLightingPerDay'] = (monthly_lighting_estimate[0, idx_value] /
                                       monthly_lighting_estimate[0, idx_numdays])
        monthly_lighting_estimate = monthly_lighting_estimate[:, idx_ts:idx_value + 1]

        return monthly_lighting_estimate, scalar, True

    if compare_summer_winter and compare_summer_spring:

        # Summer per day consumption less than winter's. No need of post processing

        scalar['scaling'] = 1
        scalar['DarkestMonth_nightHours'] = dark_hours[-1] + 0.5
        scalar['secondLightestMonth_nightHours'] = dark_hours[1] - 0.5
        scalar['maxLightingPerDay'] = (monthly_lighting_estimate[-1, idx_value] /
                                       monthly_lighting_estimate[-1, idx_numdays])

        logger.info('Skipping seasonality summer consumption is less than winter consumption |')
        return monthly_lighting_estimate, scalar, True

    return monthly_lighting_estimate, scalar, False


def post_process_lighting(input_data, config, monthly_lighting_estimate, use_hsm_present, hsm_temp, logger_pass):
    """ Return:  monthlyLightingEstimate, scalar """

    logger_base = logger_pass.get('logger_base').getChild('post_process_lighting')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Initialise indices to use

    idx_ts = 0
    idx_value = 1
    idx_numdays = 2
    idx_dark_hours = 3
    idx_total_cons = 4

    gb_sampling = input_data[1, Cgbdisagg.INPUT_EPOCH_IDX] - input_data[0, Cgbdisagg.INPUT_EPOCH_IDX]

    scalar = {
        'scaling': None,
        'DarkestMonth_nightHours': None,
        'secondLightestMonth_nightHours': None,
        'maxLightingPerDay': None,
    }

    _, _, c, d = np.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_index=True, return_inverse=True,
                           return_counts=True)

    monthly_lighting_estimate[:, idx_numdays] = d / (Cgbdisagg.SEC_IN_DAY / gb_sampling)
    tx = np.bincount(c, weights=input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    for i in range(monthly_lighting_estimate.shape[0]):
        month_idx = input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == monthly_lighting_estimate[i, idx_ts]
        # dark hours
        monthly_lighting_estimate[i, idx_dark_hours] = np.nanmedian(Cgbdisagg.HRS_IN_DAY -
                                                                    (input_data[month_idx, Cgbdisagg.INPUT_SUNSET_IDX] -
                                                                     input_data[month_idx, Cgbdisagg.INPUT_SUNRISE_IDX])
                                                                    / Cgbdisagg.SEC_IN_HOUR)
        if np.isnan(tx[i]):
            monthly_lighting_estimate[i, idx_total_cons] = np.nansum(input_data[month_idx,
                                                                                Cgbdisagg.INPUT_CONSUMPTION_IDX])
        else:
            monthly_lighting_estimate[i, idx_total_cons] = tx[i]

    estimate = monthly_lighting_estimate

    # Make a copy to generate new HSM parameters if monthly/historical mode

    monthly_lighting_estimate_copy = copy.deepcopy(monthly_lighting_estimate)

    dark_hours = monthly_lighting_estimate[:, idx_dark_hours]

    # Apply seasonality from HSM
    if use_hsm_present and not(len(hsm_temp) == 0):

        logger.info('Running seasonality with HSM |')

        # Introduce Seasonality from HSM
        hsm_scaling = hsm_temp.get('scaling')

        second_lightest_month_night_hours = hsm_temp.get('secondLightestMonth_nightHours')
        darkest_month_night_hours = hsm_temp.get('DarkestMonth_nightHours')
        max_lighting_per_day = hsm_temp.get('maxLightingPerDay')

        scaling = 1 - (1 - config['SEASONALITY']['lightMonthScaling']) * \
            (darkest_month_night_hours - dark_hours)/(darkest_month_night_hours - second_lightest_month_night_hours)

        scaling = np.fmax(np.fmin(1, scaling), config['SEASONALITY']['lowerBoundLightMonthScaling'])

        if np.sum(hsm_scaling - 1) == 0:
            try:
                scaling = np.ones(shape=(len(scaling),))
            except TypeError:
                scaling = np.ones(shape=(1,))

        scalar['scaling'] = scaling
        scalar['DarkestMonth_nightHours'] = darkest_month_night_hours
        scalar['secondLightestMonth_nightHours'] = second_lightest_month_night_hours
        scalar['maxLightingPerDay'] = max_lighting_per_day

        max_estimate = np.minimum(
            max_lighting_per_day * np.multiply(monthly_lighting_estimate[:, idx_numdays], scaling),
            monthly_lighting_estimate[:, idx_total_cons])

        monthly_lighting_estimate[:, idx_value] = max_estimate

    if len(dark_hours) < 3:

        logger.info('Skipping seasonality not enough billing cycles |')

        # Set the parameters with the first element available
        scalar['scaling'] = 1
        scalar['DarkestMonth_nightHours'] = dark_hours[0] + 0.5
        scalar['secondLightestMonth_nightHours'] = dark_hours[0] - 0.5
        scalar['maxLightingPerDay'] = (monthly_lighting_estimate[0, idx_value] /
                                       monthly_lighting_estimate[0, idx_numdays])
        monthly_lighting_estimate = monthly_lighting_estimate[:, idx_ts:idx_value + 1]

        return monthly_lighting_estimate, scalar

    # Continue if historical mode is run

    monthly_lighting_estimate, scalar, do_return = historic_post_processing(dark_hours, estimate, scalar,
                                                                            monthly_lighting_estimate, config, logger)

    if do_return:
        return monthly_lighting_estimate, scalar

    # For new HSM, work on monthly_lighting_estimate_copy since monthlyLightingEstimate might be changed

    dark_hours = monthly_lighting_estimate_copy[:, idx_dark_hours]

    # Introduce post processing

    sorted_idx = np.argsort(dark_hours)
    second_lightest_month_night_hours = dark_hours[sorted_idx[1]]
    darkest_month_night_hours = dark_hours[sorted_idx[-1]]

    logger.info('Introducing seasonality |')
    logger.info('Second Lightest Month Night Hours - %s |', str(second_lightest_month_night_hours))
    logger.info('Darkest Month Night Hours - %s |', str(darkest_month_night_hours))

    scaling = 1-(1-config['SEASONALITY']['lightMonthScaling']) * \
                (darkest_month_night_hours - dark_hours)/(darkest_month_night_hours - second_lightest_month_night_hours)
    scaling = np.maximum(np.minimum(1, scaling), config['SEASONALITY']['lowerBoundLightMonthScaling'])

    # Test results are  lowerBoundLightMonthScaling = 0.8, lightMonthScaling = 0.82  !!

    max_lighting_per_day = (monthly_lighting_estimate_copy[sorted_idx[-1], idx_value] /
                            monthly_lighting_estimate_copy[sorted_idx[-1], idx_numdays])

    max_estimate = np.minimum(max_lighting_per_day * scaling *
                              monthly_lighting_estimate_copy[:, idx_numdays],
                              monthly_lighting_estimate_copy[:, idx_total_cons])

    scalar['DarkestMonth_nightHours'] = darkest_month_night_hours
    scalar['secondLightestMonth_nightHours'] = second_lightest_month_night_hours
    scalar['maxLightingPerDay'] = max_lighting_per_day
    scalar['scaling'] = scaling

    if ~(use_hsm_present and len(hsm_temp) == 0):
        monthly_lighting_estimate[:, idx_value] = max_estimate

    monthly_lighting_estimate = monthly_lighting_estimate[:, idx_ts:idx_value + 1]

    return monthly_lighting_estimate, scalar
