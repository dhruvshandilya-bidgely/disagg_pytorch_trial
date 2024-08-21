"""
Author - Anand Kumar Singh
Date - 19th Feb 2020
Based on disagg run mode set solar config
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def update_mtd_hsm(solar_estimation_config, previous_estimation_hsm, sampling_rate_scaling_factor):
    """
    Update HSM when run mode is mtd

    Parameters:
        solar_estimation_config     (dict)              : Dictionary containing all solar config values
        previous_estimation_hsm     (dict)              : Dictionary containing previous hsm
        sampling_rate_scaling_factor(int)               : calculated factor to adjust hsm

    Returns:
        solar_estimation_config     (dict)              : Dictionary containing all solar config values
    """

    # Reading attributes from in API HSMs
    if type(previous_estimation_hsm.get('attributes').get('temp_lower')) == list:
        solar_estimation_config['normalisation_thresholds']['temp']['lower'] = \
            previous_estimation_hsm.get('attributes').get('temp_lower')[0]

        solar_estimation_config['normalisation_thresholds']['temp']['upper'] = \
            previous_estimation_hsm.get('attributes').get('temp_upper')[0]

        solar_estimation_config['normalisation_thresholds']['wind']['lower'] = \
            previous_estimation_hsm.get('attributes').get('wind_lower')[0]

        solar_estimation_config['normalisation_thresholds']['wind']['upper'] = \
            previous_estimation_hsm.get('attributes').get('wind_upper')[0]

        solar_estimation_config['normalisation_thresholds']['sky_cover']['lower'] = \
            previous_estimation_hsm.get('attributes').get('sky_cover_lower')[0]

        solar_estimation_config['normalisation_thresholds']['sky_cover']['upper'] = \
            previous_estimation_hsm.get('attributes').get('sky_cover_upper')[0]

        solar_estimation_config['previous_capacity'] = \
            previous_estimation_hsm.get('attributes').get('capacity')[0]

        if solar_estimation_config['previous_capacity'] is not None:
            solar_estimation_config['previous_capacity'] = solar_estimation_config['previous_capacity'] * sampling_rate_scaling_factor

    # Reading attributes from in memory HSMs
    if type(previous_estimation_hsm.get('attributes').get('temp_lower')) == np.float64:
        solar_estimation_config['normalisation_thresholds']['temp']['lower'] = \
            previous_estimation_hsm.get('attributes').get('temp_lower')

        solar_estimation_config['normalisation_thresholds']['temp']['upper'] = \
            previous_estimation_hsm.get('attributes').get('temp_upper')

        solar_estimation_config['normalisation_thresholds']['wind']['lower'] = \
            previous_estimation_hsm.get('attributes').get('wind_lower')

        solar_estimation_config['normalisation_thresholds']['wind']['upper'] = \
            previous_estimation_hsm.get('attributes').get('wind_upper')

        solar_estimation_config['normalisation_thresholds']['sky_cover']['lower'] = \
            previous_estimation_hsm.get(
                'attributes').get('sky_cover_lower')
        solar_estimation_config['normalisation_thresholds']['sky_cover']['upper'] = \
            previous_estimation_hsm.get('attributes').get('sky_cover_upper')

        solar_estimation_config['previous_capacity'] = \
            previous_estimation_hsm.get('attributes').get('capacity')

        if solar_estimation_config['previous_capacity'] is not None:
            solar_estimation_config['previous_capacity'] = solar_estimation_config['previous_capacity'] * sampling_rate_scaling_factor

    return solar_estimation_config


def update_hsm_in_solar_config(disagg_input_object, solar_estimation_config, logger):
    """
    Update solar config based on run mode

    Parameters:
        disagg_input_object         (dict)              : Dictionary containing all inputs
        solar_estimation_config     (dict)              : Dictionary containing all solar config values
        logger                      (logger)            : solar logger

    Returns:
        solar_estimation_config     (dict)              : Dictionary containing all solar config values
    """

    disagg_mode = disagg_input_object.get('config').get('disagg_mode')
    previous_estimation_hsm = disagg_input_object.get('appliances_hsm').get('solar')

    if not previous_estimation_hsm:
        solar_estimation_config['normalisation_thresholds'] = dict({})
        return solar_estimation_config

    # calculating factor to adjust hsm based difference sampling rate of current run and previous run
    sampling_rate_scaling_factor = get_sampling_rate_factor(disagg_input_object, logger)

    # Update solar config on mtd to use thresholds
    if disagg_mode == 'mtd':
        solar_estimation_config = update_mtd_hsm(solar_estimation_config, previous_estimation_hsm, sampling_rate_scaling_factor)
    else:
        solar_estimation_config['normalisation_thresholds'] = dict({})

    # Set config for incremental disagg mode
    if disagg_mode == 'incremental':
        if type(previous_estimation_hsm.get('attributes').get('r_squared_threshold')) == list:
            solar_estimation_config['r_squared_threshold'] = \
                previous_estimation_hsm.get('attributes').get('r_squared_threshold')[0]

            residual_capacity_lower = previous_estimation_hsm.get('attributes').get('residual_capacity_lower')
            residual_capacity_lower = np.array(residual_capacity_lower) * sampling_rate_scaling_factor
            residual_r_squared_lower = previous_estimation_hsm.get('attributes').get('residual_r_squared_lower')
            residual_r_squared_lower = np.array(residual_r_squared_lower)
            residual_capacity_upper = previous_estimation_hsm.get('attributes').get('residual_capacity_upper')
            residual_capacity_upper = np.array(residual_capacity_upper) * sampling_rate_scaling_factor
            residual_r_squared_upper = previous_estimation_hsm.get('attributes').get('residual_r_squared_upper')
            residual_r_squared_upper = np.array(residual_r_squared_upper)
            residual_good_days = previous_estimation_hsm.get('attributes').get('residual_good_days')
            residual_good_days = np.array(residual_good_days)

            residual_capacity_array = np.c_[residual_good_days, residual_capacity_lower, residual_r_squared_lower, residual_capacity_upper, residual_r_squared_upper]
            solar_estimation_config['residual_capacity_array'] = residual_capacity_array

        if type(previous_estimation_hsm.get('attributes').get('r_squared_threshold')) == np.float64:
            solar_estimation_config['r_squared_threshold'] = \
                previous_estimation_hsm.get('attributes').get('r_squared_threshold')
            residual_capacity = np.array(previous_estimation_hsm.get('attributes').get('residual_capacity')) * sampling_rate_scaling_factor
            residual_r_squared = previous_estimation_hsm.get('attributes').get('residual_r_squared')
            residual_good_days = previous_estimation_hsm.get('attributes').get('residual_good_days')

            residual_capacity_array = np.c_[residual_good_days, residual_capacity, residual_r_squared]
            solar_estimation_config['residual_capacity_array'] = residual_capacity_array


        if type(solar_estimation_config.get('r_squared_threshold')) != type(None):
            min_day_start = disagg_input_object.get('input_data')[:, Cgbdisagg.INPUT_DAY_IDX].min()

            r_square_threshold_condition = \
                solar_estimation_config.get('r_squared_threshold') < solar_estimation_config.get(
                    'neg_cap_estimation').get('r_square_threshold').get('low_r_square')

            # if r squared threshold is met then set to none
            if r_square_threshold_condition:
                solar_estimation_config['out_bill_cycles'] = None
                solar_estimation_config['residual_capacity_array'] = None

            # Can't merge these since one number of days condition can be validated unless above is met

            # if number of days conditions is not met then set to none
            elif residual_capacity_array[residual_capacity_array[:, 0] >= min_day_start].shape[0] < 14:
                solar_estimation_config['out_bill_cycles'] = None
                solar_estimation_config['residual_capacity_array'] = None

    else:
        solar_estimation_config['residual_capacity_array'] = None
        solar_estimation_config['r_squared_threshold'] = 0
        solar_estimation_config['out_bill_cycles'] = None

    return solar_estimation_config


def get_sampling_rate_factor(disagg_input_object, logger):

    """
    calculating factor to adjust hsm based difference sampling rate of current run and previous run
    # for example if previous run is of 3600 sec sampling rate while current run is 900 sec,
    # this factor will scale down the capacity hsm by 0.25 factor

    Parameters:
        disagg_input_object              (dict)              : Dictionary containing all inputs
        logger                           (logger)            : solar logger

    Returns:
        sampling_rate_scaling_factor     (int)               : calculated factor to adjust hsm
    """

    previous_li_hsm = disagg_input_object.get('appliances_hsm').get('li')

    sampling_rate_scaling_factor = 1

    if (previous_li_hsm.get('attributes') is not None) and (previous_li_hsm.get('attributes').get('sleep_hours') is not None):

        sleep_hours = previous_li_hsm.get('attributes').get('sleep_hours')

        if type(sleep_hours) == list:
            previous_run_sampling_rate = Cgbdisagg.SEC_IN_HOUR * (Cgbdisagg.HRS_IN_DAY / len(sleep_hours))
            sampling_rate_scaling_factor = disagg_input_object.get('config').get('sampling_rate') / previous_run_sampling_rate

        else:
            logger.warning('Not able to process lighting HSM for sampling rate scaling in solar module HSM | ')

    else:
        logger.warning('Not able to fetch lighting HSM for sampling rate scaling in solar module HSM | ')

    logger.info('Calculated HSM scaling factor based on sampling rate | %s', sampling_rate_scaling_factor)

    return sampling_rate_scaling_factor
