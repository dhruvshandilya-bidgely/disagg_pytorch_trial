"""
Author - Anand Kumar Singh
Date - 26th June 2021
HVAC inefficiency HSM related operations
"""

# Import python packages
import logging
import numpy as np

from python3.config.Cgbdisagg import Cgbdisagg


def prepare_peer_comparison_vector(input_inefficiency_object, output_inefficiency_object, logger_pass, device):

    """
    This function prepares features for hvac peer comparison

    Parameters:
        input_inefficiency_object       (dict)          dictionary containing all input the information
        output_inefficiency_object      (dict)          dictionary containing all output information
        logger_pass                     (object)        logger object
        device                          (str)           string indicating device, either AC or SH
    Returns:
        input_inefficiency_object       (dict)          dictionary containing all input the information
        output_inefficiency_object      (dict)          dictionary containing all output information
    """

    logger_local = logger_pass.get("logger").getChild("prepare_peer_comparison_vector")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    logger.debug('Starting HVAC peer comparison change | {}'.format(device))

    # Update rate plan when available as input
    rate_plan = np.ones((Cgbdisagg.HRS_IN_DAY, ), dtype=float)

    # Get device set point
    set_point = input_inefficiency_object.get(device).get('setpoint')
    hourly_hvac = input_inefficiency_object.get(device).get('demand_hvac_pivot').get('values')

    zero_hvac_cons = hourly_hvac == 0
    hourly_hvac[zero_hvac_cons] = np.nan

    days_without_hvac = np.nansum(hourly_hvac, axis=1) == 0
    hourly_hvac_days_with_hvac = hourly_hvac[~days_without_hvac, :]

    nan_idx = np.isnan(hourly_hvac_days_with_hvac)
    hourly_hvac_days_with_hvac[nan_idx] = 0
    hvac_hour_profile = np.nanmedian(hourly_hvac_days_with_hvac, axis=0)
    hvac_hour_profile = hvac_hour_profile / np.nanmax(hvac_hour_profile)

    # Prepare hod vector
    positive_hvac_idx = hourly_hvac_days_with_hvac > 0
    hourly_hvac_days_with_hvac[positive_hvac_idx] = 1
    hvac_usage_fraction = np.nanmean(hourly_hvac_days_with_hvac, axis=0)

    return_dictionary = {'set_point': set_point,
                         'hvac_profile': hvac_hour_profile,
                         'peak_hvac_profile': hvac_hour_profile * rate_plan,
                         'hour_usage_hours': hvac_usage_fraction}

    output_inefficiency_object[device]['peer_comparison'] = return_dictionary

    return input_inefficiency_object, output_inefficiency_object
