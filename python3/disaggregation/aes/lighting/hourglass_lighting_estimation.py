"""
Author: Neelabh Goyal
Date:   14-June-2023
Called to estimate external lighting
"""

# Import python packages
import copy
import logging
import numpy as np
import pandas as pd

# Import from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aes.lighting.get_hourglass_params import get_hourglass_params


def external_lighting_estimation(in_data, hourglass_dict, ao_hvac, static_params, logger_pass):
    """
    Function to estimate hourglass(external) lighting
    Parameters:
        in_data          (pd.DataFrame)  : Dataframe with all the raw input data
        hourglass_dict   (dict)          : Dictionary with hourglass pattern detection specific data
        ao_hvac          (np.array)      : Numpy array containing epoch level AO HVAC
        static_params    (dict/bool)     : Dictionary with hourglass specific constants or bool if dict is not passed
        logger_pass      (logging object): Logging object dictionary to be used here for logging

    Returns:
        lighting_estimate (np.array) : Array with final estimation for external lighting
    """

    logger_local = logger_pass.get("logger").getChild("external_lighting_estimation")
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    if not static_params:
        static_params = get_hourglass_params()

    params = static_params.get('lighting_estimation')

    # Allow NaN values up to 3 days to be filled in by previous day's values
    morning_hours = pd.DataFrame(hourglass_dict['morning_hours']).ffill(limit=3).values
    evening_hours = pd.DataFrame(hourglass_dict['evening_hours']).ffill(limit=3).values

    # Creating input_data_sans_hvac dataframe after removing AO-HVAC. It will be used to identify the estimation level.
    in_data_copy = copy.deepcopy(in_data)
    in_data_copy['raw-ao'] -= ao_hvac
    in_data_copy['raw-ao'][in_data_copy['raw-ao'] < 0] = 0
    input_data_sans_hvac = np.nan_to_num(in_data_copy.pivot_table(index='date', columns='time', values='raw-ao').values)

    # Removing daily AO
    sampling = hourglass_dict.get('sampling')
    daily_ao = np.sort(input_data_sans_hvac, axis=1)[:, int(params.get('daily_ao_hour') * sampling)]
    input_data_sans_hvac -= np.tile(daily_ao.reshape(-1, 1), (1, input_data_sans_hvac.shape[1]))
    input_data_sans_hvac[input_data_sans_hvac < 0] = 0

    del in_data_copy

    # Creating input_data dataframe without removing AO HVAC. This will be used for final estimations
    input_data = np.nan_to_num(in_data.pivot_table(index='date', columns='time', values='raw-ao').values)
    daily_ao = np.sort(input_data, axis=1)[:, int(params.get('daily_ao_hour') * sampling)]
    input_data -= np.tile(daily_ao.reshape(-1, 1), (1, input_data.shape[1]))
    input_data[input_data < 0] = 0

    hourglass_estimate = np.zeros_like(input_data)
    hourglass_estimate_final = np.zeros_like(input_data)

    buffer = int(np.minimum(sampling, params.get('max_allowed_buffer')))
    for row in range(len(input_data)):
        # Allow up to 2 samples as buffer since smoothening might have truncated the external lighting
        morn_col = int(np.nan_to_num(morning_hours[row])) + buffer
        eve_col = int(np.nan_to_num(evening_hours[row])) - buffer

        if not eve_col == 0:
            # Adding input_data to hourglass estimation array for preliminary estimation
            hourglass_estimate[row, :morn_col] = input_data_sans_hvac[row, :morn_col]
            hourglass_estimate[row, eve_col:] = input_data_sans_hvac[row, eve_col:]

            # Adding input_data to hourglass estimation final array for final estimation
            hourglass_estimate_final[row, :morn_col] = input_data[row, :morn_col]
            hourglass_estimate_final[row, eve_col:] = input_data[row, eve_col:]

    hourglass_estimate = np.where(hourglass_estimate < 0, 0, hourglass_estimate)
    hourglass_estimate[hourglass_estimate == 0] = np.NaN

    # To identify the estimate of lighting, we create an equally spaced line b/w 65th and 99th percentile values of
    # external light using "linspace" and identify the datapoint in the external light that is farthest from this line.
    # This point should ideally be our point of inflection, above which we have noise because of other appliances.

    min_lighting_val = np.nanpercentile(hourglass_estimate, params.get('min_estimation_percentile'))
    max_lighting_val = np.nanpercentile(hourglass_estimate, params.get('max_estimation_percentile'))

    nz_external_light = hourglass_estimate[np.logical_and(hourglass_estimate >= min_lighting_val,
                                                          hourglass_estimate <= max_lighting_val)]

    nz_external_light = np.sort(nz_external_light)
    reference_line = np.linspace(min_lighting_val, max_lighting_val, len(nz_external_light))
    lighting_cap = nz_external_light[np.argmax(reference_line - nz_external_light)]
    if lighting_cap < np.minimum(params.get('min_lighting_estimate')/sampling, max_lighting_val):
        lighting_cap = np.minimum(params.get('min_lighting_estimate')/sampling, max_lighting_val)

    logger.info(' Maximum epoch level cap for external lighting estimate calculated as | {} W'.format(lighting_cap))

    del hourglass_estimate, input_data_sans_hvac

    lighting_estimate = hourglass_estimate_final.clip(None, lighting_cap)
    residue_data = input_data - lighting_estimate
    lighting_estimate = np.where(np.logical_and(residue_data < lighting_estimate * params.get('min_residue_frac'),
                                                lighting_estimate > 0), input_data, lighting_estimate)

    # In case the remaining input data is within +- 10% of the lighting cap, take the complete data as lighting
    lower_thresh = lighting_cap * (1 - params.get('min_residue_frac'))
    higher_thresh = lighting_cap * (1 + params.get('min_residue_frac'))
    residue_data = input_data - lighting_estimate

    if np.count_nonzero(residue_data) < np.count_nonzero(input_data) * params.get('min_residue_frac') and \
            lower_thresh <= np.nanmedian(residue_data[residue_data > 0]) <= higher_thresh:
        lighting_estimate = copy.deepcopy(input_data)
        logger.info('Residue after removing lighting consumption is similar to the lighting cap | '
                    'Adding residue to lighting')

    return lighting_estimate
