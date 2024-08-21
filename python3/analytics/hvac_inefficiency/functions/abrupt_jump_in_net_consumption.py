"""
Author - Anand Kumar Singh
Date - 19th Feb 2021
This file contains code for finding abrupt change in net consumption
"""

# Import python packages

import copy
import numpy as np

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params
from python3.analytics.hvac_inefficiency.functions.abrupt_jump_in_hvac_amp import amplitude_outlier_basic_function

RANDOM_STATE = 43


def get_ao_cons_outlier(input_hvac_inefficiency_object, output_hvac_inefficiency_object, device):

    """
    Function to ao consumption outliers

    Parameters:

        input_hvac_inefficiency_object  (dict): Dictionary containing hvac ineff input object parameters
        output_hvac_inefficiency_object (dict): Dictionary containing hvac ineff output object parameters
        device                          (str) : String containing ac or sh identifier

    Returns:
        input_hvac_inefficiency_object  (dict): Dictionary containing hvac ineff input object parameters
        output_hvac_inefficiency_object (dict): Dictionary containing hvac ineff output object parameters
    """

    static_params = hvac_static_params()

    if device == 'ac':
        unconsidered_device = 'sh'
    else:
        unconsidered_device = 'ac'

    # Counting number of hours with HVAC and HVAC potential
    date_index = copy.deepcopy(input_hvac_inefficiency_object.get(device).get('energy_pivot').get('row'))
    total_consumption_matrix = copy.deepcopy(
        input_hvac_inefficiency_object.get(device).get('energy_pivot').get('values'))
    total_consumption_values = superfast_matlab_percentile(total_consumption_matrix.astype(np.float), 10, axis=1,
                                                           method='python')

    hvac_potential_matrix = input_hvac_inefficiency_object.get(device).get('{}_potential_pivot'.format(device)).get(
        'values')

    # Set limit to minimum hvac consumption
    hvac_consumption_matrix =\
        copy.deepcopy(input_hvac_inefficiency_object.get(device).get('ao_hvac_pivot').get('values'))

    hvac_temperature = copy.deepcopy(input_hvac_inefficiency_object.get(device).get('temperature_pivot').get('values'))

    unconsidered_hvac_data_master =\
        copy.deepcopy(input_hvac_inefficiency_object.get(unconsidered_device).get('ao_hvac_pivot').get('values'))

    row_idx = input_hvac_inefficiency_object.get(device).get('{}_potential_pivot'.format(device)).get('row')
    column_idx = input_hvac_inefficiency_object.get(device).get('{}_potential_pivot'.format(device)).get('columns')

    list_of_outliers_dates = np.empty(0, )
    dates_column = 2

    # Counting number of hours with HVAC and HVAC potential
    hvac_hour_count = (hvac_consumption_matrix > 0).sum(axis=1)
    hvac_potential_hour_count = (hvac_potential_matrix > 0).sum(axis=1)
    hvac_temperature_average = np.nanmean(hvac_temperature,  axis=1)
    unconsidered_hvac_count = (unconsidered_hvac_data_master > 0).sum(axis=1)

    # removing all days with NaN mean temperature
    nan_temperature_day = np.isnan(hvac_temperature_average)

    hvac_temperature_average = hvac_temperature_average[~nan_temperature_day]
    unconsidered_hvac_count = unconsidered_hvac_count[~nan_temperature_day]
    hvac_potential_hour_count = hvac_potential_hour_count[~nan_temperature_day]
    hvac_hour_count = hvac_hour_count[~nan_temperature_day]
    date_index = date_index[~nan_temperature_day]
    total_consumption_values = total_consumption_values[~nan_temperature_day]

    # Handling and managing  regions

    valid_idx = ((hvac_hour_count > 0) & (unconsidered_hvac_count == 0) & (hvac_potential_hour_count > 0)
                 & (hvac_potential_hour_count != Cgbdisagg.HRS_IN_DAY))

    valid_idx_zero_saturation = ((hvac_hour_count > 0) & (unconsidered_hvac_count == 0)
                                 & (hvac_potential_hour_count == 0))

    valid_idx_saturation = ((hvac_hour_count > 0) & (unconsidered_hvac_count == 0) & (hvac_potential_hour_count == Cgbdisagg.HRS_IN_DAY))

    # count number of zero potential days

    count_zero_potential = np.sum(valid_idx_zero_saturation)
    count_saturation_potential = np.sum(valid_idx_saturation)
    count_potential = np.sum(valid_idx)

    if count_potential > static_params.get('ineff').get('min_count_potential'):

        if count_saturation_potential < static_params.get('ineff').get('min_count_potential'):
            valid_idx = (valid_idx | valid_idx_saturation)
            valid_idx_saturation = np.zeros_like(valid_idx_saturation, dtype=bool)

        if count_zero_potential < static_params.get('ineff').get('min_count_potential'):
            valid_idx = (valid_idx | valid_idx_zero_saturation)
            valid_idx_zero_saturation = np.zeros_like(valid_idx_zero_saturation, dtype=bool)

    elif count_potential <= static_params.get('ineff').get('min_count_potential'):
        valid_idx_zero_saturation = valid_idx_zero_saturation | valid_idx | valid_idx_saturation
        valid_idx = np.zeros_like(valid_idx, dtype=bool)
        valid_idx_saturation = np.zeros_like(valid_idx_saturation, dtype=bool)

    consumption_outlier_input = np.c_[hvac_potential_hour_count[valid_idx], hvac_temperature_average[valid_idx],
                                      date_index[valid_idx]]

    return_dictionary = dict({})

    if consumption_outlier_input.shape[0] > 1:
        return_dictionary = amplitude_outlier_basic_function(consumption_outlier_input,
                                                             total_consumption_values[valid_idx], 1.5,
                                                             inlier_deviation=1.5, column_number=0,
                                                             offset=0.0)

        list_of_outliers_dates = np.r_[list_of_outliers_dates,
                                       return_dictionary['high_outliers']['quad'][0][:, dates_column],
                                       return_dictionary['high_outliers']['ransac'][0][:, dates_column]]

    consumption_outlier_input = np.c_[hvac_potential_hour_count[valid_idx_zero_saturation],
                                      hvac_temperature_average[valid_idx_zero_saturation],
                                      date_index[valid_idx_zero_saturation]]

    return_dictionary_zero_saturation = dict({})

    if consumption_outlier_input.shape[0] > 1:
        return_dictionary_zero_saturation =\
            amplitude_outlier_basic_function(consumption_outlier_input,
                                             total_consumption_values[valid_idx_zero_saturation], 1.5,
                                             inlier_deviation=1.5, column_number=1, offset=0.0)

        list_of_outliers_dates = np.r_[list_of_outliers_dates,
                                       return_dictionary_zero_saturation['high_outliers']['quad'][0][:, dates_column],
                                       return_dictionary_zero_saturation['high_outliers']['ransac'][0][:, dates_column]]

    consumption_outlier_input = np.c_[hvac_potential_hour_count[valid_idx_saturation],
                                      hvac_temperature_average[valid_idx_saturation], date_index[valid_idx_saturation]]

    return_dictionary_saturation = dict({})
    if consumption_outlier_input.shape[0] > 1:
        return_dictionary_saturation =\
            amplitude_outlier_basic_function(consumption_outlier_input, total_consumption_values[valid_idx_saturation],
                                             1.5, inlier_deviation=1.5, column_number=1, offset=0.0)

        list_of_outliers_dates = np.r_[list_of_outliers_dates,
                                       return_dictionary_saturation['high_outliers']['quad'][0][:, dates_column],
                                       return_dictionary_saturation['high_outliers']['ransac'][0][:, dates_column]]

    abrupt_hvac_hours = {
        'final_outlier_days': list_of_outliers_dates,
        'hvac_consumption_matrix': total_consumption_matrix,
        'hvac_potential_matrix': hvac_potential_matrix,
        'return_dictionary_zero_saturation': return_dictionary_zero_saturation,
        'return_dictionary_saturation': return_dictionary_saturation,
        'return_dictionary': return_dictionary,
        'row': row_idx,
        'columns': column_idx
    }

    output_hvac_inefficiency_object[device]['net_ao_outlier'] = abrupt_hvac_hours
    return input_hvac_inefficiency_object, output_hvac_inefficiency_object


def get_cons_outlier(input_hvac_inefficiency_object, output_hvac_inefficiency_object, device):

    """
    Function to return consumption outliers

    Parameters:
        input_hvac_inefficiency_object  (dict)  : Dictionary containing inefficiency payload
        output_hvac_inefficiency_object (dict)  : Dictionary containing inefficiency payload output
        device                          (str)   : String identifying ac or sh

    Returns:
        input_hvac_inefficiency_object  (dict)  : Dictionary containing inefficiency payload
        output_hvac_inefficiency_object (dict)  : Dictionary containing inefficiency payload output
    """

    static_params = hvac_static_params()
    if device == 'ac':
        unconsidered_device = 'sh'
    else:
        unconsidered_device = 'ac'

    date_index = copy.deepcopy(input_hvac_inefficiency_object.get(device).get('energy_pivot').get('row'))
    total_consumption_matrix = copy.deepcopy(
        input_hvac_inefficiency_object.get(device).get('energy_pivot').get('values'))
    total_consumption_values = total_consumption_matrix.sum(axis=1)

    hvac_potential_matrix = input_hvac_inefficiency_object.get(device).get('{}_potential_pivot'.format(device)).get(
        'values')

    # Set limit to minimum hvac consumption
    hvac_consumption_matrix =\
        copy.deepcopy(input_hvac_inefficiency_object.get(device).get('demand_hvac_pivot').get('values'))

    hvac_temperature =\
        copy.deepcopy(input_hvac_inefficiency_object.get(device).get('temperature_pivot').get('values'))

    unconsidered_hvac_data_master =\
        copy.deepcopy(input_hvac_inefficiency_object.get(unconsidered_device).get('demand_hvac_pivot').get('values'))

    row_idx = input_hvac_inefficiency_object.get(device).get('{}_potential_pivot'.format(device)).get('row')
    column_idx = input_hvac_inefficiency_object.get(device).get('{}_potential_pivot'.format(device)).get('columns')

    list_of_outliers_dates = np.empty(0, )
    dates_column = 2
    # Counting number of hours with HVAC and HVAC potential
    hvac_hour_count = (hvac_consumption_matrix > 0).sum(axis=1)
    hvac_potential_hour_count = (hvac_potential_matrix > 0).sum(axis=1)
    hvac_temperature_avergae = np.nanmean(hvac_temperature, axis=1)
    unconsidered_hvac_count = (unconsidered_hvac_data_master > 0).sum(axis=1)

    # removing all days with NaN mean temperature
    nan_temperature_day = np.isnan(hvac_temperature_avergae)

    hvac_temperature_avergae = hvac_temperature_avergae[~nan_temperature_day]
    unconsidered_hvac_count = unconsidered_hvac_count[~nan_temperature_day]
    hvac_potential_hour_count = hvac_potential_hour_count[~nan_temperature_day]
    hvac_hour_count = hvac_hour_count[~nan_temperature_day]
    date_index = date_index[~nan_temperature_day]
    total_consumption_values = total_consumption_values[~nan_temperature_day]

    # Handling and managing  regions

    valid_idx = ((hvac_hour_count > 0) & (unconsidered_hvac_count == 0)
                 & (hvac_potential_hour_count > 0) & (hvac_potential_hour_count != Cgbdisagg.HRS_IN_DAY))

    valid_idx_zero_saturation = ((hvac_hour_count > 0) & (unconsidered_hvac_count == 0)
                                 & (hvac_potential_hour_count == 0))

    # Days with less than zero potential
    valid_idx_saturation = ((hvac_hour_count > 0) & (unconsidered_hvac_count == 0)
                            & (hvac_potential_hour_count == Cgbdisagg.HRS_IN_DAY))

    # count number of zero potential days

    count_zero_potential = np.sum(valid_idx_zero_saturation)
    count_saturation_potential = np.sum(valid_idx_saturation)
    count_potential = np.sum(valid_idx)

    if count_potential > static_params.get('ineff').get('count_potential_lim'):

        if count_saturation_potential < static_params.get('ineff').get('count_potential_lim'):
            valid_idx = (valid_idx | valid_idx_saturation)
            valid_idx_saturation = np.zeros_like(valid_idx_saturation, dtype=bool)

        if count_zero_potential < static_params.get('ineff').get('count_potential_lim'):
            valid_idx = (valid_idx | valid_idx_zero_saturation)
            valid_idx_zero_saturation = np.zeros_like(valid_idx_zero_saturation, dtype=bool)

    elif count_potential <= static_params.get('ineff').get('count_potential_lim'):
        valid_idx_zero_saturation = valid_idx_zero_saturation | valid_idx | valid_idx_saturation
        valid_idx = np.zeros_like(valid_idx, dtype=bool)
        valid_idx_saturation = np.zeros_like(valid_idx_saturation, dtype=bool)

    consumption_outlier_input = np.c_[hvac_potential_hour_count[valid_idx], hvac_temperature_avergae[valid_idx],
                                      date_index[valid_idx]]

    return_dictionary = dict({})

    if consumption_outlier_input.shape[0] > 1:
        return_dictionary = amplitude_outlier_basic_function(consumption_outlier_input,
                                                             total_consumption_values[valid_idx], 1.5,
                                                             inlier_deviation=1.5, column_number=0,
                                                             offset=0.0)

        list_of_outliers_dates = np.r_[list_of_outliers_dates,
                                       return_dictionary['high_outliers']['quad'][0][:, dates_column],
                                       return_dictionary['high_outliers']['ransac'][0][:, dates_column]]

    consumption_outlier_input = np.c_[hvac_potential_hour_count[valid_idx_zero_saturation],
                                      hvac_temperature_avergae[valid_idx_zero_saturation],
                                      date_index[valid_idx_zero_saturation]]

    return_dictionary_zero_saturation = dict({})

    if consumption_outlier_input.shape[0] > 1:
        return_dictionary_zero_saturation = amplitude_outlier_basic_function(consumption_outlier_input,
                                                                             total_consumption_values[valid_idx_zero_saturation],
                                                                             1.5, inlier_deviation=1.5,
                                                                             column_number=1, offset=0.0)
        list_of_outliers_dates = \
            np.r_[list_of_outliers_dates,
                  return_dictionary_zero_saturation['high_outliers']['quad'][0][:, dates_column],
                  return_dictionary_zero_saturation['high_outliers']['ransac'][0][:, dates_column]]

    consumption_outlier_input = np.c_[hvac_potential_hour_count[valid_idx_saturation],
                                      hvac_temperature_avergae[valid_idx_saturation], date_index[valid_idx_saturation]]

    return_dictionary_saturation = dict({})

    if consumption_outlier_input.shape[0] > 1:
        return_dictionary_saturation = amplitude_outlier_basic_function(consumption_outlier_input,
                                                                        total_consumption_values[valid_idx_saturation], 1.5,
                                                                        inlier_deviation=1.5, column_number=1,
                                                                        offset=0.0)

        list_of_outliers_dates = np.r_[list_of_outliers_dates,
                                       return_dictionary_saturation['high_outliers']['quad'][0][:, dates_column],
                                       return_dictionary_saturation['high_outliers']['ransac'][0][:, dates_column]]

    abrupt_hvac_hours = {
        'final_outlier_days': list_of_outliers_dates,
        'hvac_consumption_matrix': total_consumption_matrix,
        'hvac_potential_matrix': hvac_potential_matrix,
        'return_dictionary_zero_saturation': return_dictionary_zero_saturation,
        'return_dictionary_saturation': return_dictionary_saturation,
        'return_dictionary': return_dictionary,
        'row': row_idx,
        'columns': column_idx
    }
    output_hvac_inefficiency_object[device]['net_consumption_outlier'] = abrupt_hvac_hours

    return input_hvac_inefficiency_object, output_hvac_inefficiency_object
