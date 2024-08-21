"""
Author  -   Anand Kumar Singh
Date    -   22th Feb 2021
This file contains code for preparing data for abrupt change in HVAC consumption
"""
import copy
import logging
import datetime
import numpy as np

# Import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff
from python3.utils.maths_utils.maths_utils import create_pivot_table
from python3.analytics.hvac_inefficiency.utils.downsample_n_col_data import downsample_n_col_data


def prep_hvac_potential(input_hvac_inefficiency_object, logger_pass):
    """
        Prepare HVAC data for temporal changes

        Parameters:
            input_hvac_inefficiency_object      (dict)       : dictionary containing all input the information
            logger_pass                         (logger)     : logger object
        Returns:
            cool_pot_pivot                      (np.ndarray) : Array of cooling potential in 2D
            heat_pot_pivot                      (np.ndarray) : Array of cooling potential in 2D
            s_label_pivot                       (np.ndarray) : Array of cooling potential in 2D
    """

    # Prepare logger
    logger_local = logger_pass.get("logger").getChild("prep_hvac_potential")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}

    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    prepare_hvac_data_for_temporal_change_time = datetime.datetime.now()

    # Preprering in input data and column indices for accessing columns

    input_data = copy.deepcopy(input_hvac_inefficiency_object.get('raw_input_values'))
    raw_input_idx = copy.deepcopy(input_hvac_inefficiency_object.get('raw_input_idx'))

    # Preparing list of generic column indices
    epoch_day_idx = raw_input_idx.get('day_idx')
    epoch_column = raw_input_idx.get('epoch_data')
    hour_of_day = raw_input_idx.get('hour_of_day')
    cooling_potential_col = raw_input_idx.get('cooling_potential')
    heating_potential_col = raw_input_idx.get('heating_potential')
    s_label_col = raw_input_idx.get('season_label')

    # cooling potential pivot
    temp_array = input_data[:, [epoch_column, cooling_potential_col, hour_of_day, epoch_day_idx]]

    temp_array = np.nan_to_num(temp_array)

    new_cool_pot_column = 1
    new_hour_of_day_column = 2
    new_date_column = 3

    hour_bc_columns = [new_hour_of_day_column, new_date_column]

    hourly_data = downsample_n_col_data(temp_array, target_rate=3600, hour_bc_columns=hour_bc_columns)

    cool_pot_pivot, row, col = \
        create_pivot_table(hourly_data, index=new_date_column, columns=new_hour_of_day_column,
                           values=new_cool_pot_column)

    invalid_cool_pot = (cool_pot_pivot < (-1000)) | (cool_pot_pivot > 1000)
    cool_pot_pivot[invalid_cool_pot] = np.nan

    # heating potential pivot
    temp_array = input_data[:, [epoch_column, heating_potential_col, hour_of_day, epoch_day_idx]]

    temp_array = np.nan_to_num(temp_array)

    new_heat_pot_column = 1
    new_hour_of_day_column = 2
    new_date_column = 3

    hour_bc_columns = [new_hour_of_day_column, new_date_column]

    hourly_data = downsample_n_col_data(temp_array, target_rate=3600, hour_bc_columns=hour_bc_columns)

    heat_pot_pivot, row, col = \
        create_pivot_table(hourly_data, index=new_date_column, columns=new_hour_of_day_column,
                           values=new_heat_pot_column)

    invalid_heat_pot = (heat_pot_pivot < (-1000)) | (heat_pot_pivot > 1000)
    heat_pot_pivot[invalid_heat_pot] = np.nan

    # season marking
    temp_array = input_data[:, [epoch_column, s_label_col, hour_of_day, epoch_day_idx]]
    temp_array = np.nan_to_num(temp_array)
    temp_array[:, 1][temp_array[:, 1] == 0.5] = 0
    temp_array[:, 1][temp_array[:, 1] == -0.5] = 0

    new_s_label_column = 1
    new_hour_of_day_column = 2
    new_date_column = 3

    hour_bc_columns = [new_hour_of_day_column, new_date_column]

    hourly_data = downsample_n_col_data(temp_array, target_rate=3600, hour_bc_columns=hour_bc_columns)

    s_label_pivot, row, col = \
        create_pivot_table(hourly_data, index=new_date_column, columns=new_hour_of_day_column,
                           values=new_s_label_column)

    invalid_s_label = (s_label_pivot < (-1000)) | (s_label_pivot > 1000)
    s_label_pivot[invalid_s_label] = np.nan

    time_taken = get_time_diff(prepare_hvac_data_for_temporal_change_time, datetime.datetime.now())
    logger.debug('Time taken for correcting HVAC clusters | {}'.format(time_taken))

    return cool_pot_pivot, heat_pot_pivot, s_label_pivot


def prep_hvac_for_temporal_change(input_hvac_inefficiency_object, output_hvac_inefficiency_object,
                                  logger_pass, device):
    """
        Prepare HVAC data for temporal changes

        Parameters:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            output_hvac_inefficiency_object     (dict)          dictionary containing all output information
            logger_pass                         (object)        logger object
            device                              (str)           string indicating device, either AC or SH
        Returns:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            output_hvac_inefficiency_object     (dict)          dictionary containing all output information
    """

    logger_local = logger_pass.get("logger").getChild("prepare_hvac_data_for_temporal_change")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}

    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    prepare_hvac_data_for_temporal_change_time = datetime.datetime.now()

    # Preprering in input data and column indices for accessing columns

    input_data = input_hvac_inefficiency_object.get('raw_input_values')
    ao_hvac_column = input_hvac_inefficiency_object.get(device).get('ao_column')
    demand_hvac_column = input_hvac_inefficiency_object.get(device).get('demand_column')
    raw_input_idx = input_hvac_inefficiency_object.get('raw_input_idx')

    # Preparing list of generic column indices
    epoch_day_idx = raw_input_idx.get('day_idx')
    epoch_column = raw_input_idx.get('epoch_data')
    hour_of_day = raw_input_idx.get('hour_of_day')
    net_consumption = raw_input_idx.get('consumption')

    temp_array = input_data[:, [epoch_column,  net_consumption, ao_hvac_column, demand_hvac_column, hour_of_day, epoch_day_idx]]

    temp_array = temp_array.astype(np.int)

    new_net_consumption_column = 1
    new_ao_hvac_consumption = 2
    new_demand_hvac_consumption = 3
    new_hour_of_day_column = 4
    new_date_column = 5

    hour_bc_columns = [new_hour_of_day_column, new_date_column]

    hourly_data = downsample_n_col_data(temp_array, target_rate=3600, hour_bc_columns=hour_bc_columns)

    ao_hvac_consumption_pivot, row, col = \
        create_pivot_table(hourly_data, index=new_date_column, columns=new_hour_of_day_column,
                           values=new_ao_hvac_consumption)

    input_hvac_inefficiency_object[device]['ao_hvac_pivot'] = {
        'values': ao_hvac_consumption_pivot,
        'row': row,
        'col': col
    }

    demand_hvac_consumption_pivot, row, col = \
        create_pivot_table(hourly_data, index=new_date_column, columns=new_hour_of_day_column,
                           values=new_demand_hvac_consumption)

    input_hvac_inefficiency_object[device]['demand_hvac_pivot'] = {
        'values': demand_hvac_consumption_pivot,
        'row': row,
        'col': col
    }

    net_energy_consumption_pivot, row, col = \
        create_pivot_table(hourly_data, index=new_date_column, columns=new_hour_of_day_column,
                           values=new_net_consumption_column)

    input_hvac_inefficiency_object[device]['energy_pivot'] = {
        'values' : net_energy_consumption_pivot,
        'row': row,
        'col': col
    }

    input_hvac_inefficiency_object[device]['hourly_data'] = hourly_data

    time_taken = get_time_diff(prepare_hvac_data_for_temporal_change_time, datetime.datetime.now())
    logger.debug('Time taken for preparing HVAC data clusters | {} | {}'.format(device, time_taken))

    return input_hvac_inefficiency_object, output_hvac_inefficiency_object


def prep_weather_for_temporal_change(input_hvac_inefficiency_object, output_hvac_inefficiency_object,
                                     logger_pass):
    """
        Prepare HVAC data for temporal changes

        Parameters:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            output_hvac_inefficiency_object     (dict)          dictionary containing all output information
            logger_pass                         (object)        logger object
        Returns:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            output_hvac_inefficiency_object     (dict)          dictionary containing all output information
    """

    # Prepare logger
    logger_local = logger_pass.get("logger").getChild("prepare_temperature_data_for_temporal_change")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}

    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    prepare_hvac_data_for_temporal_change_time = datetime.datetime.now()

    # Preprering in input data and column indices for accessing columns

    input_data = copy.deepcopy(input_hvac_inefficiency_object.get('raw_input_values'))
    raw_input_idx = copy.deepcopy(input_hvac_inefficiency_object.get('raw_input_idx'))
    cooling_setpoint = copy.deepcopy(input_hvac_inefficiency_object.get('ac', dict({})).get('setpoint', np.nan))
    heating_setpoint = copy.deepcopy(input_hvac_inefficiency_object.get('sh', dict({})).get('setpoint', np.nan))

    # Preparing list of generic column indices
    epoch_day_idx = raw_input_idx.get('day_idx')
    epoch_column = raw_input_idx.get('epoch_data')
    hour_of_day = raw_input_idx.get('hour_of_day')
    temperature_column = raw_input_idx.get('temperature')

    temp_array = input_data[:, [epoch_column,  temperature_column, hour_of_day, epoch_day_idx]]

    temp_array = temp_array.astype(np.int)

    new_temperature_column = 1
    new_hour_of_day_column = 2
    new_date_column = 3

    hour_bc_columns = [new_hour_of_day_column, new_date_column]

    hourly_data = downsample_n_col_data(temp_array, target_rate=3600, hour_bc_columns=hour_bc_columns)

    temperature_pivot, row, col =\
        create_pivot_table(hourly_data, index=new_date_column, columns=new_hour_of_day_column,
                           values=new_temperature_column)

    invalid_temperature = (temperature_pivot < (-1000)) | (temperature_pivot > 1000)
    temperature_pivot[invalid_temperature] = np.nan

    # Storing HVAC based temperature value
    input_hvac_inefficiency_object['ac']['temperature_pivot'] = {
        'values': (temperature_pivot - cooling_setpoint),
        'row': row,
        'col': col
    }

    input_hvac_inefficiency_object['sh']['temperature_pivot'] = {
        'values': -1 * (temperature_pivot - heating_setpoint),
        'row': row,
        'col': col
    }

    time_taken = get_time_diff(prepare_hvac_data_for_temporal_change_time, datetime.datetime.now())
    logger.debug('Time taken for correcting HVAC clusters | {}'.format(time_taken))

    return input_hvac_inefficiency_object, output_hvac_inefficiency_object
