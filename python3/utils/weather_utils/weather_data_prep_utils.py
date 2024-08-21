"""
Author - Sahana M
Date - 28/10/2021
Contains various utils for weather analytics input data preparation
"""


# Import python packages
import logging
import numpy as np
import pandas as pd

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.weather_utils.weather_config import Weatherconfigs


def flatten_json(nested_json, exclude=['']):
    """
    Flatten json object with nested keys into a single level.
    Parameters:
        nested_json         (json)              : A nested json object.
        exclude             (list)              : Keys to exclude from output.
    Returns:
        out                 (list)              : The flattened json object if successful, None otherwise.
    """

    out = {}

    def flatten(x, name='', exclude=exclude):
        """
        This function is used to flatten a nested json array
        Parameters:
            x       (list/dict)     : List or dictionary in a json object
        Returns:
            out     (array)         : Flattened array
        """
        if type(x) is dict:
            for a in x:
                if a not in exclude:
                    flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(nested_json)
    return out


def remove_columns(derived_weather_data):
    """
    This function is used to remove columns which are not required for the disagg pipeline
    Parameters:
        derived_weather_data       (dataframe)    : Contains derived weather data
    Returns:
        derived_weather_data       (dataframe)    : Contains derived weather data
    """

    # Get the columns currently present and columns to drop

    drop_columns = Weatherconfigs.DROP_COLUMNS
    current_columns_order = Weatherconfigs.INPUT_COLS_ORDER

    # Identify the column indexes to drop

    columns_to_remove = np.isin(current_columns_order, drop_columns)
    derived_weather_data = derived_weather_data[:, ~columns_to_remove]

    return derived_weather_data


def timestamp_mapping(input_data, derived_weather_data):
    """
    This function is used to map the timestamp corresponding derived data and the input raw data timestamps
    Parameters:
        input_data                      (np.ndarray)    : Contains the input consumption data
        derived_weather_data            (dataframe)     : Contains derived weather data
    Returns:
        refined_weather_data            (np.ndarray)    : Contains the refined and timestamp aligned weather data
    """

    weather_columns = derived_weather_data.columns

    # Identify the column indexes required for timestamp mapping

    timestamp_col = int(np.where(weather_columns == 'timestamp')[0])

    extrapolated_weather_data = np.full(shape=(input_data.shape[0], derived_weather_data.shape[1]), fill_value=np.nan)
    extrapolated_weather_data[:, timestamp_col] = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]

    # Find the input data epoch timestamp to derived weather data input mapping

    _, weather_data_idx, input_data_idx_map = np.intersect1d(derived_weather_data['timestamp'], input_data[:, Cgbdisagg.INPUT_EPOCH_IDX],
                                                             return_indices=True)

    extrapolated_weather_data[input_data_idx_map, :] = derived_weather_data.iloc[weather_data_idx, :]

    return extrapolated_weather_data


def combine_weather_columns(input_data, derived_weather_data):
    """
    This function combines and prepares the final input data containing the derived weather data
    Parameters:
        input_data                  (np.ndarray)        : Input data
        derived_weather_data        (np.ndarray)        : Derived weather data
    Returns:
        input_data                  (np.ndarray)        : Input data
    """

    input_data = np.c_[input_data, derived_weather_data]

    # Create final input data which has columns rearranged

    final_input_data = np.full_like(input_data, fill_value=np.nan)

    # Rearrange columns according to Cgbdisagg

    final_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] = input_data[:, Weatherconfigs.INPUT_BILL_CYCLE_IDX]
    final_input_data[:, Cgbdisagg.INPUT_WEEK_IDX] = input_data[:, Weatherconfigs.INPUT_WEEK_IDX]
    final_input_data[:, Cgbdisagg.INPUT_DAY_IDX] = input_data[:, Weatherconfigs.INPUT_DAY_IDX]
    final_input_data[:, Cgbdisagg.INPUT_DOW_IDX] = input_data[:, Weatherconfigs.INPUT_DOW_IDX]
    final_input_data[:, Cgbdisagg.INPUT_HOD_IDX] = input_data[:, Weatherconfigs.INPUT_HOD_IDX]
    final_input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] = input_data[:, Weatherconfigs.INPUT_EPOCH_IDX]
    final_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = input_data[:, Weatherconfigs.INPUT_CONSUMPTION_IDX]
    final_input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX] = input_data[:, Weatherconfigs.INPUT_TEMPERATURE_IDX]
    final_input_data[:, Cgbdisagg.INPUT_SKYCOV_IDX] = input_data[:, Weatherconfigs.INPUT_SKYCOV_IDX]
    final_input_data[:, Cgbdisagg.INPUT_WIND_SPD_IDX] = input_data[:, Weatherconfigs.INPUT_WIND_SPD_IDX]
    final_input_data[:, Cgbdisagg.INPUT_DEW_IDX] = input_data[:, Weatherconfigs.INPUT_DEW_IDX]
    final_input_data[:, Cgbdisagg.INPUT_SUNRISE_IDX] = input_data[:, Weatherconfigs.INPUT_SUNRISE_IDX]
    final_input_data[:, Cgbdisagg.INPUT_SUNSET_IDX] = input_data[:, Weatherconfigs.INPUT_SUNSET_IDX]
    final_input_data[:, Cgbdisagg.INPUT_FEELS_LIKE_IDX] = input_data[:, Weatherconfigs.INPUT_FEELS_LIKE_IDX]
    final_input_data[:, Cgbdisagg.INPUT_PREC_IDX] = input_data[:, Weatherconfigs.INPUT_PREC_IDX]
    final_input_data[:, Cgbdisagg.INPUT_SNOW_IDX] = input_data[:, Weatherconfigs.INPUT_SNOW_IDX]
    final_input_data[:, Cgbdisagg.INPUT_SL_PRESS_IDX] = input_data[:, Weatherconfigs.INPUT_SL_PRESS_IDX]
    final_input_data[:, Cgbdisagg.INPUT_SPC_HUM_IDX] = input_data[:, Weatherconfigs.INPUT_SPC_HUM_IDX]
    final_input_data[:, Cgbdisagg.INPUT_REL_HUM_IDX] = input_data[:, Weatherconfigs.INPUT_REL_HUM_IDX]
    final_input_data[:, Cgbdisagg.INPUT_WET_BULB_IDX] = input_data[:, Weatherconfigs.INPUT_WET_BULB_IDX]
    final_input_data[:, Cgbdisagg.INPUT_WIND_DIR_IDX] = input_data[:, Weatherconfigs.INPUT_WIND_DIR_IDX]
    final_input_data[:, Cgbdisagg.INPUT_VISIBILITY_IDX] = input_data[:, Weatherconfigs.INPUT_VISIBILITY_IDX]
    final_input_data[:, Cgbdisagg.INPUT_COOLING_POTENTIAL_IDX] = input_data[:, Weatherconfigs.INPUT_COOLING_POTENTIAL_IDX]
    final_input_data[:, Cgbdisagg.INPUT_HEATING_POTENTIAL_IDX] = input_data[:, Weatherconfigs.INPUT_HEATING_POTENTIAL_IDX]
    final_input_data[:, Cgbdisagg.INPUT_WH_POTENTIAL_IDX] = input_data[:, Weatherconfigs.INPUT_WH_POTENTIAL_IDX]
    final_input_data[:, Cgbdisagg.INPUT_COLD_EVENT_IDX] = input_data[:, Weatherconfigs.INPUT_COLD_EVENT_IDX]
    final_input_data[:, Cgbdisagg.INPUT_HOT_EVENT_IDX] = input_data[:, Weatherconfigs.INPUT_HOT_EVENT_IDX]
    final_input_data[:, Cgbdisagg.INPUT_S_LABEL_IDX] = input_data[:, Weatherconfigs.INPUT_S_LABEL_IDX]

    return final_input_data


def combine_raw_data_and_weather_data(input_data, weather_analytics_data, logger_pass):
    """
    Function used to combine the input data and weather derived data into a single array
    Parameters:
        input_data              (dict)          : Input data
        weather_analytics_data  (dict)          : Weather data
        logger_pass             (Logger)        : Logger

    Returns:
        input_data              (dict)          : Input data
    """

    # Initialise logger

    logger_base = logger_pass.get('logger_base').getChild('combine_raw_data_and_weather_data')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Flatten the json to a 2D array

    derived_weather_data = pd.DataFrame([flatten_json(x) for x in weather_analytics_data])
    logger.info("Flattening of Derived weather JSON data into 2D matrix completed | ")

    # Drop zipcode column

    derived_weather_data = derived_weather_data.drop(columns=['zipcode'])

    # Order the weather columns accordingly

    derived_weather_data = derived_weather_data[Weatherconfigs.INPUT_COLS_ORDER]

    # Perform timestamp mapping and expansion

    derived_weather_data = timestamp_mapping(input_data, derived_weather_data)
    logger.info("Timestamp mapping of Consumption epoch and Weather epoch completed | ")

    # Remove unnecessary columns

    derived_weather_data = remove_columns(derived_weather_data)
    logger.info("Removing  unnecessary columns from the Derived weather data | ")

    # Combined the extrapolated weather data to input data

    input_data = combine_weather_columns(input_data, derived_weather_data)
    logger.info("Final mapping of all the input columns completed | ")

    return input_data
