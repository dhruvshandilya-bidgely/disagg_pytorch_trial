"""
Author - Sahana M
Date - 2/3/2021
Initialising data required for Weather data module to run
"""

# Import python packages

from copy import deepcopy
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_weather_data(input_data):

    """
    Returns the required weather data columns from the 21 column input data matrix

    Parameters:
        input_data              (np.ndarray)        : The 21 column input data matrix

    Returns:
        weather_data            (dict)              : Dictionary containing weather data columns

    """

    data = deepcopy(input_data)

    # Get all the column indexes required

    cols = [Cgbdisagg.INPUT_EPOCH_IDX, Cgbdisagg.INPUT_DAY_IDX, Cgbdisagg.INPUT_TEMPERATURE_IDX,
            Cgbdisagg.INPUT_FEELS_LIKE_IDX, Cgbdisagg.INPUT_PREC_IDX, Cgbdisagg.INPUT_SNOW_IDX]

    # Extract the above columns into an array

    output_data = data[:, cols]

    # Initialise a dictionary to return the weather data

    weather_data = dict()
    weather_data['weather'] = {}
    weather_data['weather']['raw_weather'] = output_data

    return weather_data


def get_meta_data(weather_data, wh_config):

    """
    Returns the required meta data from the wh_config

    Parameters:
        weather_data            (dict)              : Dictionary containing weather data
        wh_config               (dict)              : Dictionary containing meta data of the user

    Returns:
        weather_data            (dict)              : Dictionary containing weather data along with meta data

    """

    # Get the current time

    current_time = int(datetime.now().timestamp())

    # Get all the required meta data from wh_config

    weather_data['meta_data'] = {}
    weather_data['meta_data']['uuid'] = wh_config.get('uuid')
    weather_data['meta_data']['pilot_id'] = wh_config.get('pilot_id')
    weather_data['meta_data']['ownership_type'] = wh_config['home_meta_data'].get('ownershipType', 'N/A')
    weather_data['meta_data']['timezone'] = wh_config['home_meta_data'].get('timezone', 'UTC')
    weather_data['meta_data']['property_type'] = wh_config['home_meta_data'].get('dwdelling')
    weather_data['meta_data']['property_area'] = wh_config.get('livingArea', 'N/A')
    weather_data['meta_data']['city'] = wh_config.get('city', 'N/A')
    weather_data['meta_data']['country'] = wh_config.get('country', 'N/A')
    weather_data['meta_data']['current_ts'] = current_time

    return weather_data
