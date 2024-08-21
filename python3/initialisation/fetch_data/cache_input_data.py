"""
Author - Mayank Sharan
Date - 14/01/19
Caches input data intelligently
"""

# Import python packages

import os
import pickle
import numpy as np

# Import functions from within the project

from python3.config.path_constants import PathConstants


def cache_input_data(input_data, fetch_params, pipeline_run_data, weather_analytics_data, home_meta_data, app_profile,
                     hsm_appliances):

    """
    Parameters:
        input_data              (np.ndarray)        : 21 column input data
        fetch_params            (dict)              : Contains parameters for the data fetch
        pipeline_run_data       (dict)              : Contains info about the pipeline run
        weather_analytics_data  (dict)              : Contains weather data input
        home_meta_data          (dict)              : Contains data about the home
        app_profile             (dict)              : Contains data about the user appliances
        hsm_appliances          (dict)              : Contains data about the HSMs

    Returns:
        cache_status            (string)            : Returns how the cache operation went
    """

    # Get user id to cache data for

    uuid = fetch_params.get('uuid')

    # Create cache directory if it does not exist

    if not os.path.exists(PathConstants.CACHE_DIR):
        os.mkdir(PathConstants.CACHE_DIR)

    user_data_path = PathConstants.CACHE_DIR + '/data_' + uuid + '.pb'

    if not os.path.exists(user_data_path):

        # No cache currently exists for the user
        # Prepare data dictionary for the user

        data_dict = {
            'input_data': input_data,
            'app_profile': app_profile,
            'home_meta_data': home_meta_data,
            'hsm_appliances': hsm_appliances,
            'disagg_run_data': pipeline_run_data,
            'weather_analytics_data': weather_analytics_data,
        }

        # Save the data

        with open(user_data_path, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return 'Data cached successfully'

    elif fetch_params.get('cache_update'):

        # Delete previously existing cache
        os.remove(user_data_path)

        # Prepare data dictionary for the user

        data_dict = {
            'input_data': input_data,
            'app_profile': app_profile,
            'home_meta_data': home_meta_data,
            'hsm_appliances': hsm_appliances,
            'disagg_run_data': pipeline_run_data,
            'weather_analytics_data': weather_analytics_data,
        }

        # Save the data

        with open(user_data_path, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Cache exists for the user but we want to update
        return 'Data cache updated successfully'

    else:

        # Cache exists for the user and we do not want to update
        return 'Data cache exists and wasn\'t updated'
