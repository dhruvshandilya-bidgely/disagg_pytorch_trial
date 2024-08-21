"""
Author - Sahana M
Date - 24/4/2021
Performs data sanity checks
"""

# Import python packages
import numpy as np


def data_sanity_check(wh_potential, cooling_potential, swh_data_matrix, debug):
    """
    This function checks data sanity and cleans the data
    Parameters:
        wh_potential                (np.array)          : Numpy array containing Water heater potential
        cooling_potential           (nd.array)          : 2D array containing the cooling potential
        swh_data_matrix             (nd.array)          : 2D array containing the cleaned input data
        debug                       (dict)              : Dictionary containing all the necessary keys

    Returns:
        wh_potential                (np.array)          : Numpy array containing Water heater potential
        cooling_potential           (nd.array)          : 2D array containing the cooling potential
        debug                       (dict)              : Dictionary containing all the necessary keys
    """

    # Check the difference in the number of days between wh potential & input data

    if len(wh_potential) > len(swh_data_matrix):

        # Extract the weather data day time stamps and input data day time stamps
        # Note : There would be extra days present in the weather data day time stamps & they should be removed

        weather_data_day_ts = debug['weather_data_output']['weather']['day_wise_data']['day_ts']
        input_data_day_ts = debug['weather_data_output']['weather']['raw_weather'][:, 1]
        feels_like_temperature = debug['weather_data_output']['weather']['day_wise_data']['fl']

        # Get the common days time stamps

        non_extra_days = np.intersect1d(weather_data_day_ts, input_data_day_ts)
        non_extra_days_indexes = np.where(np.in1d(weather_data_day_ts, non_extra_days))[0]

        # Extract only the common days time stamps and update the following arrays

        wh_potential = wh_potential[non_extra_days_indexes]
        cooling_potential = cooling_potential[non_extra_days_indexes]
        feels_like_temperature = feels_like_temperature[non_extra_days_indexes]

        # Update the debug dict

        debug['wh_potential'] = wh_potential
        debug['cooling_potential'] = cooling_potential
        debug['non_extra_days'] = non_extra_days
        debug['fl'] = feels_like_temperature

    if len(wh_potential) < len(swh_data_matrix):

        # Extract the weather data day time stamps and input data day time stamps
        # Note : There would be less days present in the weather data day time stamps & they should be interpolated

        weather_data_day_ts = debug['weather_data_output']['weather']['day_wise_data']['day_ts']
        input_data_day_ts = debug['weather_data_output']['weather']['raw_weather'][:, 1]
        feels_like_temperature = debug['weather_data_output']['weather']['day_wise_data']['fl']

        # Get the days to be filled

        input_unique_days = np.unique(input_data_day_ts)
        days_to_fill = np.full(shape=input_unique_days.shape[0], fill_value=False)

        # Identify the missing days

        for i in range(len(input_unique_days)):

            if input_unique_days[i] not in weather_data_day_ts:
                days_to_fill[i] = True

        new_wh_potential = np.full(shape=(len(input_unique_days)), fill_value=0.0)
        new_cooling_potential = np.full(shape=(len(input_unique_days), cooling_potential.shape[1]), fill_value=np.nan)
        new_feels_like_temperature = np.full(shape=(len(input_unique_days), cooling_potential.shape[1]),
                                             fill_value=np.nan)

        # For each missing day, interpolate the data of the previous day

        for i in range(len(days_to_fill)):

            if not days_to_fill[i]:
                new_wh_potential[i] = wh_potential[i]
                new_cooling_potential[i] = cooling_potential[i]
                new_feels_like_temperature[i] = feels_like_temperature[i]
            else:
                new_wh_potential[i] = wh_potential[i-1]
                new_cooling_potential[i] = cooling_potential[i-1]
                new_feels_like_temperature[i] = feels_like_temperature[i-1]

        # Extract only the common days time stamps and update the following arrays

        wh_potential = new_wh_potential
        cooling_potential = new_cooling_potential
        feels_like_temperature = new_feels_like_temperature

        # Update the debug dict

        debug['wh_potential'] = wh_potential
        debug['cooling_potential'] = cooling_potential
        debug['fl'] = feels_like_temperature

    return wh_potential, cooling_potential, debug
