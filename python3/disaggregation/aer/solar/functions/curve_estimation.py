"""
Author - Anand Kumar Singh
Date - 14th Feb 2020
This file has code to estimate normalised solar generation curve

"""

# Import python packages

import logging
import datetime
import numpy as np
from pytz import timezone
from pysolar import solar as pys

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff


def get_irradiance(epoch_time, latitude, longitude, user_timezone, logger_pass):

    """
        Get Irradiance for all epochs timestamps

        Parameters:
            epoch_time              (np.ndarray)        : numpy array containing epoch_timestamp
            latitude                (float)             : user's latitude
            longitude               (float)             : user's longitude
            user_timezone           (dict)              : user's timezone

        Returns:
            irradiance              (np.ndarray)        : numpy array containing irradiance for each input timestamp
    """

    # Taking new logger base for this module
    logger_local = logger_pass.get("logger").getChild("get_irradiance")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    logger.info('Calculating irradiance for each data point')

    benchmark_time = datetime.datetime.now()

    logger.debug('Convert epoch to timezone aware datetime datetime |')
    date = [datetime.datetime.fromtimestamp(x, timezone(user_timezone)) for x in epoch_time]
    date = np.array(date)
    benchmark_time = get_time_diff(benchmark_time, datetime.datetime.now())
    logger.debug('Timing: Total time for creating datetime array datetime | %0.3f', benchmark_time)

    benchmark_time = datetime.datetime.now()

    logger.debug('Get altitude degree from lat, long and date |')
    altitude_deg = pys.get_altitude_fast(latitude, longitude, date)
    benchmark_time = get_time_diff(benchmark_time, datetime.datetime.now())
    logger.debug('Timing: Total time for calculating altitude degree required for irradiance |  %0.3f', benchmark_time)

    benchmark_time = datetime.datetime.now()
    irradiance = [pys.radiation.get_radiation_direct(x[0], x[1]) for x in np.c_[date, altitude_deg]]
    benchmark_time = get_time_diff(benchmark_time, datetime.datetime.now())
    logger.debug('Timing: Total time calculating irradiance from altitude and datetime | %0.3f', benchmark_time)

    logger.info('Finished computing irradiance for all data points |')
    return np.array(irradiance)


def min_max_normalisation(data_array, logger_pass, min_quantile=0, max_quantile=1, capping=True, min_max_dict=None):

    """
        Normalise feature vector using min max normalisation

        Parameters:
            data_array              (np.ndarray)        : numpy array containing feature data
            logger_pass             (object)            : Logger object
            min_quantile            (float)             : float value between 0 and 1. Will be used to compute lower limit
            max_quantile            (float)             : float value between 0 and 1. Will be used to compute upper limit
            capping                 (Bool)              : Boolean value, will be used to decide if cap outliers.
            min_max_dict            (dict)              : Dictionary containing pre define min and max for normalisation

        Returns:
            data_array              (np.ndarray)        : normalised feature vector
            upper_value             (float)             : upper value for normalisation
            lower_value             (float)             : upper value for normalisation
    """

    logger_local = logger_pass.get("logger").getChild("min_max_normalisation")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    logger.debug('Computing non nan indices in data |')
    valid_idx = ~np.isnan(data_array)

    logger.debug('Get min and max for  normalisation from input dictionary |')
    if not min_max_dict.get('upper'):
        logger.debug('Computing upper value to be treated as max |')
        upper_value = np.nanquantile(data_array[valid_idx], max_quantile)
    else:
        logger.debug('Found upper value in dict, not computing quantile |')
        upper_value = min_max_dict.get('upper')

    logger.debug('Computing min and max for normalisation |')
    if not min_max_dict.get('lower'):
        logger.debug('Computing lower value to be treated as min |')
        lower_value = np.nanquantile(data_array[valid_idx], min_quantile)
    else:
        logger.debug('Found lower value in dict, not computing quantile |')
        lower_value = min_max_dict.get('lower')
    logger.info('Normalisation data using min and max | %0.3f, %0.3f', lower_value, upper_value)
    data_array[valid_idx] = (upper_value - data_array[valid_idx]) / (upper_value - lower_value)

    logger.debug('Capping higher values to 1 and lower to 0 |')
    if capping:
        logger.debug('Capping upper and lower values to 1 and 0 respectively |')
        high_value_idx = data_array > 1
        data_array[high_value_idx] = 1
        low_value_idx = data_array < 0
        data_array[low_value_idx] = 0
    return data_array, lower_value, upper_value


def estimate_normalised_curve(input_data, solar_config, irradiance, logger_pass):

    """
        Estimation solar potentials for each data point

        Parameters:
            input_data              (np.ndarray)        : numpy array containing 21 column input matrix
            solar_config            (dict)              : solar config dict
            logger_pass             (object)            : Logger object

        Returns:
            input_data              (np.ndarray)        : numpy array containing 21 column input matrix with solar potentials
            input_features          (np.ndarray)        : input  feature vector used for prediction
            curve_hsm               (dict)              : solar potential hsm
            debug_dictionary        (dict)              : debug dictionary for additional information

    """

    # Taking new logger base for this module

    # Debug dictionary
    debug_dictionary = {}

    logger_local = logger_pass.get("logger").getChild("estimate_normalised_curve")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    logger.info('Loading solar curve estimation model |')
    grid = solar_config.get('estimation_model')
    logger.debug('Successfully loaded curve estimation model |')

    logger.debug('Calculating day light indices for further slicing of data |')
    sun_time_idx = (input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] > input_data[:, Cgbdisagg.INPUT_SUNRISE_IDX]) & \
                   (input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] < input_data[:, Cgbdisagg.INPUT_SUNSET_IDX])


    logger.debug('Calculating index for data points in out_bill_cycles |')
    out_bill_cycles = solar_config.get('out_bill_cycles')
    if type(out_bill_cycles) == np.ndarray:
        out_bill_cycles_idx = np.isin(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], out_bill_cycles)
    else:
        out_bill_cycles_idx = np.ones_like(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX], dtype=bool)

    logger.info('Computing initial features for curve estimation |')
    logger.debug('Computing total sun time on the day |')
    total_sun_time = input_data[:, Cgbdisagg.INPUT_SUNSET_IDX] - input_data[:, Cgbdisagg.INPUT_SUNRISE_IDX]
    logger.debug('Computing distance from sunrise for each data point |')
    distance_from_sunrise = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] - input_data[:, Cgbdisagg.INPUT_SUNRISE_IDX]

    # Update input data array with additional features
    list_of_input_data_features = solar_config.get('features_from_input_data')
    input_features = np.c_[
        input_data[:, list_of_input_data_features[:-1]], distance_from_sunrise, irradiance, total_sun_time]

    debug_dictionary['input_features'] = input_features.tolist()

    logger.info('Normalising feature before giving to XGB model |')

    # min, max normalisation for temperature, sky_cover and wind
    normalisation_threshold = solar_config.get('normalisation_thresholds')
    max_irradiance = solar_config.get('normalisation_params').get('max_irradiance')
    upper_percentile = solar_config.get('normalisation_params').get('upper_percentile')
    lower_percentile = solar_config.get('normalisation_params').get('lower_percentile')
    temp_column = 0
    sky_column = 1
    wind_column = 2
    sunrise_column = 3
    irradiance_column = 4
    total_sun_time_column = 5
    consumption_column = 6

    # subtract nightload from consumption
    new_consumption = np.zeros(input_data.shape[0])
    for day in np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX]):
        day_idx = input_data[:, Cgbdisagg.INPUT_DAY_IDX] == day
        valid_idx = (~sun_time_idx) & (day_idx)
        day_nightload = np.nanquantile(input_data[valid_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX], .10)
        new_consumption[day_idx] = input_data[day_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] - day_nightload

    # add new consumption to input features
    input_features = np.c_[input_features, new_consumption]

    # normalising distance from sunrise by total sun time
    input_features[:, sunrise_column] = input_features[:, sunrise_column] / input_features[:, total_sun_time_column]

    # normalising sun time duration with hours in day
    input_features[:, total_sun_time_column] = input_features[:, total_sun_time_column] / Cgbdisagg.SEC_IN_DAY

    # normalising irradiance with max values being 1000
    input_features[:, irradiance_column] = input_features[:, irradiance_column] / max_irradiance

    # Initialising min max for normalisation in case of MTD run
    min_max_dict_temp = normalisation_threshold.get('temp', {})
    min_max_dict_wind = normalisation_threshold.get('wind', {})
    min_max_dict_sky = normalisation_threshold.get('sky_cover', {})
    min_max_dict_consumption = normalisation_threshold.get('consumption', {})

    # normalising weather features using min max

    input_features[:, temp_column], lower_temp, upper_temp = \
        min_max_normalisation(input_features[:, temp_column], logger_pass, min_quantile=lower_percentile,
                              max_quantile=upper_percentile, min_max_dict=min_max_dict_temp)

    input_features[:, wind_column], lower_wind, upper_wind = \
        min_max_normalisation(input_features[:, wind_column], logger_pass, min_quantile=lower_percentile,
                              max_quantile=upper_percentile, min_max_dict=min_max_dict_wind)

    input_features[:, sky_column], lower_sky, upper_sky = \
        min_max_normalisation(input_features[:, sky_column],logger_pass,min_quantile=lower_percentile,
                              max_quantile=upper_percentile,min_max_dict=min_max_dict_sky)

    input_features[:, consumption_column], lower_cons, upper_cons = \
        min_max_normalisation(input_features[:, consumption_column], logger_pass, min_quantile=lower_percentile,
                              max_quantile=upper_percentile, min_max_dict=min_max_dict_consumption)

    logger.debug('Calculating solar potential for daylight hours |')

    logger.debug('Creating 0 array for solar potential |')
    solar_potential = np.zeros_like(input_data[:, 0])

    valid_predict_index = sun_time_idx & out_bill_cycles_idx


    logger.info('Predicting solar potentials for each data point |')
    benchmark_time = datetime.datetime.now()
    solar_potential[valid_predict_index] = grid.predict(input_features[valid_predict_index, :])
    benchmark_time = get_time_diff(benchmark_time, datetime.datetime.now())
    logger.debug('Timing: Total time taken for predicting all solar potentials is %0.3f', benchmark_time)

    logger.debug('Creating curve hsm for the user |')
    neg_points = np.round((solar_potential < 0).sum() / sun_time_idx.shape[0], 3)

    # Create HSM dictionary
    curve_hsm = {'curve_min_estimate': solar_potential.min(),
                 'curve_max_estimate': solar_potential.max(),
                 'curve_neg_points_fraction': neg_points,
                 'temp_lower': lower_temp,
                 'temp_upper': upper_temp,
                 'sky_cover_lower': lower_sky,
                 'sky_cover_upper': upper_sky,
                 'wind_lower': lower_wind,
                 'wind_upper': upper_wind
                }

    debug_dictionary['curve_hsm'] = curve_hsm

    logger.debug('Curve hsm \t %s', curve_hsm)

    # Set Negative solar potential to 0
    logger.debug('Setting negative solar potential to 0')
    neg_solar_potential = solar_potential < 0

    logger.debug('Total negative points count | %d', neg_solar_potential.sum())
    logger.info('Capping negative solar consumption to zero |')
    solar_potential[neg_solar_potential] = -solar_potential[neg_solar_potential]

    # Adding solar potential column to input data array
    logger.debug('Adding solar potential array to input data array |')
    input_data = np.c_[input_data, solar_potential]

    debug_dictionary['solar_potential'] = solar_potential
    logger.debug('Returning input data array |')
    return input_data, input_features, curve_hsm, debug_dictionary
