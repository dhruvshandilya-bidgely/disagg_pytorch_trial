"""
Author - Anand Kumar Singh
Date - 14th Feb 2020
This file has code for estimate solar capacity

"""

# Import python packages

import math
import logging
import datetime
import numpy as np
from sklearn.metrics import r2_score
from copy import deepcopy
from scipy import signal
# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff


def moving_average(array, length):
    """
    This function calculates moving average for a given array and length

        Parameters:
        input_data             (numpy.ndarray)     numpy array
        length                 (int)               length of moving average

        Return:
        output                 (numpy.ndarray)     numpy array of moving average
    """
    return np.convolve(array, np.ones(length), 'valid') / length


def valid_day_check(input_data, date, date_idx, valid_day_array, sun_time_idx, solar_potential_column, solar_config, logger_pass):
    """
    This function checks for valid solar generation days

    Parameters:
        input_data             (numpy.ndarray)     numpy array containing 21 column matrix and solar potential
        date                   (int)               date index
        date_idx               (numpy.ndarray)     numpy array boolean for present day
        valid_day_array        (numpy.ndarray)     numpy array containing valid days for estimating solar
        sun_time_idx           (numpy.ndarray)     numpy array containing indices of suntime data points
        solar_potential_column (int)               column index for solar potential column
        solar_config           (dict)              solar config dict
        logger_pass            (object)            logger object

    Returns
        day_sun_idx            (numpy.ndarray)     numpy array containing indices of suntime data points
        day_data               (numpy.ndarray)     numpy array containing 21 column matrix and solar potential
        daily_stats            (numpy.ndarray)     stats for peak points and depth
        max_potential          (float)             max potential for the day
        std_dev_daily          (float)             standard deviation of consumption for the day
    """

    # Taking new logger base for this module
    logger_local = logger_pass.get("logger").getChild("fit_capacity_regression_lower_bound")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    day_sun_idx = sun_time_idx[date_idx]
    day_data = input_data[date_idx, :]
    daily_stats = [0, 2 ** 15]

    # Checking of given day is type 2 vacation
    std_dev_daily = np.nanstd(input_data[date_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    max_potential = np.nanmax(day_data[:, solar_potential_column])

    if std_dev_daily == 0:
        valid_day_array[date_idx] = 0
        logger.debug('Std deviation 0 for consumption, skipping date | %s', datetime.datetime.fromtimestamp(date))

    if max_potential <= 0:
        logger.debug('No solar potential found on this day, skipping date | %s',
                     datetime.datetime.fromtimestamp(date))

    daily_data = np.array(day_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    sampling_rate = int(Cgbdisagg.SEC_IN_HOUR/ solar_config.get('sampling_rate'))

    for i in range(2*sampling_rate):
        daily_data = daily_data[:-1] - (daily_data[:-1] - daily_data[1:]).clip(min=0)

    week_factor = Cgbdisagg.DAYS_IN_WEEK - 1
    if len(daily_data) > week_factor*sampling_rate:
        daily_data = daily_data - [np.nanquantile(daily_data[:6*sampling_rate], .1)] * len(daily_data)
        peaks = signal.find_peaks(-daily_data)[0]
        peaks = [i for i in peaks if day_sun_idx[i]]
        if len(peaks) > 0:

            prominences = signal.peak_prominences(-daily_data, peaks)[0]
            max_prominence = max(prominences)
            peak_depth = max_prominence
            value_at_peak = day_data[peaks[int(np.where(prominences == max_prominence)[0])], Cgbdisagg.INPUT_CONSUMPTION_IDX]
            daily_stats = [peak_depth, value_at_peak]

    return day_sun_idx, day_data, daily_stats, max_potential, std_dev_daily


def fit_capacity_regression(input_data, date, valid_day_array, sun_time_idx, solar_potential_column, solar_config, logger_pass):

    """
    This function fits regression  to estimate capacity

            Parameters:
                input_data                 (numpy.ndarray)     numpy array containing 21 column matrix and solar potential
                date                       (int)               date index
                valid_day_array            (numpy.ndarray)     numpy array containing valid days for estimating solar
                sun_time_idx               (numpy.ndarray)     numpy array containing indices of suntime data points
                solar_potential_column     (int)               column index for solar potential column
                solar_config               (dict)              solar config dict
                logger_pass                (object)            logger object

            Returns:
                predicted_cap_lower         (float)             estimated lower capacity of the solar panel
                predicted_cap_upper         (float)             estimated upper capacity of the solar panel
                r_square_lower              (float)             lower r squared value for regression line
                r_square_upper              (float)             upper r squared value for regression line
                valid_day_array             (numpy.ndarray)     numpy array containing valid days for estimating solar
                consumption_no_nightload    (numpy.ndarray)     numpy array containing consumption values without nightload
                daily_stats                 (numpy.ndarray)     stats for peak points and depth
                night_baseload              (numpy.ndarray)     numpy array containing nightload

    """

    # Taking new logger base for this module
    logger_local = logger_pass.get("logger").getChild("fit_capacity_regression_lower_bound")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    # Initialising 0 for predicted capacity
    r_square_lower = 0
    predicted_cap_lower = 0
    r_square_upper = 0
    predicted_cap_upper = 0
    daily_stats = [0, 2**15]

    # Set baseload difference threshold
    baseload_threshold = solar_config.get('baseload_threshold', {})
    baseload_difference_threshold = baseload_threshold * solar_config.get('sampling_rate') / Cgbdisagg.SEC_IN_HOUR
    date_idx = input_data[:, Cgbdisagg.INPUT_DAY_IDX] == date
    consumption_no_nightload = input_data[date_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    night_baseload = 0
    try:
        day_data = input_data[date_idx, :]
        day_sun_idx = sun_time_idx[date_idx]

        day_sun_idx, day_data, daily_stats, max_potential, std_dev_daily = valid_day_check(
            input_data, date, date_idx, valid_day_array, sun_time_idx, solar_potential_column, solar_config, logger_pass)

        if (std_dev_daily == 0) or (max_potential <= 0):
            valid_day_array[date_idx] = 0
            predicted_cap_lower = np.nan
            return predicted_cap_lower, r_square_lower, predicted_cap_upper, r_square_upper, valid_day_array, \
                   input_data[date_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX], daily_stats, night_baseload

        # Compute night baseload
        night_baseload = np.nanquantile(day_data[~day_sun_idx][:, Cgbdisagg.INPUT_CONSUMPTION_IDX], 0.1)

        # Find consistency in consumption before sunrise and after sunrise
        baseload_duration_sec = 2 * Cgbdisagg.SEC_IN_HOUR
        left_baseload_idx = \
            (day_data[:, Cgbdisagg.INPUT_EPOCH_IDX] > (day_data[:, Cgbdisagg.INPUT_SUNRISE_IDX] - baseload_duration_sec)) & \
            (day_data[:, Cgbdisagg.INPUT_EPOCH_IDX] < day_data[:, Cgbdisagg.INPUT_SUNRISE_IDX])

        right_baseload_idx = \
            (day_data[:, Cgbdisagg.INPUT_EPOCH_IDX] < (day_data[:, Cgbdisagg.INPUT_SUNSET_IDX] + baseload_duration_sec)) & \
            (day_data[:, Cgbdisagg.INPUT_EPOCH_IDX] > day_data[:, Cgbdisagg.INPUT_SUNSET_IDX])

        left_baseload = np.nanmin(day_data[left_baseload_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        right_baseload = np.nanmin(day_data[right_baseload_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX])

        baseload_difference = np.abs(right_baseload - left_baseload)

        # Subtracting nightload from consumption
        day_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] -= night_baseload
        minimum_day_consumption = np.nanmin(day_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

        # Saving consumption without nightload
        consumption_no_nightload = day_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

        # Restricting data for sun time hours
        day_data = day_data[day_sun_idx]


        if minimum_day_consumption >= 0:
            valid_day_array[date_idx] = 0
            logger.debug('Minimum daily consumption for the day is non negative, skipping date | %s',
                         datetime.datetime.fromtimestamp(date))
            return predicted_cap_lower, r_square_lower, predicted_cap_upper, r_square_upper, valid_day_array, consumption_no_nightload, daily_stats, night_baseload

        distance_from_sunrise = \
            (day_data[:, Cgbdisagg.INPUT_EPOCH_IDX] - day_data[:, Cgbdisagg.INPUT_SUNRISE_IDX]) / \
            (day_data[:, Cgbdisagg.INPUT_SUNSET_IDX] - day_data[:, Cgbdisagg.INPUT_SUNRISE_IDX])

        x = np.sin(math.pi * distance_from_sunrise)
        y = -1 * day_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
        data = np.c_[x, y]
        data = data[~np.isnan(data).any(axis=1)]

        if (data.shape[0] < 5) or (baseload_difference > baseload_difference_threshold):
            logger.debug('%s data points available after filtering, and baseload_difference , skipping date for lower '
                         'capacity estimation | %s, %s',
                         str(data.shape[0]), str(baseload_difference), datetime.datetime.fromtimestamp(date))
            return predicted_cap_lower, r_square_lower, predicted_cap_upper, r_square_upper, valid_day_array, consumption_no_nightload, daily_stats, night_baseload

        # Data checks passed, fitting regression
        z = np.polyfit(data[:, 0], data[:, 1], 1)

        r_square = r2_score(y, z[0] * x + z[1])
        slope, intercept = z[0], z[1]
        r_square_lower = np.round(r_square, 2)

        # Find lower capacity for each day
        predicted_cap_lower = np.round((slope * 1 + intercept), 2)
        logger.debug('date: %s, predicted capacity_lower:%0.3f and r_square_lower | %0.2f',
                     datetime.datetime.fromtimestamp(date), predicted_cap_lower, r_square_lower)

        #Max Bucket Method
        data = np.array(sorted(np.c_[x, y, np.round(20 * x) / 20], key=lambda x: x[1], reverse=True))

        #Taking Points only above lower bound
        data = data[data[:, 1] > (data[:, 0] * slope + intercept)]

        #Creating unique buckets
        _, indices = np.unique(data[:, 2], return_index=True)
        data = data[indices]
        data = data[~np.isnan(data).any(axis=1)]

        # Data checks passed, fitting regression
        z = np.polyfit(data[:, 0], data[:, 1], 1)
        r_square = r2_score(y, z[0] * x + z[1])
        slope, intercept = z[0], z[1]
        r_square_upper = np.round(r_square, 2)

        # Find upper capacity for each day
        predicted_cap_upper = np.round((slope * 1 + intercept), 2)
        logger.debug('date: %s, predicted capacity_upper:%0.3f and r_square_upper | %0.2f',
                     datetime.datetime.fromtimestamp(date), predicted_cap_upper, r_square_upper)

    except Exception as e:
        logger.debug('date: %s, failed to fit regression curve. Error: %s |', datetime.datetime.fromtimestamp(date), str(e))

    return predicted_cap_lower, r_square_lower, predicted_cap_upper, r_square_upper, valid_day_array, consumption_no_nightload, daily_stats, night_baseload


def find_capacity_over_time(capacity_array, r_squared_threshold, base_capacity, solar_config, logger_pass, number_of_good_days=14):
    """
    This function finds capacity for all valid days

        Parameters:
            capacity_array             (numpy.ndarray)     numpy array containing 21 column matrix and solar potential
            r_squared_threshold        (float)             r_squared threshold for good days
            solar_config               (dict)              solar config dict
            logger_pass                (object)            logger object
            number_of_good_days        (int)               batch size fosr minimum number of days to compute capacity

        Returns:
            predicted_capacity_for_day (float)             Chunk level estimated capacity of the solar panel
            enough_good_days           (boolean)           Presence of enough good data days
    """

    # Taking new logger base for this module
    capacity_array = deepcopy(capacity_array[:,:4])
    logger_local = logger_pass.get("logger").getChild("find_capacity_over_time")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    logger.info('Calculating sun time index')
    benchmark_time = datetime.datetime.now()

    date_column = 0
    capacity_lower_column = 1
    daily_capacity_column = 1
    r_squared_lower_column = 2
    capacity_upper_column = 3

    enough_good_days = True

    # Find good days based on r squared values
    good_days = capacity_array[:, r_squared_lower_column] >= r_squared_threshold

    predicted_capacity_for_day = np.zeros_like(capacity_array[:, date_column])
    predicted_capacity_for_day = np.c_[capacity_array[:, date_column], predicted_capacity_for_day]

    if good_days.sum() < number_of_good_days:
        enough_good_days = False

    # Loading capacity params : coefficients for lower/upper capacity estimates and upper capacity limit
    alpha = solar_config.get('capacity_params').get('lower_capacity_coefficient')
    beta = solar_config.get('capacity_params').get('upper_capacity_coefficient')
    upper_limit_for_capacity = solar_config.get('capacity_params').get('upper_limit_for_capacity')

    lower_bound = np.max([np.nanquantile(capacity_array[good_days, capacity_lower_column], upper_limit_for_capacity), base_capacity])
    upper_bound = np.max([np.nanquantile(capacity_array[good_days, capacity_upper_column], upper_limit_for_capacity), base_capacity])

    predicted_capacity_for_day[:, daily_capacity_column] = alpha * lower_bound + beta * upper_bound

    benchmark_time = get_time_diff(benchmark_time, datetime.datetime.now())
    logger.debug('Timing:: Finished computing capacity in sliding window | %0.3f', benchmark_time)

    return predicted_capacity_for_day, enough_good_days


def estimate_solar_capacity_neg_cons(input_data, solar_config, detection_hsm, logger_pass):
    """
    This functions estimates capacity using negative consumption and solar potential

            Parameters:
                input_data         (numpy.ndarray)     numpy array with 21 column matrix and solar potential
                solar_config       (dict)              solar config dict
                detection_hsm      (list)              solar detection hsm
                logger_pass        (object)            logger object

            Returns:
                effective_capacity      (numpy.ndarray)     estimated capacity of the solar panel
                capacity_hsm            (dict)              dict containing capacity information
                valid_day_array         (numpy.ndarray)     array containing valid days
                predicted_capacity_array(numpy.ndarray)     array containing capacity for respective days

    """

    # Taking new logger base for this module
    logger_local = logger_pass.get("logger").getChild("estimate_solar_capacity_neg_cons")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    logger.info('Calculating sun time index |')

    unique_dates = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX])
    sun_time_idx = (input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] > input_data[:, Cgbdisagg.INPUT_SUNRISE_IDX]) & \
                   (input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] < input_data[:, Cgbdisagg.INPUT_SUNSET_IDX])

    logger.info('Calculating capacity for each day |')
    benchmark_time = datetime.datetime.now()

    # Also adding code to remove days with vacation type 2, creating array with all ones
    valid_day_array = np.ones_like(input_data[:, Cgbdisagg.INPUT_DAY_IDX])
    solar_potential_column = solar_config.get('solar_potential_column')

    predicted_capacity_array = []
    consumption_without_baseload = []
    valid_day_stats = []
    nightload_array = []

    # Find capacity for each day
    for date in unique_dates:
        predicted_cap_lower, r_square_lower, predicted_cap_upper, r_square_upper, valid_day_array, day_consumption_without_baseload, daily_stats, nightload = \
            fit_capacity_regression(input_data, date, valid_day_array,
                                    sun_time_idx, solar_potential_column, solar_config, logger_pass)

        # Creating numpy array for each day to avoid merger issues
        predicted_capacity_array.append(
            [date, predicted_cap_lower, r_square_lower, predicted_cap_upper, r_square_upper])
        consumption_without_baseload.append(day_consumption_without_baseload)
        valid_day_stats.append(daily_stats)
        nightload_array.append([nightload]*day_consumption_without_baseload.shape[0])

    nightload_array = sum(nightload_array, [])
    base_capacity = -np.nanquantile(np.concatenate(consumption_without_baseload), .01)
    predicted_capacity_array = np.array(predicted_capacity_array)
    benchmark_time = get_time_diff(benchmark_time, datetime.datetime.now())
    logger.info('Timing:: Finished capacity calculation | %0.3f', benchmark_time)

    start_date = detection_hsm.get('attributes', {}).get('start_date', {})
    end_date = detection_hsm.get('attributes', {}).get('end_date', {})

    panel_status = detection_hsm.get('attributes', {}).get('kind', {})
    moving_average_factor = Cgbdisagg.DAYS_IN_WEEK - 1

    # Valid day logic implementation
    if panel_status:
        if panel_status == 1:
            valid_date_arr = 1 * (input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] >= start_date)
        else:
            valid_date_arr = 1 * (input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] <= end_date)
        valid_day_stats = np.array(valid_day_stats)
        mask_solar_moving_avg = moving_average(valid_day_stats[:, 0] > base_capacity / 3, Cgbdisagg.DAYS_IN_WEEK) > .5
        mask_solar_moving_avg = np.append(np.array([mask_solar_moving_avg[0]] * moving_average_factor), mask_solar_moving_avg)
        mask_solar_negative_points = np.array((valid_day_stats[:, 1] < 0))
        mask_solar_valid_day = np.logical_or(mask_solar_moving_avg, mask_solar_negative_points)
        #Map back unique values to input data
        _, indices = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)
        mask_solar_valid_day = 1 * np.array(mask_solar_valid_day)[indices]
        valid_day_array = valid_day_array * mask_solar_valid_day * valid_date_arr

    # Initialising column names
    date_column = 0
    capacity_lower_column = 1
    daily_capacity_column = 1
    r_squared_lower_column = 2
    capacity_upper_column = 3
    r_squared_upper_column = 4

    # Updating  predicted capacity array with previous values

    input_day_start = input_data[:, Cgbdisagg.INPUT_DAY_IDX].min()

    predicted_capacity_array_idx = ~np.isnan(predicted_capacity_array[:, capacity_lower_column])
    predicted_capacity_array = predicted_capacity_array[predicted_capacity_array_idx, :]

    residual_capacity_array = solar_config.get('residual_capacity_array')

    if type(residual_capacity_array) == np.ndarray:
        predicted_capacity_array = np.r_[residual_capacity_array, predicted_capacity_array]

    # Discarding information older than disagg period.
    predicted_capacity_array = predicted_capacity_array[predicted_capacity_array[:, date_column] >= input_day_start, :]

    r_squared_threshold = solar_config.get('neg_cap_estimation').get('r_square_threshold').get('hi_r_square')

    # Create capacity array similar shape to input column
    daily_capacity, enough_days_status = \
        find_capacity_over_time(predicted_capacity_array, r_squared_threshold, base_capacity, solar_config, logger_pass)

    if not enough_days_status:
        r_squared_threshold = solar_config.get('neg_cap_estimation').get('r_square_threshold').get('mid_r_square')
        daily_capacity, enough_days_status = \
            find_capacity_over_time(predicted_capacity_array, r_squared_threshold, base_capacity, solar_config, logger_pass)

    if not enough_days_status:
        r_squared_threshold = solar_config.get('neg_cap_estimation').get('r_square_threshold').get('low_r_square')
        daily_capacity, enough_days_status = \
            find_capacity_over_time(predicted_capacity_array, r_squared_threshold, base_capacity, solar_config, logger_pass)

    if not enough_days_status:

        # Loading capacity params : coefficients for lower/upper capacity estimates and upper capacity limit
        alpha = solar_config.get('capacity_params').get('lower_capacity_coefficient')
        beta = solar_config.get('capacity_params').get('upper_capacity_coefficient')
        upper_limit_for_capacity = solar_config.get('capacity_params').get('upper_limit_for_capacity')

        non_zero_rows = predicted_capacity_array[:, capacity_lower_column] > 0

        lower_bound = np.max([np.nanquantile(predicted_capacity_array[non_zero_rows, capacity_lower_column], upper_limit_for_capacity), base_capacity])
        upper_bound = np.max([np.nanquantile(predicted_capacity_array[non_zero_rows, capacity_upper_column], upper_limit_for_capacity), base_capacity])

        daily_capacity[:, daily_capacity_column] = alpha * lower_bound + beta * upper_bound

    predicted_cap_neg_consumption = daily_capacity[-1, daily_capacity_column]
    daily_capacity = np.full((input_data.shape[0]), predicted_cap_neg_consumption)
    effective_capacity = daily_capacity
    benchmark_time = datetime.datetime.now()

    benchmark_time = get_time_diff(benchmark_time, datetime.datetime.now())
    logger.info('Timing:: Finished merging date index | %0.3f', benchmark_time)

    # Prepare capacity hsm
    hsm_idx = predicted_capacity_array[:, r_squared_lower_column] > r_squared_threshold
    residual_capacity_lower = predicted_capacity_array[hsm_idx, capacity_lower_column].tolist()
    residual_good_days = predicted_capacity_array[hsm_idx, date_column].tolist()
    residual_r_squared_lower = predicted_capacity_array[hsm_idx, r_squared_lower_column].tolist()
    residual_capacity_upper = predicted_capacity_array[hsm_idx, capacity_upper_column].tolist()
    residual_r_squared_upper = predicted_capacity_array[hsm_idx, r_squared_upper_column].tolist()

    # Create capcity HSM
    if r_squared_threshold >= solar_config.get('neg_cap_estimation').get('r_square_threshold').get('low_r_square'):
        capacity_hsm = {'capacity': predicted_cap_neg_consumption,
                        'residual_capacity_lower': residual_capacity_lower,
                        'residual_good_days': residual_good_days,
                        'residual_r_squared_lower': residual_r_squared_lower,
                        'residual_capacity_upper': residual_capacity_upper,
                        'residual_r_squared_upper': residual_r_squared_upper,
                        'r_squared_threshold': r_squared_threshold}
    else:
        capacity_hsm = {'capacity': predicted_cap_neg_consumption,
                        'r_squared_threshold': r_squared_threshold}

    return effective_capacity, capacity_hsm, valid_day_array, predicted_capacity_array, nightload_array
