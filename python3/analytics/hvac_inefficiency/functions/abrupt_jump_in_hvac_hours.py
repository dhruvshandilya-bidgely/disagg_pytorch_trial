"""
Author  -   Anand Kumar Singh
Date    -   22th Feb 2021
This file contains code for finding abrupt change in HVAC
"""

# Import python packages

import copy
import logging
import numpy as np

from python3.config.Cgbdisagg import Cgbdisagg
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params
from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile as super_percentile


def forward_fill(arr):
    """ Utility function to forward fill NaN values"""

    mask = np.array([(np.isnan(arr))]).reshape(-1)
    if len(mask) > 1:
        if mask[0]:
            arr[0] = 0
            mask[0] = False
        idx = np.where(~mask, np.arange(mask.shape[0]), 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        arr[mask] = arr[idx[mask]]

    return arr


def outlier_basic_function(main_input_hvac, main_output_hvac, hvac_outlier_limit, inlier_deviation=2, column_number=0,
                           hour_gap=2, random_state=43):
    """
        This function is a basic outlier detection function

        Parameters:
            main_input_hvac             (numpy.ndarray)         array containing input variables
            main_output_hvac            (numpy.ndarray)         array containing output variables
            hvac_outlier_limit          (float)                 limit defining allowed deviation
            inlier_deviation            (float)                 limits that define outliers
            column_number               (int)                   column of input array to fit ransac
            hour_gap                    (int)                   offset
            random_state                (int)                   random state for repeatability
        Returns:
            return_dictionary  (dict)              dictionary containing all input the information
    """

    static_params = hvac_static_params()

    # Imputing main_input_hvac to remove NaN values
    main_input_hvac[:, column_number] = forward_fill(main_input_hvac[:, column_number].squeeze())

    # Removing 2nd and 4th quadrant outliers

    lower_limit_x_axis = super_percentile(main_input_hvac[:, column_number], static_params.get('ineff').get('low_quant') * 100)
    higher_limit_x_axis = super_percentile(main_input_hvac[:, column_number], static_params.get('ineff').get('high_quant') * 100)

    lower_limit_y_axis = super_percentile(main_output_hvac, static_params.get('ineff').get('low_quant') * 100)
    higher_limit_y_axis = super_percentile(main_output_hvac, static_params.get('ineff').get('high_quant') * 100)

    x_delimiter = (lower_limit_x_axis + higher_limit_x_axis) / 2
    y_delimiter = (lower_limit_y_axis + higher_limit_y_axis) / 2

    left_top_quadrant = (main_input_hvac[:, column_number] < x_delimiter) & (main_output_hvac > y_delimiter)
    right_bottom_quadrant = (main_input_hvac[:, column_number] > x_delimiter) & (main_output_hvac < y_delimiter)

    # Mark left top and right bottom as non outliers, everything else as outlier

    non_outliers = ~(left_top_quadrant | right_bottom_quadrant)

    if (non_outliers.shape[0] == non_outliers.sum()) or (non_outliers.sum() == 0):
        left_top_quadrant = np.full_like(left_top_quadrant, False)
        right_bottom_quadrant = np.full_like(right_bottom_quadrant, False)
        non_outliers = np.full_like(non_outliers, True)

    # create ransac input and output based on non outliers only

    ransac_input = main_input_hvac[non_outliers, :]
    ransac_output = main_output_hvac[non_outliers]

    # Create deviation for residual_threshold

    median_abs_deviation = (np.median(np.abs(ransac_output - np.median(ransac_output))))

    # Setting up a regressor

    residual_threshold = median_abs_deviation * hvac_outlier_limit
    reg = RANSACRegressor(LinearRegression(), random_state=random_state, residual_threshold=residual_threshold)

    # Fit RANSAC
    try:
        reg.fit(ransac_input[:, column_number].reshape(-1, 1), ransac_output)
    except:
        return_dictionary = {'inliers': [main_input_hvac, main_output_hvac],
                             'high_outliers': {'quad': [np.empty(shape=[0, 2]), np.empty(shape=[0]), np.empty(shape=[0])],
                                               'ransac': [np.empty(shape=[0, 2]), np.empty(shape=[0]), np.empty(shape=[0])]},
                             'low_outliers': {'quad': [np.empty(shape=[0, 2]), np.empty(shape=[0])],
                                              'ransac': [np.empty(shape=[0, 2]), np.empty(shape=[0])]},
                             'trendline': [np.empty(shape=[0, 2]), np.empty(shape=[0])]}

        return return_dictionary

    # Get prediction for each input

    predicted_values_hvac = reg.predict(ransac_input[:, column_number].reshape(-1, 1)).reshape(1, -1)[0]

    # RANSAC outliers

    ransac_outlier = ~reg.inlier_mask_

    # Quadrant outlier analysis

    prediction_main_output_hvac = reg.predict(main_input_hvac[:, column_number].reshape(-1, 1))

    # Creating distribution of differences

    absolute_difference = np.abs(prediction_main_output_hvac[non_outliers] - main_output_hvac[non_outliers])

    std_deviation = np.std(absolute_difference)
    median_distance = np.median(absolute_difference)

    difference_threshold_high = median_distance + (inlier_deviation * std_deviation) + hour_gap
    difference_threshold_low = (-1 * median_distance) - (inlier_deviation * std_deviation) - hour_gap

    # Dealing with left top quadrant

    left_quadrant_outlier_distance =\
        (prediction_main_output_hvac[left_top_quadrant] - main_output_hvac[left_top_quadrant])

    left_quadrant_high_outlier = left_quadrant_outlier_distance < difference_threshold_low

    # Dealing with right bottom quadrant
    right_quadrant_outlier_distance =\
        (prediction_main_output_hvac[right_bottom_quadrant] - main_output_hvac[right_bottom_quadrant])

    right_quadrant_low_outlier = right_quadrant_outlier_distance > difference_threshold_high

    # Mapping High Outliers

    more_hvac_usage = (ransac_output.reshape(1, -1)[0][ransac_outlier] - predicted_values_hvac[ransac_outlier]) > 0

    # Mapping low outliers

    less_hvac_usage = (ransac_output.reshape(1, -1)[0][ransac_outlier] - predicted_values_hvac[ransac_outlier]) < 0

    # Preparing inliers

    input_inliers = np.concatenate((ransac_input[~ransac_outlier, :],
                                    main_input_hvac[left_top_quadrant][~left_quadrant_high_outlier, :],
                                    main_input_hvac[right_bottom_quadrant][~right_quadrant_low_outlier, :]),
                                   axis=0)

    output_inliers = np.concatenate((ransac_output[~ransac_outlier].reshape(1, -1)[0],
                                     main_output_hvac[left_top_quadrant][~left_quadrant_high_outlier],
                                     main_output_hvac[right_bottom_quadrant][~right_quadrant_low_outlier]),
                                    axis=0)

    # Preparing Quad Outliers

    input_quad_outliers_high = main_input_hvac[left_top_quadrant][left_quadrant_high_outlier, :]
    output_quad_outliers_high = main_output_hvac[left_top_quadrant][left_quadrant_high_outlier]

    input_quad_outliers_low = main_input_hvac[right_bottom_quadrant][right_quadrant_low_outlier, :]
    output_quad_outliers_low = main_output_hvac[right_bottom_quadrant][right_quadrant_low_outlier]

    # Correcting RANSAC high outlier

    input_ransac_outliers_high = ransac_input[ransac_outlier, :][more_hvac_usage]
    output_ransac_outliers_high = ransac_output[ransac_outlier][more_hvac_usage].reshape(1, -1)[0]

    if input_ransac_outliers_high.shape[0] > 1:

        # Compute distance from prediction and readjust RANSAC Outliers

        predicted_output_outlier_high = reg.predict(input_ransac_outliers_high[:, column_number].reshape(-1, 1))

        predicted_output_outlier_high_difference = (predicted_output_outlier_high - output_ransac_outliers_high)
        rearranged_high_ransac_outlier = (predicted_output_outlier_high_difference < difference_threshold_low)

        input_inliers = np.r_[input_inliers, input_ransac_outliers_high[~rearranged_high_ransac_outlier]]
        output_inliers = np.r_[output_inliers, output_ransac_outliers_high[~rearranged_high_ransac_outlier]]

        input_ransac_outliers_high = input_ransac_outliers_high[rearranged_high_ransac_outlier]
        output_ransac_outliers_high = output_ransac_outliers_high[rearranged_high_ransac_outlier]

    # Correcting RANSAC low outlier

    input_ransac_outliers_low = ransac_input[ransac_outlier, :][less_hvac_usage]
    output_ransac_outliers_low = ransac_output[ransac_outlier][less_hvac_usage].reshape(1, -1)[0]

    if input_ransac_outliers_low.shape[0] > 0:

        # Compute distance from prediction and readjust RANSAC Outliers

        predicted_output_outlier_low = reg.predict(input_ransac_outliers_low[:, column_number].reshape(-1, 1))

        predicted_output_outlier_low_difference = (predicted_output_outlier_low - output_ransac_outliers_low)
        rearranged_low_ransac_outlier = (predicted_output_outlier_low_difference > difference_threshold_high)

        input_inliers = np.r_[input_inliers, input_ransac_outliers_low[~rearranged_low_ransac_outlier]]
        output_inliers = np.r_[output_inliers, output_ransac_outliers_low[~rearranged_low_ransac_outlier]]

        input_ransac_outliers_low = input_ransac_outliers_low[rearranged_low_ransac_outlier]
        output_ransac_outliers_low = output_ransac_outliers_low[rearranged_low_ransac_outlier]

    # Removing Left top and right bottom quadrant

    lower_limit_x_axis = super_percentile(input_inliers[:, column_number], static_params.get('ineff').get('low_quant') * 100)
    higher_limit_x_axis = super_percentile(input_inliers[:, column_number], static_params.get('ineff').get('high_quant') * 100)
    x_delimiter = (lower_limit_x_axis + higher_limit_x_axis) / 2

    lower_limit_y_axis = super_percentile(output_inliers, static_params.get('ineff').get('low_quant') * 100)
    higher_limit_y_axis = super_percentile(output_inliers, static_params.get('ineff').get('high_quant') * 100)
    y_delimiter = (lower_limit_y_axis + higher_limit_y_axis) / 2

    left_top_quadrant = (input_inliers[:, column_number] < x_delimiter) & (output_inliers > y_delimiter)
    right_bottom_quadrant = (input_inliers[:, column_number] > x_delimiter) & (output_inliers < y_delimiter)
    non_outliers = ~(left_top_quadrant | right_bottom_quadrant)

    # Create deviation for residual_threshold

    median_abs_deviation = (np.median(np.abs(output_inliers - np.median(output_inliers))))

    # Setting up regressor
    residual_threshold = median_abs_deviation * hvac_outlier_limit
    reg_retrain = RANSACRegressor(LinearRegression(), random_state=random_state, residual_threshold=residual_threshold)

    return_dictionary_default = {'regressor': [reg, None],
                                 'inliers': [main_input_hvac, main_output_hvac],
                                 'high_outliers': {'quad': [np.empty(shape=[0, 2]), np.empty(shape=[0]), np.empty(shape=[0])],
                                                   'ransac': [np.empty(shape=[0, 2]), np.empty(shape=[0]), np.empty(shape=[0])]},
                                 'low_outliers': {'quad': [np.empty(shape=[0, 2]), np.empty(shape=[0])],
                                                  'ransac': [np.empty(shape=[0, 2]), np.empty(shape=[0])]},
                                 'trendline': [np.empty(shape=[0, 2]), np.empty(shape=[0])]}

    return_default_flag = False

    if input_inliers[non_outliers, column_number].reshape(-1, 1).shape[0] <= 3:

        # Count of inliers is less than 3, return
        return_default_flag = True

    try:

        reg_retrain.fit(input_inliers[non_outliers, column_number].reshape(-1, 1), output_inliers[non_outliers])

    except:

        return_default_flag = True

    if return_default_flag == True:

        return return_dictionary_default

    # Recompute difference threshold, Get prediction for each retraining input

    prediction_main_output_hvac = reg_retrain.predict(input_inliers[:, column_number].reshape(-1, 1))

    # Creating distribution of differences

    absolute_difference = np.abs(prediction_main_output_hvac - output_inliers)

    std_deviation = np.std(absolute_difference)
    median_distance = np.median(absolute_difference)

    difference_threshold_low = (-1 * median_distance) - (inlier_deviation * std_deviation) - hour_gap

    # Final Adjustment of High outliers

    output_quad_outliers_high_score = np.empty(shape=[0])
    output_ransac_outliers_high_score = np.empty(shape=[0])

    # Adjusting RANSAC outlier

    if input_ransac_outliers_high.shape[0] > 0:

        # Compute distance from prediction and readjust RANSAC Outliers

        predicted_output_outlier_high = reg_retrain.predict(input_ransac_outliers_high[:, column_number].reshape(-1, 1))

        predicted_output_outlier_high_difference = (predicted_output_outlier_high - output_ransac_outliers_high)
        rearranged_high_ransac_outlier = (predicted_output_outlier_high_difference < difference_threshold_low)

        input_inliers = np.r_[input_inliers, input_ransac_outliers_high[~rearranged_high_ransac_outlier]]
        output_inliers = np.r_[output_inliers, output_ransac_outliers_high[~rearranged_high_ransac_outlier]]

        input_ransac_outliers_high = input_ransac_outliers_high[rearranged_high_ransac_outlier]
        output_ransac_outliers_high = output_ransac_outliers_high[rearranged_high_ransac_outlier]

        # Creating scores for all outliers

        output_ransac_outliers_high_score =\
            (predicted_output_outlier_high_difference[rearranged_high_ransac_outlier] / difference_threshold_low) - 1

    # Adjusting Quad outlier

    if input_quad_outliers_high.shape[0] > 0:

        # Compute distance from prediction and readjust RANSAC Outliers

        predicted_output_outlier_high = reg_retrain.predict(input_quad_outliers_high[:, column_number].reshape(-1, 1))

        predicted_output_outlier_high_difference = (predicted_output_outlier_high - output_quad_outliers_high)
        rearranged_high_quad_outlier = (predicted_output_outlier_high_difference < difference_threshold_low)

        input_inliers = np.r_[input_inliers, input_quad_outliers_high[~rearranged_high_quad_outlier]]
        output_inliers = np.r_[output_inliers, output_quad_outliers_high[~rearranged_high_quad_outlier]]

        input_quad_outliers_high = input_quad_outliers_high[rearranged_high_quad_outlier]
        output_quad_outliers_high = output_quad_outliers_high[rearranged_high_quad_outlier]

        # Creating scores for all outliers

        output_quad_outliers_high_score =\
            (predicted_output_outlier_high_difference[rearranged_high_quad_outlier] / difference_threshold_low) - 1

    # Preparing Trendline
    dummy_input = np.arange(1, Cgbdisagg.HRS_IN_DAY, 0.1)
    dummy_output = reg_retrain.predict(dummy_input.reshape(-1, 1))

    # Updating output dictionary
    return_dictionary = {'regressor': [reg, reg_retrain],
                         'inliers': [input_inliers, output_inliers],
                         'high_outliers': {'quad': [input_quad_outliers_high, output_quad_outliers_high,
                                                    output_quad_outliers_high_score],
                                           'ransac': [input_ransac_outliers_high, output_ransac_outliers_high,
                                                      output_ransac_outliers_high_score]},
                         'low_outliers': {'quad': [input_quad_outliers_low, output_quad_outliers_low],
                                          'ransac': [input_ransac_outliers_low, output_ransac_outliers_low]},
                         'trendline': [dummy_input, dummy_output]}

    return return_dictionary


def get_unconsidered_device(device):

    """
    Function to get unconsidered device

    Parameters:
        device  (str)   : HVAC identifier
    Returns:
        unconsidered_device (str) : Unconsidered HVAC device
    """

    if device == 'ac':

        unconsidered_device = 'sh'
    else:

        unconsidered_device = 'ac'

    return unconsidered_device


def get_final_outlier_days(outliers, final_outlier_days, outlier_days_column_idx):

    """
    Function to get final outlier days

    Parameters:
        outliers                    (dict)          : Outliers
        final_outlier_days          (np.ndarray)    : Final outlier days
        outlier_days_column_idx     (int)           : Outlier index

    Returns:
        final_outlier_days          (np.ndarray)    : Final outlier days
    """

    if outliers['quad'][0].shape[0] > 0:
        final_outlier_days = np.r_[final_outlier_days, outliers['quad'][0][:, outlier_days_column_idx]]

    if outliers['ransac'][0].shape[0] > 0:
        final_outlier_days = np.r_[final_outlier_days, outliers['ransac'][0][:, outlier_days_column_idx]]

    return final_outlier_days


def abrupt_change_in_hvac_hours(input_hvac_inefficiency_object, output_hvac_inefficiency_object, logger_pass, device):

    """
        This function estimates abrupt change in HVAC hours

        Parameters:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            output_hvac_inefficiency_object     (dict)          dictionary containing all output information
            logger_pass                         (object)        logger object
            device                              (str)           string indicating device, either AC or SH
        Returns:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            output_hvac_inefficiency_object     (dict)          dictionary containing all output information
    """

    static_params = hvac_static_params()

    # Taking new logger base for this module

    logger_local = logger_pass.get("logger").getChild("cluster_hvac_consumption")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}

    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    random_state = input_hvac_inefficiency_object.get('RANDOM_STATE')

    low_outliers = static_params.get('ineff').get('low_outlier')
    high_outliers = static_params.get('ineff').get('high_outlier')
    total_outliers = static_params.get('ineff').get('total_outlier')
    minimum_hvac_days = static_params.get('ineff').get('minimum_hvac_days')
    lower_limit_quantile = static_params.get('ineff').get('lower_limit_quantile')
    hvac_outlier_limit_list = static_params.get('ineff').get('hvac_outlier_limit_list')

    unconsidered_device = get_unconsidered_device(device)

    # Counting number of hours with HVAC and HVAC potential

    date_index = copy.deepcopy(input_hvac_inefficiency_object.get(device).get('demand_hvac_pivot').get('row'))
    hvac_potential = input_hvac_inefficiency_object.get(device).get('{}_potential_pivot'.format(device)).get('values')

    # Set limit to minimum hvac consumption

    hvac_data_master = copy.deepcopy(input_hvac_inefficiency_object.get(device).get('demand_hvac_pivot').get('values'))
    hvac_temperature = copy.deepcopy(input_hvac_inefficiency_object.get(device).get('temperature_pivot').get('values'))

    unconsidered_hvac_data_master =\
        copy.deepcopy(input_hvac_inefficiency_object.get(unconsidered_device).get('demand_hvac_pivot').get('values'))

    row_idx = input_hvac_inefficiency_object.get(device).get('{}_potential_pivot'.format(device)).get('row')
    column_idx = input_hvac_inefficiency_object.get(device).get('{}_potential_pivot'.format(device)).get('columns')

    hvac_data = hvac_data_master[hvac_data_master != 0]

    # Checking if there is any HVAC consumption

    if hvac_data.shape[0] <= minimum_hvac_days:

        logger.debug('Not enough hvac consumption, skipping outlier |')

        low_outliers = static_params.get('ineff').get('low_outlier')
        high_outliers = static_params.get('ineff').get('high_outlier')
        total_outliers = static_params.get('ineff').get('total_outlier')

        final_outlier_days = np.empty(0,)

        string_for_heading = '{} outliers count: {}, (high: {}, low: {})'.format(device, total_outliers,
                                                                                 high_outliers, low_outliers)

        abrupt_hvac_hours = {'string': string_for_heading,
                             'final_outlier_days': final_outlier_days,
                             'hvac_consumption_matrix': hvac_data_master,
                             'hvac_potential_matrix': hvac_potential,
                             'return_dictionary_zero_saturation': dict({}),
                             'return_dictionary_saturation': dict({}),
                             'return_dictionary': dict({}),
                             'row': row_idx,
                             'columns': column_idx}

        output_hvac_inefficiency_object[device]['abrupt_hvac_hours'] = abrupt_hvac_hours

        return input_hvac_inefficiency_object, output_hvac_inefficiency_object

    lower_limit = super_percentile(hvac_data_master, lower_limit_quantile * 100)

    # Preparing input values for HVAC

    hvac_hour_count = (hvac_data_master > lower_limit).sum(axis=1)
    hvac_potential_hour_count = (hvac_potential > 0).sum(axis=1)

    hvac_temperature_average = np.nanmean(hvac_temperature, axis=1)
    unconsidered_hvac_count = (unconsidered_hvac_data_master > 0).sum(axis=1)

    # removing all days with NaN mean temperature
    nan_temperature_day = np.isnan(hvac_temperature_average)

    hvac_temperature_average = hvac_temperature_average[~nan_temperature_day]
    unconsidered_hvac_count = unconsidered_hvac_count[~nan_temperature_day]
    hvac_potential_hour_count = hvac_potential_hour_count[~nan_temperature_day]
    hvac_hour_count = hvac_hour_count[~nan_temperature_day]
    date_index = date_index[~nan_temperature_day]

    # Handling and managing  regions

    valid_idx = ((hvac_hour_count > 0) & (unconsidered_hvac_count == 0) & (hvac_potential_hour_count > 0) &
                 (hvac_potential_hour_count != Cgbdisagg.HRS_IN_DAY))

    valid_idx_zero_saturation = (hvac_hour_count > 0) & (hvac_potential_hour_count == 0) & \
                                (unconsidered_hvac_count == 0)

    valid_idx_saturation = (hvac_hour_count > 0) & (hvac_potential_hour_count == Cgbdisagg.HRS_IN_DAY) & (unconsidered_hvac_count == 0)

    # count number of zero potential days

    count_zero_potential = np.sum(valid_idx_zero_saturation)
    count_saturation_potential = np.sum(valid_idx_saturation)
    count_potential = np.sum(valid_idx)

    if count_potential > minimum_hvac_days:

        if count_saturation_potential < minimum_hvac_days:
            valid_idx = (valid_idx | valid_idx_saturation)
            valid_idx_saturation = np.zeros_like(valid_idx_saturation, dtype=bool)

        if count_zero_potential < minimum_hvac_days:
            valid_idx = (valid_idx | valid_idx_zero_saturation)
            valid_idx_zero_saturation = np.zeros_like(valid_idx_zero_saturation, dtype=bool)

    elif count_potential <= minimum_hvac_days:
        valid_idx_zero_saturation = valid_idx_zero_saturation | valid_idx | valid_idx_saturation
        valid_idx = np.zeros_like(valid_idx, dtype=bool)
        valid_idx_saturation = np.zeros_like(valid_idx_saturation, dtype=bool)

    column_number = 0

    # Adding Date indexes
    main_input_hvac = np.c_[hvac_potential_hour_count[valid_idx], hvac_temperature_average[valid_idx],
                            date_index[valid_idx]]

    main_output_hvac = hvac_hour_count[valid_idx]

    # Initialising date index for all the outlier days.

    final_outlier_days = np.empty(0,)

    # Append all outlier days to day indices

    outlier_days_column_idx = 2

    return_dictionary = dict({})

    if not (main_input_hvac.shape[0] == 0):
        return_dictionary = outlier_basic_function(main_input_hvac, main_output_hvac,
                                                   hvac_outlier_limit=hvac_outlier_limit_list[0], inlier_deviation=2,
                                                   column_number=column_number, random_state=random_state)

        # Plotting High Outliers
        outliers = return_dictionary['high_outliers']
        total_outliers += len(outliers['quad'][0])
        high_outliers += len(outliers['quad'][0])

        final_outlier_days = get_final_outlier_days(outliers, final_outlier_days, outlier_days_column_idx)

        total_outliers += len(outliers['ransac'][0])
        high_outliers += len(outliers['ransac'][0])

        # Plotting Low outliers

        outliers = return_dictionary['low_outliers']
        total_outliers += len(outliers['quad'][0])
        low_outliers += len(outliers['quad'][0])

        total_outliers += len(outliers['ransac'][0])
        low_outliers += len(outliers['ransac'][0])

    column_number = 1

    # Counting number of hours with HVAC and HVAC potential

    main_input_hvac = np.c_[hvac_potential_hour_count[valid_idx_zero_saturation],
                            hvac_temperature_average[valid_idx_zero_saturation], date_index[valid_idx_zero_saturation]]

    main_output_hvac = hvac_hour_count[valid_idx_zero_saturation]

    return_dictionary_zero_saturation = dict({})

    if not (main_input_hvac.shape[0] == 0):
        return_dictionary_zero_saturation = outlier_basic_function(main_input_hvac, main_output_hvac,
                                                                   hvac_outlier_limit=hvac_outlier_limit_list[1],
                                                                   inlier_deviation=3, column_number=column_number,
                                                                   random_state=random_state)

        # Plotting High Outliers
        outliers = return_dictionary_zero_saturation['high_outliers']
        total_outliers += len(outliers['quad'][0])
        high_outliers += len(outliers['quad'][0])

        final_outlier_days = get_final_outlier_days(outliers, final_outlier_days, outlier_days_column_idx)

        total_outliers += len(outliers['ransac'][0])
        high_outliers += len(outliers['ransac'][0])

        outliers = return_dictionary_zero_saturation['low_outliers']

        total_outliers += len(outliers['quad'][0])
        low_outliers += len(outliers['quad'][0])

        total_outliers += len(outliers['ransac'][0])
        low_outliers += len(outliers['ransac'][0])

    # Counting number of hours with HVAC and HVAC potential

    main_input_hvac = np.c_[hvac_potential_hour_count[valid_idx_saturation],
                            hvac_temperature_average[valid_idx_saturation], date_index[valid_idx_saturation]]

    main_output_hvac = hvac_hour_count[valid_idx_saturation]

    return_dictionary_saturation = dict({})

    if not (main_input_hvac.shape[0] == 0):
        return_dictionary_saturation = outlier_basic_function(main_input_hvac, main_output_hvac,
                                                              hvac_outlier_limit=hvac_outlier_limit_list[2],
                                                              inlier_deviation=2, column_number=column_number,
                                                              random_state=random_state)

        outliers = return_dictionary_saturation['high_outliers']

        total_outliers += len(outliers['quad'][0])
        high_outliers += len(outliers['quad'][0])

        final_outlier_days = get_final_outlier_days(outliers, final_outlier_days, outlier_days_column_idx)

        total_outliers += len(outliers['ransac'][0])
        high_outliers += len(outliers['ransac'][0])

        outliers = return_dictionary_saturation['low_outliers']

        total_outliers += len(outliers['quad'][0])
        low_outliers += len(outliers['quad'][0])

        total_outliers += len(outliers['ransac'][0])
        low_outliers += len(outliers['ransac'][0])

    string_for_heading =\
        '{} outliers count: {}, (high: {}, low: {})'.format(device, total_outliers, high_outliers, low_outliers)

    abrupt_hvac_hours = {'string': string_for_heading,
                         'final_outlier_days': final_outlier_days,
                         'hvac_consumption_matrix': hvac_data_master,
                         'hvac_potential_matrix': hvac_potential,
                         'return_dictionary_zero_saturation': return_dictionary_zero_saturation,
                         'return_dictionary_saturation': return_dictionary_saturation,
                         'return_dictionary': return_dictionary,
                         'row': row_idx,
                         'columns': column_idx,
                         'input_array': np.c_[hvac_hour_count, date_index]}

    output_hvac_inefficiency_object[device]['abrupt_hvac_hours'] = abrupt_hvac_hours

    return input_hvac_inefficiency_object, output_hvac_inefficiency_object
