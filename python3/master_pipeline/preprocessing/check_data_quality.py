"""
Author - Mayank Sharan
Date - 16/10/18
Check data quality performs a bunch of checks on the input data and decides if pipeline is to be run or not
"""

# Import python packages

import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.find_seq import find_seq
from python3.master_pipeline.preprocessing.split_data_by_sample_rate import split_data_by_sample_rate


def check_timezone(meta_data, run_pipeline, rejection_reasons, logger):

    """Utility function to check timezone """

    # Timezone should be present and be a non empty string

    if meta_data.get('timezone') is not None:
        timezone = meta_data.get('timezone')

        if type(timezone) == str:
            if len(timezone) > 0:
                logger.info('Timezone is present | %s', timezone)
            else:
                run_pipeline = False
                rejection_reasons.append('Timezone is empty string')
                logger.warn('Timezone is empty string | %s', timezone)
        else:
            run_pipeline = False
            rejection_reasons.append('Timezone is not a string')
            logger.warn('Timezone is not a string | %s', str(timezone))
    else:
        run_pipeline = False
        rejection_reasons.append('Timezone field missing from the data')
        logger.warn('Timezone field missing from the data |')

    return run_pipeline, rejection_reasons


def check_country(meta_data, run_pipeline, rejection_reasons, logger):

    """Utility function to check country"""

    # Country should be present and be a non empty string

    if meta_data.get('country') is not None:
        country = meta_data.get('country')

        if type(country) == str:
            if len(country) > 0:
                logger.info('Country is present | %s', country)
            else:
                run_pipeline = False
                rejection_reasons.append('Country is empty string')
                logger.warn('Country is empty string | %s', country)
        else:
            run_pipeline = False
            rejection_reasons.append('Country is not a string')
            logger.warn('Country is not a string | %s', str(country))
    else:
        run_pipeline = False
        rejection_reasons.append('Country field missing from the data')
        logger.warn('Country field missing from the data |')

    return run_pipeline, rejection_reasons


def check_pilot_id(pilot_id, run_pipeline, rejection_reasons, logger):

    """Utility function to check pilot id"""

    # Country should be present and be a positive string

    if pilot_id is not None:

        if type(pilot_id) == int:
            if pilot_id > 0:
                logger.info('Pilot id is present | %d', pilot_id)
            else:
                run_pipeline = False
                rejection_reasons.append('Pilot id is invalid')
                logger.warn('Pilot id is invalid | %d', pilot_id)
        else:
            run_pipeline = False
            rejection_reasons.append('Pilot id is not an int')
            logger.warn('Pilot id is not an int | %s', str(pilot_id))
    else:
        run_pipeline = False
        rejection_reasons.append('Pilot id field missing from the data')
        logger.warn('Pilot id field missing from the data |')

    return run_pipeline, rejection_reasons


def check_bc_data(disagg_mode, bill_cycle_count, bill_cycle_reqd_points, last_bc_invalid_count, quality_thresholds,
                  run_pipeline, rejection_reasons, logger):

    """Utility to check data quality per billing cycle"""

    # The following checks are performed here
    # min_pts_per_bc - Minimum percentage of points needed in a billing cycle to consider it as valid
    # minimum_valid_bill_cycles - Minimum percentage of bill cycles needed to be valid to pass the data

    # 'min_pts_per_bc': 90
    # 'minimum_valid_bill_cycles': 70

    bc_pts_perc = [100]

    if disagg_mode == 'incremental' or disagg_mode == 'historical':

        bc_pts_perc = bill_cycle_count * 100 / bill_cycle_reqd_points
        perc_valid_bc = np.sum(bc_pts_perc > quality_thresholds['min_pts_per_bc']) * 100 / len(bc_pts_perc)

        if perc_valid_bc < quality_thresholds['minimum_valid_bill_cycles']:

            run_pipeline = False
            rejection_reasons.append('Percentage of valid billing cycles less than required')
            logger.warn('Percentage of valid billing cycles less than minimum required | %.2f', perc_valid_bc)

        else:
            logger.info('Percentage of valid billing cycles more than minimum required | %.2f', perc_valid_bc)

    # Check for proper amount of data in last billing cycle for incremental mode

    # incremental_last_bc_min_points - In the last billing cycle for incremental mode we need at least 80% data present
    # 'incremental_last_bc_min_points': 80

    if disagg_mode == 'incremental':

        bc_pts_perc[-1] -= last_bc_invalid_count * 100 / bill_cycle_reqd_points[-1]

        if bc_pts_perc[-1] < quality_thresholds['incremental_last_bc_min_points']:

            run_pipeline = False
            rejection_reasons.append('Percentage of points needed in last billing cycle is less than required')
            logger.warn('Percentage of data in last billing cycle less than minimum required | %.3f', bc_pts_perc[-1])

        else:
            logger.info('Percentage of data in last billing cycle more than minimum required | %.3f', bc_pts_perc[-1])

    return run_pipeline, rejection_reasons


def check_bc_col(nan_count_columns, diff_by_col, quality_thresholds, run_pipeline, rejection_reasons, logger):

    """Utility to check thresholds on column containing bill cycle timestamp of the data"""

    # Check that there are no nans in the column

    if nan_count_columns[Cgbdisagg.INPUT_BILL_CYCLE_IDX] > quality_thresholds['maximum_nan_c1to6']:

        run_pipeline = False
        rejection_reasons.append('Billing cycle timestamps have more NaN than maximum')
        logger.warn('Billing cycle timestamps have more NaN than maximum | %d',
                    nan_count_columns[Cgbdisagg.INPUT_BILL_CYCLE_IDX])

    # Check that the values are always non-decreasing

    elif np.sum(diff_by_col[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] < 0) > 0:

        run_pipeline = False
        rejection_reasons.append('Billing cycle timestamps difference is negative')
        logger.warn('Billing cycle timestamps difference is negative at | %s',
                    str(np.where(diff_by_col[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] < 0)[0]))

    else:
        logger.info('Billing cycle timestamps adhere to required standards |')

    return run_pipeline, rejection_reasons


def check_week_col(nan_count_columns, diff_by_col, quality_thresholds, run_pipeline, rejection_reasons, logger):

    """Utility to check thresholds on column containing week timestamp of the data"""

    # Check that there are no nans in the column

    if nan_count_columns[Cgbdisagg.INPUT_WEEK_IDX] > quality_thresholds['maximum_nan_c1to6']:

        run_pipeline = False
        rejection_reasons.append('Week timestamps have more NaN than maximum')
        logger.warn('Week timestamps have more NaN than maximum | %d', nan_count_columns[Cgbdisagg.INPUT_WEEK_IDX])

    # Check that the values are always non-decreasing

    elif np.sum(diff_by_col[:, Cgbdisagg.INPUT_WEEK_IDX] < 0) > 0:

        run_pipeline = False
        rejection_reasons.append('Week timestamps difference is negative')
        logger.warn('Week timestamps difference is negative at | %s',
                    str(np.where(diff_by_col[:, Cgbdisagg.INPUT_WEEK_IDX] < 0)[0]))

    else:
        logger.info('Week timestamps adhere to required standards |')

    return run_pipeline, rejection_reasons


def check_day_col(nan_count_columns, diff_by_col, quality_thresholds, run_pipeline, rejection_reasons, logger):

    """Utility to check thresholds on column containing day timestamp of the data"""

    # Check that there are no nans in the column

    if nan_count_columns[Cgbdisagg.INPUT_DAY_IDX] > quality_thresholds['maximum_nan_c1to6']:

        run_pipeline = False
        rejection_reasons.append('Day timestamps have more NaN than maximum')
        logger.warn('Day timestamps have more NaN than maximum | %d', nan_count_columns[Cgbdisagg.INPUT_DAY_IDX])

    # Check that the values are always non-decreasing

    elif np.sum(diff_by_col[:, Cgbdisagg.INPUT_DAY_IDX] < 0) > 0:

        run_pipeline = False
        rejection_reasons.append('Day timestamps difference is negative')
        logger.warn('Day timestamps difference is negative at | %s',
                    str(np.where(diff_by_col[:, Cgbdisagg.INPUT_DAY_IDX] < 0)[0]))

    else:
        logger.info('Day timestamps adhere to required standards |')

    return run_pipeline, rejection_reasons


def check_dow_col(nan_count_columns, diff_by_col, quality_thresholds, run_pipeline, rejection_reasons, logger):

    """Utility to check thresholds on column containing day of the week of the data"""

    # Check that there are no nans in the column

    if nan_count_columns[Cgbdisagg.INPUT_DOW_IDX] > quality_thresholds['maximum_nan_c1to6']:

        run_pipeline = False
        rejection_reasons.append('Day of week has more NaN than maximum')
        logger.warn('Day of week has more NaN than maximum | %d', nan_count_columns[Cgbdisagg.INPUT_DOW_IDX])

    else:
        logger.info('Day of week adheres to required standards |')

    return run_pipeline, rejection_reasons


def check_hod_col(nan_count_columns, diff_by_col, quality_thresholds, run_pipeline, rejection_reasons, logger):

    """Utility to check thresholds on column containing hour of day of the data"""

    # Check that there are no nans in the column

    if nan_count_columns[Cgbdisagg.INPUT_HOD_IDX] > quality_thresholds['maximum_nan_c1to6']:

        run_pipeline = False
        rejection_reasons.append('Hour of day has more NaN than maximum')
        logger.warn('Hour of day has more NaN than maximum | %d', nan_count_columns[Cgbdisagg.INPUT_HOD_IDX])

    else:
        logger.info('Hour of day adheres to required standards |')

    return run_pipeline, rejection_reasons


def check_epoch_col(nan_count_columns, diff_by_col, quality_thresholds, run_pipeline, rejection_reasons, logger):

    """Utility to check thresholds on column containing epoch timestamp of the data"""

    # Check that there are no nans in the column

    if nan_count_columns[Cgbdisagg.INPUT_EPOCH_IDX] > quality_thresholds['maximum_nan_c1to6']:

        run_pipeline = False
        rejection_reasons.append('Epoch timestamps have more NaN than maximum')
        logger.warn('Epoch timestamps have more NaN than maximum | %d', nan_count_columns[Cgbdisagg.INPUT_EPOCH_IDX])

    # Check that the values are always increasing

    elif np.sum(diff_by_col[:, Cgbdisagg.INPUT_EPOCH_IDX] <= 0) > 0:

        run_pipeline = False
        rejection_reasons.append('Epoch timestamps difference is negative')
        logger.warn('Epoch timestamps difference is non-positive at | %s',
                    str(np.where(diff_by_col[:, Cgbdisagg.INPUT_EPOCH_IDX] <= 0)[0]))

    else:
        logger.info('Epoch timestamps adhere to required standards |')

    return run_pipeline, rejection_reasons


def perform_check_min(value, threshold, success_message, failure_message, write_format, run_pipeline, rejection_reasons,
                      logger):

    """Utility to test min based thresholds in a generic format"""

    if value < threshold:
        run_pipeline = False
        rejection_reasons.append(failure_message)
        logger.warn(failure_message + ' ' + ' | ' + write_format, value)
    else:
        logger.info(success_message + ' ' + ' | ' + write_format, value)

    return run_pipeline, rejection_reasons


def perform_check_max(value, threshold, success_message, failure_message, write_format, run_pipeline, rejection_reasons,
                      logger):

    """Utility to test max based thresholds in a generic format"""

    if value > threshold:
        run_pipeline = False
        rejection_reasons.append(failure_message)
        logger.warn(failure_message + ' ' + ' | ' + write_format, value)
    else:
        logger.info(success_message + ' ' + ' | ' + write_format, value)

    return run_pipeline, rejection_reasons


def compute_disconnection_metrics(input_data, input_split, quality_thresholds):

    """Utility to compute parameters regarding disconnections in data we get"""

    num_constant_cons = 0
    is_const_cons = np.full(shape=(input_data.shape[0],), fill_value=False)

    for split in input_split:
        num_allowed_constant = (quality_thresholds['minimum_hrs_for_disconnection'] *
                                int(Cgbdisagg.SEC_IN_HOUR / split[0]) - 1)
        split_cons_diff_data = np.diff(input_data[int(split[1]): int(split[2]) + 1, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        diff_seq = find_seq(split_cons_diff_data, num_allowed_constant)

        if len(diff_seq > 0):
            diff_seq_constant = diff_seq[diff_seq[:, 0] == 0, :]

            num_constant_cons += np.sum(diff_seq_constant[:, 3])

            for disconnection in diff_seq_constant:
                is_const_cons[int(disconnection[1]): int(disconnection[2]) + 2] = True

    return num_constant_cons, is_const_cons


def check_data_quality(pipeline_input_object, parent_logger):

    """
    Perform data quality checks on the raw data to determine whether the disagg pipeline should be run

    Parameters:
        pipeline_input_object (dict)              : Dictionary containing all inputs needed to run the pipeline
        parent_logger       (logger)            : Logger object to inherit child logger from

    Return:
        pipeline_input_object (dict)              : Dictionary containing all inputs needed to run the pipeline
        run_pipeline        (bool)              : If true data is good enough to run pipeline else we skip
        rejection_reasons   (list)              : Lists all issues with data due to which it was rejected
    """

    # Initialize the logger

    logger_base = parent_logger.getChild('check_data_quality')
    logger = logging.LoggerAdapter(logger_base, pipeline_input_object.get('logging_dict'))

    # Initialize variables indicating data quality

    run_pipeline = True
    rejection_reasons = []

    # Initialize quality thresholds for different columns, These are used to decide selection / rejection of data
    # All thresholds are percentage values

    # minimum_hrs_for_disconnection - The number of hours for which the consumption value needs to be constant to mark
    # the period as a disconnection. Seems to be prior knowledge.

    # maximum_data_points_missing - Maximum percentage of points missing in the data allowed w.r.t. expected number of
    # data points. For example this allows ~320 days of missing data in 13 months of expected data.

    # maximum_nan_c1to6 - Maximum percentage of nan values allowed in the first 6 columns of the data. Currently we do
    # not tolerate any nans.

    # minimum_sunrise_sunset_perc - Minimum percentage of days with sunset / sunrise values required w.r.t. expected
    # number of days. Set to 30% right now.

    # minimum_temp_perc - Minimum percentage of temperature values required w.r.t. expected number of hours of data. We
    # need 90% of the given data to contain values. Since the given data is allowed to have 10% missing from expected
    # points 90% of 90% sets the threshold at 81%. Relaxing it by 1% we arrive at the current number of 80%.

    # minimum_valid_cons - Minimum percentage of valid consumption values required w.r.t expected number of data points.
    # Criteria of invalidity include disconnections characterised as constant consumption value for >= 3 hours,
    # 0 consumption values, nan consumption values and missing consumption values. Since missing values have an allowed
    # threshold of 20% in best case we allow 5% bad data. In 13 months 15% is ~2 months of bad data.

    # max_consecutive_missing_days - Maximum consecutive days of missing data allowed.

    # min_pts_per_bc - Minimum percentage of points needed in a billing cycle to consider it as valid

    # minimum_valid_bill_cycles - Minimum percentage of bill cycles needed to be valid to pass the data

    # incremental_last_bc_min_points - In the last billing cycle for incremental mode we need at least 80% data present

    quality_thresholds = {
        'minimum_hrs_for_disconnection': 3,
        'maximum_data_points_missing': 80,
        'maximum_nan_c1to6': 0,
        'minimum_sunrise_sunset_perc': 10,
        'minimum_temp_perc': 20,
        'minimum_valid_cons': 15,
        'max_consecutive_missing_days': 45,
        'min_pts_per_bc': 90,
        'minimum_valid_bill_cycles': 0,
        'incremental_last_bc_min_points': 20,
        'min_entries': 24 * 7,
    }

    # Check for existence of critical fields of metadata

    meta_data = pipeline_input_object.get('home_meta_data')

    # Check for timezone value

    run_pipeline, rejection_reasons = check_timezone(meta_data, run_pipeline, rejection_reasons, logger)

    # Check for Country value

    run_pipeline, rejection_reasons = check_country(meta_data, run_pipeline, rejection_reasons, logger)

    # Check for pilot id

    pilot_id = pipeline_input_object.get('global_config').get('pilot_id')
    run_pipeline, rejection_reasons = check_pilot_id(pilot_id, run_pipeline, rejection_reasons, logger)

    # Initialize things to perform checks

    input_data = pipeline_input_object.get('input_data')
    num_entries = input_data.shape[0]

    # Check for minimum number of data points

    write_format = '%d'
    success_message = 'Number of data points greater than minimum required'
    failure_message = 'Less data points than minimum required'
    run_pipeline, rejection_reasons = perform_check_min(num_entries, quality_thresholds['min_entries'], success_message,
                                                        failure_message, write_format, run_pipeline, rejection_reasons,
                                                        logger)

    # Ensure that is the data fails minimum data points required we skip it

    if not run_pipeline:
        quality_metrics = {
            'constant_cons_perc': -1,
            'missing_data_perc': -1,
            'present_sunrise_perc': -1,
            'present_sunset_perc': -1,
            'present_temp_perc': -1,
            'valid_cons_perc': -1,
            'zero_cons_perc': -1,
            'neg_cons_perc': -1,
        }

        pipeline_input_object['data_quality_metrics'] = quality_metrics

        return pipeline_input_object, run_pipeline, rejection_reasons

    # Introduce splitting data to handle variable sampling rate

    # Input split is of dimensions n x 4

    col_num_pts_in_split = 3
    col_pts_start_idx = 1
    col_sampling_rate = 0
    min_pts_for_data_split = 10

    input_split = split_data_by_sample_rate(input_data)

    # Check for the maximum consecutive gap size. Minor bug here where a boundary case with DST shift can fail

    input_split_max_gap = np.max(input_split[:, col_sampling_rate])
    num_days_max_gap = input_split_max_gap / Cgbdisagg.SEC_IN_DAY

    disagg_mode = pipeline_input_object.get('global_config').get('disagg_mode')

    if disagg_mode == 'historical':

        logger.info('Number of consecutive missing days | %.1f', num_days_max_gap)

        # Comment 1 - write_format = '%.1f'
        # Comment 2 - success_message = 'Number of consecutive missing days within limit'
        # Comment 3 - failure_message = 'Greater consecutive missing days than maximum allowed'
        # Comment 4 - run_pipeline, rejection_reasons = perform_check_max(num_days_max_gap,
        # Comment 4 - quality_thresholds['max_consecutive_missing_days'],
        # Comment 5 - success_message, failure_message, write_format,
        # Comment 6 - run_pipeline, rejection_reasons, logger)

    # Merge input splits for further operations

    if np.sum(input_split[:, col_num_pts_in_split] > min_pts_for_data_split) > 0:
        input_split = input_split[input_split[:, col_num_pts_in_split] > min_pts_for_data_split, :]

    num_chunks = input_split.shape[0]

    # Merge consecutive chunks

    if num_chunks > 1:
        for idx in range(1, num_chunks):

            # Check if consecutive chunks have same sampling rate

            if input_split[idx, col_sampling_rate] == input_split[idx - 1, col_sampling_rate]:

                # Shift all elements from the previous chunk to the next one

                input_split[idx - 1, col_sampling_rate] = 0
                input_split[idx, col_pts_start_idx] = input_split[idx - 1, col_pts_start_idx]
                input_split[idx, col_num_pts_in_split] += input_split[idx - 1, col_num_pts_in_split]

    input_split = input_split[input_split[:, col_sampling_rate] > 0, :]

    # Compute number of expected points for each split and the billing cycles

    bill_cycles, bill_cycle_count = np.unique(ar=input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_counts=True)
    bill_cycle_reqd_points = np.full(shape=(len(bill_cycles),), fill_value=0)

    num_total_points = 0
    num_hourly_points = 0

    for split in input_split:

        split = split.astype(int)
        split_time = input_data[split[2], Cgbdisagg.INPUT_EPOCH_IDX] - input_data[split[1], Cgbdisagg.INPUT_EPOCH_IDX]

        # Pull out the first and last timestamps of each billing cycle within the split
        split_data = input_data[split[1]: split[2] + 1, :]
        split_data_bc = split_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]
        bill_cycles_in_split = np.unique(ar=split_data_bc)

        for bill_cycle in bill_cycles_in_split:
            temp_data = split_data[split_data_bc == bill_cycle, :]
            num_pts_bc = int((temp_data[-1, Cgbdisagg.INPUT_EPOCH_IDX] - temp_data[0, Cgbdisagg.INPUT_EPOCH_IDX]) /
                             split[0])
            bill_cycle_reqd_points[bill_cycles == bill_cycle] += (num_pts_bc + 1)

        num_total_points += int(split_time / split[0]) + 1
        num_hourly_points += int(split_time / Cgbdisagg.SEC_IN_HOUR) + 1

    num_days = np.ceil(input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX] -
                       input_data[0, Cgbdisagg.INPUT_EPOCH_IDX] / Cgbdisagg.SEC_IN_DAY) + 1

    # Check percentage of data available vs expected data

    missing_data_perc = float(num_total_points - num_entries) * 100 / num_total_points

    write_format = '%.2f'
    success_message = 'Percentage of missing data points within limit'
    failure_message = 'Greater percentage of data points missing than maximum allowed'
    run_pipeline, rejection_reasons = perform_check_max(missing_data_perc,
                                                        quality_thresholds['maximum_data_points_missing'],
                                                        success_message, failure_message, write_format, run_pipeline,
                                                        rejection_reasons, logger)

    # Compute number of nans in each column

    nan_bool_matrix = np.isnan(input_data)
    nan_count_columns = np.sum(nan_bool_matrix, axis=0)

    # Compute diff for each columns

    diff_by_col = np.diff(input_data, axis=0)

    # Perform checks column by column

    # Column 0, Billing cycle timestamp, Check for no Nan and no -ve diff

    run_pipeline, rejection_reasons = check_bc_col(nan_count_columns, diff_by_col, quality_thresholds, run_pipeline,
                                                   rejection_reasons, logger)

    # Column 1, Week start timestamp, Check for no Nan and -ve diff

    run_pipeline, rejection_reasons = check_week_col(nan_count_columns, diff_by_col, quality_thresholds, run_pipeline,
                                                     rejection_reasons, logger)

    # Column 2, Day start timestamp, Check for no Nan and -ve diff

    run_pipeline, rejection_reasons = check_day_col(nan_count_columns, diff_by_col, quality_thresholds, run_pipeline,
                                                    rejection_reasons, logger)

    # Column 3, Day of week, check for no Nan

    run_pipeline, rejection_reasons = check_dow_col(nan_count_columns, diff_by_col, quality_thresholds, run_pipeline,
                                                    rejection_reasons, logger)

    # Column 4, Hour of day, check for no Nan

    run_pipeline, rejection_reasons = check_hod_col(nan_count_columns, diff_by_col, quality_thresholds, run_pipeline,
                                                    rejection_reasons, logger)

    # Column 5, Epoch timestamp, check for no Nan; no 0 or -ve difference

    run_pipeline, rejection_reasons = check_epoch_col(nan_count_columns, diff_by_col, quality_thresholds, run_pipeline,
                                                      rejection_reasons, logger)

    # Column 7, Temperature, check for 80% hvac values to be present

    temp_present_perc = (float(num_entries - nan_count_columns[Cgbdisagg.INPUT_TEMPERATURE_IDX]) * 100 /
                         num_hourly_points)

    write_format = '%.2f'
    success_message = 'Temperature present adheres to minimum required percentage'
    failure_message = 'Temperature present is less than required percentage'
    run_pipeline, rejection_reasons = perform_check_min(temp_present_perc,
                                                        quality_thresholds['minimum_temp_perc'],
                                                        success_message, failure_message, write_format, run_pipeline,
                                                        rejection_reasons, logger)

    # Column 8 - 10, Not to be checked as of now since nobody uses it. Checks can be added here if needed later

    # Prepare sunrise and sunset for day wise examination

    _, day_map_idx = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_index=True)

    day_wise_sunrise_sunset = input_data[day_map_idx, Cgbdisagg.INPUT_SUNRISE_IDX: Cgbdisagg.INPUT_SUNSET_IDX + 1]
    sunrise_sunset_nan_bool_matrix = np.isnan(day_wise_sunrise_sunset)
    sunrise_sunset_nan_count = np.sum(sunrise_sunset_nan_bool_matrix, axis=0)

    # Column 11, Sunrise timestamps, check for 70% by day values to be present

    sunrise_present_perc = 100 - (float(sunrise_sunset_nan_count[0]) * 100 / num_days)

    write_format = '%.2f'
    success_message = 'Sunrise timestamps present adhere to minimum required percentage'
    failure_message = 'Sunrise timestamps present are less than required percentage'
    run_pipeline, rejection_reasons = perform_check_min(sunrise_present_perc,
                                                        quality_thresholds['minimum_sunrise_sunset_perc'],
                                                        success_message, failure_message, write_format, run_pipeline,
                                                        rejection_reasons, logger)

    # Column 12, Sunset timestamps, check for 50% overall values to be present [Need meaningful threshold]

    sunset_present_perc = 100 - (float(sunrise_sunset_nan_count[1]) * 100 / num_days)

    write_format = '%.2f'
    success_message = 'Sunset timestamps present adhere to minimum required percentage'
    failure_message = 'Sunset timestamps present are less than required percentage'
    run_pipeline, rejection_reasons = perform_check_min(sunset_present_perc,
                                                        quality_thresholds['minimum_sunrise_sunset_perc'],
                                                        success_message, failure_message, write_format, run_pipeline,
                                                        rejection_reasons, logger)

    # Column 6, the holy grail, Multiple checks on consumption values

    # Set percentage of 0 consumption values

    is_zero_cons = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] == 0
    num_zeros_cons = np.sum(is_zero_cons)
    zero_cons_perc = float(num_zeros_cons) * 100 / num_total_points

    # Set percentage of constant consumption values

    # Compute count of constant values that span over 3 hours for each sampling chunk

    num_constant_cons, is_const_cons = compute_disconnection_metrics(input_data, input_split, quality_thresholds)

    constant_cons_perc = float(num_constant_cons) * 100 / num_total_points

    # Set percentage of nan consumption values

    nan_cons_perc = float(nan_count_columns[Cgbdisagg.INPUT_CONSUMPTION_IDX]) * 100 / num_total_points

    # Overall validity check on consumption values

    is_invalid_cons = np.logical_or(np.logical_or(is_zero_cons, is_const_cons),
                                    nan_bool_matrix[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    num_valid_cons = num_entries - np.sum(is_invalid_cons)
    valid_cons_perc = float(num_valid_cons) * 100 / num_total_points

    # Get percentage of points that are negative

    neg_cons_perc = np.sum(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] < 0)

    # Add additional bill cycle level data missing checks for incremental and historical mode

    last_bc_idx = input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == bill_cycles[-1]
    last_bc_invalid_count = np.sum(np.logical_and(last_bc_idx, is_invalid_cons))

    run_pipeline, rejection_reasons = check_bc_data(disagg_mode, bill_cycle_count, bill_cycle_reqd_points,
                                                    last_bc_invalid_count, quality_thresholds, run_pipeline,
                                                    rejection_reasons, logger)

    # Finalise if consumption values pass all checks or not

    write_format = '%.2f'
    success_message = 'Valid consumption percentage adheres to the minimum required'
    failure_message = 'Valid consumption percentage is less than required'
    run_pipeline, rejection_reasons = perform_check_min(valid_cons_perc,
                                                        quality_thresholds['minimum_valid_cons'],
                                                        success_message, failure_message, write_format, run_pipeline,
                                                        rejection_reasons, logger)

    logger.info('Percentage of consumption values that are 0 = | %.2f', zero_cons_perc)
    logger.info('Percentage of consumption values that are nan = | %.2f', nan_cons_perc)
    logger.info('Percentage of consumption values that are negative = | %.2f', neg_cons_perc)
    logger.info('Percentage of consumption values that are constant = | %.2f', constant_cons_perc)

    # Set quality metrics field in disagg input object and return stuff

    # constant_cons_perc    : Percentage of constant consumption points wrt expected points
    # missing_data_perc     : Percentage of missing points wrt expected points
    # present_sunrise_perc  : Percentage of sunrise points present
    # present_sunset_perc   : Percentage of sunset points present
    # present_temp_perc     : Hourly % of points for which temperature data is present
    # valid_cons_perc       : Percentage of valid consumption points wrt expected points
    # zero_cons_perc        : Percentage of zero consumption points wrt expected points
    # neg_cons_perc         : Percentage of negative consumption points wrt expected points
    # is_valid_temp         : To be filled after homogenise data
    # is_valid_cons         : To be filled after homogenise data
    # is_nan_cons           : To be filled after homogenise data
    # is_neg_cons           : To be filled after homogenise data

    quality_metrics = {
        'constant_cons_perc': constant_cons_perc,
        'missing_data_perc': missing_data_perc,
        'present_sunrise_perc': sunrise_present_perc,
        'present_sunset_perc': sunset_present_perc,
        'present_temp_perc': temp_present_perc,
        'valid_cons_perc': valid_cons_perc,
        'zero_cons_perc': zero_cons_perc,
        'neg_cons_perc': neg_cons_perc,
        'is_valid_temp': None,
        'is_valid_cons': None,
        'is_nan_cons': None,
        'is_neg_cons': None,
    }

    pipeline_input_object['data_quality_metrics'] = quality_metrics

    return pipeline_input_object, run_pipeline, rejection_reasons
