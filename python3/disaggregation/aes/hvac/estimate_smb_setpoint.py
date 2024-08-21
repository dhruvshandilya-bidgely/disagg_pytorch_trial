"""
Author - Mirambika Sikdar
Date - 21/12/2023
Estimating setpoint for smb hvac
"""

# Import python packages

import copy
import logging
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.linear_fit import linear_fit
from python3.disaggregation.aes.hvac.init_hourly_smb_hvac_params import hvac_static_params

static_params = hvac_static_params()


def remove_outlier_days(degree_day, hvac_params, daily_consumption, valid_idx):
    """
    Function to Remove outlier weeks while determining setpoints for cooling and heating

    Parameters:
        degree_day (numpy array)                : 2D Array of cdd/hdd
        hvac_params (dict)                      : Dictionary containing hvac algo related initialized parameters
        daily_consumption (numpy array)         : Array of day level consumption aggregates
        valid_idx (numpy array)                 : array to identify valid days to perform regression
    Returns:
        valid_idx (numpy array)                 : array to identify valid days to perform regression
    """

    degree_day_max = np.mean(degree_day[degree_day > 0]) + hvac_params['ARM_OF_STANDARD_DEVIATION'] * \
                     np.std(degree_day[degree_day > 0])
    degree_day_min = np.mean(degree_day[degree_day > 0]) - hvac_params['ARM_OF_STANDARD_DEVIATION'] * \
                     np.std(degree_day[degree_day > 0])

    daily_consumption_max = np.mean(daily_consumption[daily_consumption > 0]) + \
                            hvac_params['ARM_OF_STANDARD_DEVIATION'] * np.std(daily_consumption[daily_consumption > 0])
    daily_consumption_min = np.mean(daily_consumption[daily_consumption > 0]) - \
                            hvac_params['ARM_OF_STANDARD_DEVIATION'] * np.std(daily_consumption[daily_consumption > 0])

    valid_idx[(daily_consumption < daily_consumption_min)] = 0

    no_hvac_days_1 = np.sum((daily_consumption < daily_consumption_min) & (degree_day > degree_day_max))
    no_hvac_days_2 = np.sum((daily_consumption > daily_consumption_max) & (degree_day < degree_day_min))

    max_days_to_remove = hvac_params['MAX_DAYS_TO_REMOVE']
    if no_hvac_days_1 <= max_days_to_remove:
        valid_idx[(daily_consumption < daily_consumption_min) & (degree_day > degree_day_max)] = 0
    if no_hvac_days_2 <= max_days_to_remove:
        valid_idx[(daily_consumption > daily_consumption_max) & (degree_day < degree_day_min)] = 0

    return valid_idx


def get_consumption_thresholds(detection_debug, logger_hvac, hvac_params, hvac_input_consumption):
    """
    Function to get the consumption thresholds for filtering out improbable HVAC data

    Parameters:
        detection_debug (dict)                  : Dictionary containing detection related debugging information
        logger_hvac (logging object)            : Writes logs during code flow
        hvac_params (dict)                      : Dictionary containing HVAC algo related parameters
        hvac_input_consumption (np.ndarray)     : Array containing epoch level energy consumption info

    Returns:
        min_threshold (float)                   : Minimum permissible hvac consumption at epoch level
        cap_threshold (float)                   : Maximum permissble hvac consumption at epoch level
    """

    if np.isnan(detection_debug['mu']) or np.isnan(detection_debug['sigma']):

        min_threshold = hvac_params['MIN_AMPLITUDE']
        cap_threshold = np.max(hvac_input_consumption)

        logger_hvac.info(' Min and Cap thresholds are taken from hvac-params because mu or sigma detection failed |')

    else:

        min_threshold = detection_debug['amplitude_cluster_info']['cluster_limits'][0][0]

        if detection_debug['amplitude_cluster_info']['cluster_limits'][1][1] == np.inf:
            cap_threshold = detection_debug['amplitude_cluster_info']['cluster_limits'][0][1]
        else:
            cap_threshold = detection_debug['amplitude_cluster_info']['cluster_limits'][1][1]

        logger_hvac.info(' Min and Cap thresholds are taken from detection |')

    return min_threshold, cap_threshold


def get_epoch_temperature_degree(hvac_params, temperature, setpoint_index):
    """
    Function calculates cdd/hdd using epoch level temperature information

    Parameters:
        hvac_params (dict)                      : Dictionary containing HVAC algo related parameters
        temperature (np.ndarray)                : Array containing epoch level temperature
        setpoint_index (int)                    : Index integer, looping over list of setpoints

    Returns:
        epoch_temperature_degree (np.ndarray)   : Array containing epoch level cooling/heating degree values
    """

    if hvac_params['IS_AC']:

        # for ac storing epoch level cdd lever in temp
        epoch_temperature_degree = np.maximum(temperature - hvac_params['SETPOINTS'][setpoint_index], 0)
        epoch_temperature_degree[np.isnan(epoch_temperature_degree)] = 0

    else:

        # for sh storing epoch level hdd lever in temp
        epoch_temperature_degree = np.maximum(hvac_params['SETPOINTS'][setpoint_index] - temperature, 0)
        epoch_temperature_degree[np.isnan(epoch_temperature_degree)] = 0

    return epoch_temperature_degree


def choose_model_kind(fit_model_linear, fit_model_sqrt):
    """
    Function to pick best trend function for setpoint estimate

    Parameters:
        fit_model_linear (object)       : Model object containing model information (linear fit)
        fit_model_sqrt (object)         : Model object containing model information (root fit)

    Returns:
        fit_model (object)              : Chosen Model object containing model information
    """

    if fit_model_linear['Rsquared']['Ordinary'] > fit_model_sqrt['Rsquared']['Ordinary']:
        fit_model = fit_model_linear
    else:
        fit_model = fit_model_sqrt

    return fit_model


def get_degree_day_setpoint_metrics(best_setpoint_row, qualified_df, hvac_params, df_fit_coefficients, logger_hvac):
    """
    Function to extract best setpoint, corresponding r-square and p-value

    Parameters :
        best_setpoint_row (np.ndarray)  : Row indicator for best setpoint
        qualified_df (object)           : Dataframe containing setpoint and its attributes, with valid p-values
        hvac_params (dict)              : Dictionary containing HVAC algo related config parameters
        df_fit_coefficients (object)    : Dataframe containing only attributes for setpoints
        logger_base (logging object)    : Writes logs during code flow

    Returns:
        degree_day_setpoint (int) : Setpoint value
        degree_day_rsq (float) : Corresponding r square
        degree_day_pval (float) : Corresponding p-value
    """

    if any(best_setpoint_row):

        degree_day_rsq = float(qualified_df['r_squared'][best_setpoint_row])
        degree_day_setpoint = int(qualified_df['setpoints'][best_setpoint_row])
        degree_day_pval = float(qualified_df['p_value'][best_setpoint_row])

        logger_hvac.info(
            ' Best degree day Setpoint : {} with r-square : {} |'.format(degree_day_setpoint, degree_day_rsq))

        if degree_day_rsq > 0.5:

            logger_hvac.info(' Raw degree-day-rSquare greater than 0.5 is good : {} |'.format(degree_day_rsq))

            qualified_df = qualified_df[qualified_df['r_squared'] >= 0.5]
            selected_row = -1
            degree_day_rsq = float(qualified_df['r_squared'].iloc[selected_row])
            degree_day_setpoint = int(qualified_df['setpoints'].iloc[selected_row])
            degree_day_pval = float(qualified_df['p_value'].iloc[selected_row])

        else:

            logger_hvac.info(' Raw degree-day-rSquare less than 0.5. No adjustment required |')

    else:

        degree_day_setpoint = hvac_params['SETPOINTS'][0]
        degree_day_rsq = float(
            df_fit_coefficients[df_fit_coefficients['setpoints'] == degree_day_setpoint]['r_squared'])
        degree_day_pval = float(df_fit_coefficients[df_fit_coefficients['setpoints'] == degree_day_setpoint]['p_value'])

    return degree_day_setpoint, degree_day_rsq, degree_day_pval


def ensure_single_best(best_setpoint_row):
    """
    Function to ensure only one best ssetpoint row exists

    Prameters:

        best_setpoint_row (object)  : Array containing booleans of best setpoints

    Returns:
        best_setpoint_row (object)  : Array containing booleans of best setpoints, with only one true
    """

    if np.sum(best_setpoint_row) > 1:

        flag = 1

        for index in best_setpoint_row.index:

            if (flag == 1) and best_setpoint_row[index]:
                flag = 0
            elif flag == 0:
                best_setpoint_row[index] = False

    return best_setpoint_row


def setpoint_by_mode_at_x_hour(detection_debug, day_idx, hvac_params, min_cdd_quantile, logger_base, common_objects):
    """
    Function to detect setpoint for sh or ac for a user

    Parameters:

        detection_debug(dict)             : Dictionary containing hvac detection stage attributes
        day_idx (numpy array)             : Array of day identifier indexes (epochs of a day have same index)
        hvac_params (dict)                : Dictionary containing hvac algo related initialized parameters
        min_cdd_quantile (float)          : Indicates the percentile limits of mid temperature range to consider
        logger_base (logging object)      : Writes logs during code flow
        common_objects (dict)             : Dictionary containing common objects like input data, config, output object

    Returns:

        setpoint_estimate (numpy array)   : 2D array of day epoch and daily cooling-heating estimates
        hvac_exit_status (dict)           : Dictionary containing hvac setpoint value and its existence flag
    """

    logger_local = logger_base.get("logger").getChild("detect_setpoint")
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    hvac_input_data = common_objects['hvac_input_data']
    invalid_idx = common_objects['invalid_idx']
    global_config = common_objects['global_config']
    hvac_exit_status = common_objects['hvac_exit_status']
    disagg_output_object = common_objects['disagg_output_object']

    hvac_input_consumption = copy.deepcopy(hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    temperature = copy.deepcopy(hvac_input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX])

    if not detection_debug.get('mu'):
        logger_hvac.info(' Exiting without setpoint : appliance amplitude not estimated in detection module. |')
        setpoint_estimate = {'exist': False, 'setpoint': np.nan}
        return setpoint_estimate, hvac_exit_status

    try:

        logger_hvac.info(' Trying to get the minimum and maximum permissible epoch level energy |')
        min_threshold, cap_threshold = get_consumption_thresholds(detection_debug, logger_hvac, hvac_params,
                                                                  hvac_input_consumption)
        logger_hvac.info(
            ' Epoch Min Threshold : {} W , Epoch Cap Threshold : {} W |'.format(min_threshold, cap_threshold))

    except (ValueError, IndexError, KeyError):

        logger_hvac.info(' Exiting without setpoint: mu exists, but evaluation of min amplitude per epoch failed |')
        hvac_exit_status[' Exit_status'].append('mu exists, but evaluation of min amplitude per epoch failed')
        setpoint_estimate = {'exist': False, 'setpoint': np.nan}

        return setpoint_estimate

    # beginning filtering for setpoint estimation
    logger_hvac.info(' F0: {} : beginning filtering |'.format(hvac_input_data.shape[0]))

    # filtering out epoch consumption lower than min threshold
    hvac_input_consumption[np.logical_or(invalid_idx, hvac_input_consumption < min_threshold)] = 0
    hvac_input_consumption[hvac_input_consumption > cap_threshold] = cap_threshold
    logger_hvac.info(' F1:E: {} : filtering out epoch consumption lower than min threshold: {} |'.format(
        np.sum(hvac_input_consumption > 0), np.around(min_threshold, 2)))

    # Aggregating at X-Hours level for best setpoint estimate
    epoch_array = np.arange(len(hvac_input_consumption))
    epochs_per_hour = Cgbdisagg.SEC_IN_HOUR / global_config['sampling_rate']
    hours_aggregation = disagg_output_object['switch']['hvac']['hour_aggregate_level']
    aggregation_factor = hours_aggregation * epochs_per_hour
    aggregate_identifier = ((epoch_array + static_params['inequality_handler']) // aggregation_factor).astype(int)
    aggregated_consumption = np.bincount(aggregate_identifier, hvac_input_consumption)
    logger_hvac.info(' Aggregated consumption at {} hours |'.format(hours_aggregation))

    logger_hvac.info(' Initializing fit-coefficients matrix {}X6 |'.format(len(hvac_params['SETPOINTS'])))
    fit_coefficients = np.empty(shape=[len(hvac_params['SETPOINTS']), 6])
    logger_hvac.info(' loop at {} setpoints to fill fit-coefficient matrix |'.format(len(hvac_params['SETPOINTS'])))

    for setpoint_index in range(len(hvac_params['SETPOINTS'])):

        epoch_temperature_degree = get_epoch_temperature_degree(hvac_params, temperature, setpoint_index)

        aggregated_degree = np.bincount(aggregate_identifier, epoch_temperature_degree)
        valid_idx = np.logical_and(aggregated_degree > 0, aggregated_consumption > 0)

        if min_cdd_quantile > 0:
            # Being Conservative : if setpoint is not detected with full consumption and temperature data
            valid_idx = remove_outlier_days(aggregated_degree, hvac_params, aggregated_consumption, valid_idx)

            low_cdd = 2 * mquantiles(aggregated_degree[valid_idx], min_cdd_quantile, alphap=0.5, betap=0.5)
            low_cons = static_params['min_consumption_for_setpoint']

            valid_idx = np.logical_and(aggregated_degree > low_cdd, aggregated_consumption > low_cons)

        try:

            fit_model_linear = linear_fit(aggregated_degree[valid_idx], aggregated_consumption[valid_idx])
            fit_model_sqrt = linear_fit(np.sqrt(aggregated_degree[valid_idx]), aggregated_consumption[valid_idx])

            fit_model = choose_model_kind(fit_model_linear, fit_model_sqrt)

            fit_coefficients[setpoint_index, :] = [fit_model['Rsquared']['Ordinary'],
                                                   fit_model['Coefficients']['Estimate'][1],
                                                   fit_model['Coefficients']['pValue'][1] / 2. + 0.5 *
                                                   (fit_model['Coefficients']['Estimate'][1] <= 0.),
                                                   fit_model['Coefficients']['tValue'][1], np.sum(valid_idx),
                                                   np.sum(fit_model['Coefficients']['Estimate'][1] * aggregated_degree)]

        except (ValueError, IndexError, KeyError, np.linalg.linalg.LinAlgError):

            logger_hvac.info(
                ' Contender setpoint {} failed to fit linear model |'.format(hvac_params['SETPOINTS'][setpoint_index]))
            continue

    df_setpoint = pd.DataFrame()
    df_setpoint['setpoints'] = hvac_params['SETPOINTS']
    df_fit_coefficients = pd.DataFrame(fit_coefficients,
                                       columns=['r_squared', 'estimate_1', 'p_value', 't_value', 'valid_count',
                                                'degree_day'])
    df_fit_coefficients = pd.concat([df_setpoint, df_fit_coefficients], axis=1)
    df_fit_coefficients['qualified_setpoint'] = df_fit_coefficients['p_value'] < hvac_params['PVAL_THRESHOLD']

    qualified_df = df_fit_coefficients[df_fit_coefficients['qualified_setpoint'] == True]
    logger_hvac.info(' Total qualified setpoints based on p-values : {} |'.format(qualified_df.shape[0]))

    best_setpoint_row = qualified_df['r_squared'] == np.max(qualified_df['r_squared'])

    best_setpoint_row = ensure_single_best(best_setpoint_row)

    degree_day_setpoint, degree_day_rsq, degree_day_pval = get_degree_day_setpoint_metrics(best_setpoint_row,
                                                                                           qualified_df, hvac_params,
                                                                                           df_fit_coefficients,
                                                                                           logger_hvac)
    setpoint_estimate = {'setpoint': degree_day_setpoint, 'rsq': degree_day_rsq, 'pval': degree_day_pval,   'exist': False}
    setpoint_estimate['exist'] = (setpoint_estimate['rsq'] > 0) and (setpoint_estimate['pval'] < hvac_params['PVAL_THRESHOLD'])

    logger_hvac.info(' >> Final setpoint so far: {}, r-square : {}, validity to exist:{} |'.format(degree_day_setpoint,
                                                                                                   degree_day_rsq,
                                                                                                   setpoint_estimate[
                                                                                                       'exist']))

    if not setpoint_estimate['exist'] and min_cdd_quantile > 0 and detection_debug['mu'] > 1000 and not hvac_params['IS_AC']:
        logger_hvac.info(' Setpoint so far is invalid. but epoch amplitude of SH>1000. Setting setpoint as valid |')
        setpoint_estimate['exist'] = True

    if not setpoint_estimate['exist'] and min_cdd_quantile == 0:
        logger_hvac.info(' Unable to find valid setpoint. Retrying with tighter quantile of 0.4 |')
        adjusted_disagg_output_object = copy.deepcopy(disagg_output_object)
        adjusted_disagg_output_object['switch']['hvac']['hour_aggregate_level'] = static_params[
            'fallback_hour_aggregate']

        common_objects['disagg_output_object'] = adjusted_disagg_output_object

        logger_hvac.info(' >>>>>>>>>> Fallback setpoint estimate <<<<<<<<<<< |')
        setpoint_estimate = \
            setpoint_by_mode_at_x_hour(detection_debug, day_idx, hvac_params, static_params['fallback_cdd_quantile'],
                                       logger_base, common_objects)[0]

    return setpoint_estimate, hvac_exit_status
