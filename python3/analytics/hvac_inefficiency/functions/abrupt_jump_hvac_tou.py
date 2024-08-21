"""
Author - Anand Kumar Singh
Date - 12th March 2021
Function to plot multiple  heatmaps
"""

# Import python packages

import copy
import numpy as np

from sklearn.linear_model import LogisticRegression

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params
from python3.analytics.hvac_inefficiency.utils.continuous_consumption import get_continuous_high_consumption

RANDOM_STATE = 43


def local_outlier_detection(hvac_consumption, hvac_temperature, cut_off_probability_higher=0.05, cut_off_probability_lower=0.3, min_limit=0.15,
                            window_size=28, step=14, exclusion_col=np.empty(0, )):
    """
    Function to get outlier

    Parameters:
        hvac_consumption            (np.ndarray): Array of hvac consumption
        hvac_temperature            (np.ndarray): Array of temperature
        cut_off_probability_higher  (float)     : High probability cut off
        cut_off_probability_lower   (float)     : Low probability cut off
        min_limit                   (float)     : Minimum limit parameter
        window_size                 (int)       : window size parameter
        step                        (int)       : Step parametr
        exclusion_col               (np.ndarray):Array containing exclusion

    Returns:
        outlier_days_local          (np.ndarray): Array of outlier days
        outlier_days_local_prob     (np.ndarray): Array of outlier days probability

    """

    # Creating zero like outliers
    outlier_days_local = np.zeros(hvac_consumption.shape)
    outlier_days_local_prob = np.zeros(hvac_consumption.shape)

    if hvac_consumption[hvac_consumption > 0].shape[0] == 0:
        return outlier_days_local, outlier_days_local_prob

    min_limit_hvac = np.percentile(hvac_consumption[hvac_consumption > 0], min_limit * 100)

    total_number_of_rows = hvac_consumption.shape[0]

    for row_start in range(0, total_number_of_rows, step):
        local_outlier, local_outlier_prob =\
            outlier_detection_basic(hvac_consumption[row_start:(row_start + window_size)],
                                    hvac_temperature[row_start:(row_start + window_size)],
                                    cut_off_probability_lower=cut_off_probability_lower,
                                    cut_off_probability_higher=cut_off_probability_higher,
                                    min_limit_hvac=min_limit_hvac, region='local', exclusion_col=exclusion_col)

        outlier_days_local[row_start:(row_start + window_size)] += local_outlier
        outlier_days_local_prob[row_start:(row_start + window_size)] += local_outlier_prob

    return outlier_days_local, outlier_days_local_prob


def find_cons_outlier(consumption, offset=1.2, deviation=2):

    """
    Function to find consumption outlier

    Parameters:
        consumption     (np.ndarray): Array of consumptions
        offset          (float)     : Offset parameter
        deviation       (int)       : Deviation parameter

    Returns:
        outlier_index   (np.ndarray): Array of outlier indexes
    """
    threshold = (offset * np.nanmedian(consumption)) + (np.nanstd(consumption) * deviation)
    outlier_index = consumption > threshold

    return outlier_index


def find_local_and_global_outliers(total_consumption_col):

    """
    Function to get loccal and global outliers

    Parameters:
        total_consumption_col   (np.ndarray): Array of total consumption
    Returns:
        global_outlier_index    (np.ndarray): Array of global outlier index
        local_outlier_index     (np.ndarray): Array of local outlier index
    """

    static_params = hvac_static_params()

    total_number_of_rows = len(total_consumption_col)
    step = static_params.get('ineff').get('abrupt_tou_step')
    window_size = static_params.get('ineff').get('abrupt_tou_window')
    local_outlier_array = np.zeros_like(total_consumption_col)

    for row_start in range(0, total_number_of_rows, step):
        local_outlier_temp =\
            find_cons_outlier(total_consumption_col[row_start: row_start + window_size], offset=1, deviation=2)
        local_outlier_array[row_start: row_start + window_size][local_outlier_temp] += 1

    global_outlier_index = find_cons_outlier(total_consumption_col)

    local_outlier_index = (local_outlier_array >= static_params.get('ineff').get('local_outlier_lim'))

    return global_outlier_index, local_outlier_index


def get_consumption_outlier_tou(total_consumption_array):

    """
    Function to get consumption outlier tou

    Parameters:
        total_consumption_array (np.ndarray)    : Array containing total consumption

    Returns:
        all_consumption_outlier (np.ndarray)    : Array of outliers
    """

    all_local_outliers = np.zeros_like(total_consumption_array)
    all_global_outliers = np.zeros_like(total_consumption_array)

    for col in range(0, total_consumption_array.shape[1]):
        global_outlier_single, local_outlier_single = find_local_and_global_outliers(total_consumption_array[:, col])
        all_global_outliers[global_outlier_single, col] = 1
        all_local_outliers[local_outlier_single, col] = 1

    all_consumption_outlier = all_global_outliers + all_local_outliers

    unrolled_data = all_consumption_outlier.ravel()
    unrolled_data = get_continuous_high_consumption(unrolled_data, fill_value=0, length=4, min_value=0)
    axis_dimension = min(len(unrolled_data), Cgbdisagg.HRS_IN_DAY)
    all_consumption_outlier = unrolled_data.reshape(-1, axis_dimension)

    return all_consumption_outlier


def get_probabilities(col, higher_values, cut_off_probability_higher, retrain_probablity_list,
                      cut_off_probability_lower):
    """
    Function to get cut off and retrain probabilities

    Parameters:

        col                         : Column
        higher_values               : Higher_values
        cut_off_probability_higher  : Probability cut off
        retrain_probablity_list     : Probability retrain
        cut_off_probability_lower   : Probability lower

    Return:
        cut_off_probability         : Probability cut off
        retrain_probability         : Probability retrain
    """

    if col in higher_values:
        cut_off_probability = cut_off_probability_higher
        retrain_probability = retrain_probablity_list[0]
    else:
        cut_off_probability = cut_off_probability_lower
        retrain_probability = retrain_probablity_list[1]

    return cut_off_probability, retrain_probability


def get_outlier_days_prob(outlier_days_all, outlier_days, outlier_day_probability_all, outlier_day_probability):

    """
    Function to return outlier probabilities

    Parameters:
        outlier_days_all            (np.ndarray)        : Outlier days all
        outlier_days                (np.ndarray)        : Outlier days
        outlier_day_probability_all (np.ndarray)        : Outlier days Prob all
        outlier_day_probability     (np.ndarray)        : Outlier days Prob

    Returns:
        outlier_days_all            (np.ndarray)        : Outlier days all
        outlier_day_probability_all (np.ndarray)        : Outlier days Prob all
    """

    if outlier_days_all.shape[0] > 0:
        outlier_days_all = np.c_[outlier_days_all, outlier_days]
        outlier_day_probability_all = np.c_[outlier_day_probability_all, outlier_day_probability]
    else:
        outlier_days_all = copy.deepcopy(outlier_days)
        outlier_day_probability_all = copy.deepcopy(outlier_day_probability)

    return outlier_days_all, outlier_day_probability_all


def outlier_detection_basic(hvac_consumption, hvac_temperature, retrain_probablity_list=[0.6, 0.6],
                            cut_off_probability_higher=0.05, cut_off_probability_lower=0.3, min_limit=0.15,
                            min_limit_hvac=None, region='local', exclusion_col=np.empty(0, )):
    """
    Function with outlier detection basic

    Parameters:
        hvac_consumption            (np.ndarray): Array of hvac consumption
        hvac_temperature            (np.ndarray): Array containing temperature
        retrain_probablity_list     (list)      : Probability list
        cut_off_probability_higher  (float)     : High cut off probability
        cut_off_probability_lower   (float)     : Low cut off probability
        min_limit                   (float)     : Minimum limit parameter
        min_limit_hvac              (float)     : Minimum limit hvac
        region                      (str)       : String indicating region
        exclusion_col               (np.ndarray): Array containing exclusions

    Returns:
        outlier_days_all            (np.ndarray): Array of outlier days
        outlier_day_probability_all (np.ndarray): Probability
    """

    static_params = hvac_static_params()

    if len(exclusion_col) == 0:
        exclusion_col = np.zeros_like(hvac_consumption[0, :])

    if hvac_consumption[hvac_consumption > 0].shape[0] == 0:
        outlier_days_all = np.zeros_like(hvac_consumption, dtype=bool)
        outlier_day_probability_all = np.zeros_like(hvac_consumption)
        return outlier_days_all, outlier_day_probability_all

    if min_limit_hvac is None:
        min_limit_hvac = np.percentile(hvac_consumption[hvac_consumption > 0], 100 * min_limit)

    hvac_consumption[hvac_consumption > min_limit_hvac] = 1

    hvac_distribution = np.count_nonzero(hvac_consumption, axis=0)
    median = np.percentile(hvac_distribution, static_params.get('ineff').get('outlier_det_med_pct'))

    distance_from_curve = (hvac_distribution - median)

    filter_criterion = distance_from_curve > 0
    higher_values = np.argwhere(filter_criterion == True)

    # Existing Code

    hvac_consumption_present = copy.deepcopy(hvac_consumption)

    hvac_consumption_present[hvac_consumption_present > 0] = 1

    outlier_days_all = np.empty(0, )
    outlier_day_probability_all = np.empty(0, dtype=np.float)

    min_temp_arr = np.nanmin(hvac_temperature, axis=0)
    max_temp_arr = np.nanmax(hvac_temperature, axis=0)
    norm_temp = np.divide(hvac_temperature - min_temp_arr, (max_temp_arr - min_temp_arr) +
                          static_params.get('ineff').get('norm_temp_epsilon'))

    for col in range(0, hvac_consumption.shape[1]):

        exclude_col = exclusion_col[col]

        cut_off_probability, retrain_probability = get_probabilities(col, higher_values, cut_off_probability_higher,
                                                                     retrain_probablity_list, cut_off_probability_lower)

        min_temperature = min_temp_arr[col]
        max_temperature = max_temp_arr[col]

        hvac = hvac_consumption_present[:, col]
        outlier_days = np.zeros_like(hvac)
        outlier_day_probability = np.zeros_like(hvac, dtype=np.float)

        if (min_temperature != max_temperature) and (exclude_col == False):

            temperature = norm_temp[:, col]

            # Find list of index with valid temperature and HVAC
            valid_idx = ~(np.isnan(temperature) | np.isnan(hvac))
            fit_temperature = temperature[valid_idx].reshape(-1, 1)
            fit_hvac = hvac[valid_idx]

            if len(np.unique(fit_hvac)) > 1:

                clf = LogisticRegression(random_state=RANDOM_STATE).fit(fit_temperature, fit_hvac)

                # Create Numpy Array with Nans to store probability
                output_probability = np.empty(shape=(temperature.shape[0], 2), dtype=np.float)
                output_probability[:] = np.nan

                output_probability[valid_idx] = clf.predict_proba(fit_temperature)
                retrain_idx = ~((output_probability[:, 1] > retrain_probability) & (hvac == 0))
                retrain_idx = (retrain_idx & valid_idx)

                if region == 'local':
                    outlier_days = ((output_probability[:, 1] < cut_off_probability) & (hvac == 1))
                    outlier_day_probability[outlier_days] = static_params.get('ineff').get('outlier_prob_pivot') - output_probability[outlier_days, 1]

                if len(np.unique(hvac[retrain_idx])) > 1:
                    clf = LogisticRegression(random_state=RANDOM_STATE).fit(temperature[retrain_idx].reshape(-1, 1),
                                                                            hvac[retrain_idx])
                    output_probability_new = np.empty(shape=(temperature.shape[0], 2), dtype=np.float)
                    output_probability_new[:] = np.nan

                    output_probability_new[valid_idx] = clf.predict_proba(fit_temperature)
                    outlier_days = ((output_probability_new[:, 1] < cut_off_probability) & (hvac == 1))
                    outlier_day_probability[outlier_days] = static_params.get('ineff').get('outlier_prob_pivot') - output_probability_new[outlier_days, 1]

        outlier_days_all, outlier_day_probability_all = get_outlier_days_prob(outlier_days_all, outlier_days,
                                                                              outlier_day_probability_all,
                                                                              outlier_day_probability)

    outlier_days_all = outlier_days_all.astype(np.float)

    return outlier_days_all, outlier_day_probability_all


def get_tou_outliers(input_hvac_inefficiency_object, output_hvac_inefficiency_object, logger_pass=None, device=None):

    """
    Function to get tou outliers

    Parameters:
        input_hvac_inefficiency_object  (dict)  : Dictionary containing input hvac inefficiency
        output_hvac_inefficiency_object (dict)  : Dictionary containing output hvac inefficiency
        logger_pass                     (object): Logging object
        device                          (str)   : String identifying device

    Returns:
        input_hvac_inefficiency_object  (dict)  : Dictionary containing input hvac inefficiency
        output_hvac_inefficiency_object (dict)  : Dictionary containing output hvac inefficiency
    """

    static_params = hvac_static_params()

    cooling_consumption = copy.deepcopy(input_hvac_inefficiency_object.get('ac').get('demand_hvac_pivot').get('values'))
    column_idx = input_hvac_inefficiency_object.get('ac').get('demand_hvac_pivot').get('columns')

    heating_consumption = copy.deepcopy(input_hvac_inefficiency_object.get('sh').get('demand_hvac_pivot').get('values'))

    total_consumption = copy.deepcopy(input_hvac_inefficiency_object.get('sh').get('energy_pivot').get('values'))

    temperature_heatmap_cooling = copy.deepcopy(
        input_hvac_inefficiency_object.get('ac').get('temperature_pivot').get('values'))

    temperature_heatmap_heating = copy.deepcopy(
        input_hvac_inefficiency_object.get('sh').get('temperature_pivot').get('values'))
    index = copy.deepcopy(input_hvac_inefficiency_object.get('sh').get('temperature_pivot').get('row'))

    # Getting consumption outlier as numpy array
    net_consumption_outliers = get_consumption_outlier_tou(total_consumption_array=total_consumption)

    columns_without_cons_outlier = np.sum(net_consumption_outliers, axis=0)
    columns_without_cons_outlier = columns_without_cons_outlier == 0

    net_consumption_outliers_idx = net_consumption_outliers.sum(axis=1) > 0

    # If no consumption outlier than exist code
    if net_consumption_outliers_idx.sum() == 0:
        final_outliers = np.zeros_like(total_consumption, dtype=np.float)
        hvac_outlier_hours_local = np.zeros_like(total_consumption, dtype=np.float)
        hvac_outlier_hours_global = np.zeros_like(total_consumption, dtype=np.float)

        abrupt_tou_change = {
            'final_outlier_days': final_outliers,
            'total_consumption': total_consumption,
            'local_outliers': hvac_outlier_hours_local,
            'global_outliers': hvac_outlier_hours_global,
            'consumption_outlier': net_consumption_outliers,
            'ac': cooling_consumption,
            'sh': heating_consumption,
            'row': index,
            'columns': column_idx
        }

        output_hvac_inefficiency_object['abrupt_tou_change'] = abrupt_tou_change

        return input_hvac_inefficiency_object, output_hvac_inefficiency_object

    # Cut off probability
    low_prob = static_params.get('ineff').get('tou_outlier_low_prob')
    high_prob = static_params.get('ineff').get('tou_outlier_high_prob')

    hvac_consumption = heating_consumption - cooling_consumption

    _, cooling_outlier_hours_global = outlier_detection_basic(cooling_consumption,
                                                              temperature_heatmap_cooling,
                                                              cut_off_probability_lower=low_prob,
                                                              cut_off_probability_higher=high_prob,
                                                              region='global',
                                                              exclusion_col=columns_without_cons_outlier)

    _, heating_outlier_hours_global = outlier_detection_basic(heating_consumption,
                                                              temperature_heatmap_heating,
                                                              cut_off_probability_lower=low_prob,
                                                              cut_off_probability_higher=high_prob, region='global',
                                                              exclusion_col=columns_without_cons_outlier)

    cooling_outlier_hours_global = cooling_outlier_hours_global * (-1)
    hvac_outlier_hours_global = cooling_outlier_hours_global + heating_outlier_hours_global

    _, cooling_outlier_hours_local = local_outlier_detection(cooling_consumption,
                                                             temperature_heatmap_cooling,
                                                             cut_off_probability_lower=low_prob,
                                                             cut_off_probability_higher=high_prob,
                                                             window_size=static_params.get('ineff').get('outlier_window_size'),
                                                             exclusion_col=columns_without_cons_outlier)

    _, heating_outlier_hours_local = local_outlier_detection(heating_consumption,
                                                             temperature_heatmap_heating,
                                                             cut_off_probability_lower=low_prob,
                                                             cut_off_probability_higher=high_prob,
                                                             window_size=static_params.get('ineff').get('outlier_window_size'),
                                                             exclusion_col=columns_without_cons_outlier)

    cooling_outlier_hours_local = cooling_outlier_hours_local * (-1)
    hvac_outlier_hours_local = cooling_outlier_hours_local + heating_outlier_hours_local

    hvac_outlier_hours_local = hvac_outlier_hours_local / 2

    hvac_outlier_hours_combined = hvac_outlier_hours_global + hvac_outlier_hours_local

    hvac_consumption[hvac_consumption > 0] = 1
    hvac_consumption[hvac_consumption < 0] = -1

    final_outlier_limit = static_params.get('ineff').get('final_outlier_lim')
    hvac_outlier_hours_combined[hvac_outlier_hours_combined > final_outlier_limit] = 1
    hvac_outlier_hours_combined[hvac_outlier_hours_combined < - final_outlier_limit] = -1

    hvac_outlier_hours_combined[np.abs(hvac_outlier_hours_combined) < final_outlier_limit] = 0

    # Outliers masking with consumption outliers
    hvac_outlier_hours_combined = net_consumption_outliers * hvac_outlier_hours_combined

    # Filtering less than 3 hour a day outlier
    large_negative_values = static_params.get('ineff').get('large_neg_value')
    unrolled_data = hvac_outlier_hours_combined.reshape(1, -1)[0]
    zero_idx = (unrolled_data == 0)
    unrolled_data[zero_idx] = large_negative_values

    unrolled_data =\
        get_continuous_high_consumption(unrolled_data, fill_value=0, length=4, min_value=large_negative_values)
    final_outliers = unrolled_data.reshape(-1 * Cgbdisagg.HRS_IN_DAY, Cgbdisagg.HRS_IN_DAY)

    abrupt_tou_change = {
        'final_outlier_days': final_outliers,
        'total_consumption': total_consumption,
        'local_outliers': hvac_outlier_hours_local,
        'global_outliers': hvac_outlier_hours_global,
        'consumption_outlier': net_consumption_outliers,
        'ac': cooling_consumption,
        'sh': heating_consumption,
        'row': index,
        'columns': column_idx
    }

    output_hvac_inefficiency_object['abrupt_tou_change'] = abrupt_tou_change

    return input_hvac_inefficiency_object, output_hvac_inefficiency_object
