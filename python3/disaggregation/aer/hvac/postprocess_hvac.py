"""
Author - Abhinav Srivastava / Mirambika Sikdar
Date - 06/12/23
Call the hvac postprocess module and get processed results for Historical / Incremental modes
"""

# Import python packages
import copy
import scipy
import numpy as np
import pandas as pd

# Import functions from within the project
from python3.utils.find_runs import find_runs
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.hvac_utility import max_min_without_inf_nan
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params
from python3.disaggregation.aer.hvac.post_process_consumption import binary_search_for_cut_level


def get_guess_in_overestimation(hvac_contenders_in_month, appliance_df, hvac, required_reduction, residual_to_meet,
                                month_base_residue):
    """
    Function to get the cut point, below which hvac estimates are moved to residue from hvac

    Parameters:

        hvac_contenders_in_month    (np.ndarray)        : Array containing booleans of qualified entries w.r.t ac/sh
        appliance_df                (pd.Dataframe)      : Dataframe containing hvac consumption
        hvac                        (str)               : String identifier of whether hvac is ac or sh
        required_reduction          (float)             : The required amount of reduction to handle overestimation
        residual_to_meet            (float)             : The desired residual in month
        month_base_residue          (float)             : The original residue from month

    Returns:
        guess                       (float)             : The cut point, below which hvac estimates are moved to residue from hvac
        valid_consumption           (np.ndarray)        : Array containing booleans where consumption is greater than guess
    """

    # initializing hvac array to be modified
    concerned_hvac_array = np.zeros((len(hvac_contenders_in_month)))
    concerned_hvac_array[hvac_contenders_in_month] = appliance_df[hvac][hvac_contenders_in_month]

    concerned_hvac_array_sorted = np.sort(concerned_hvac_array)

    # getting non zero valid hvac points to be adjusted
    non_zero_concerned_hvac_array_sorted = concerned_hvac_array_sorted[concerned_hvac_array_sorted > 0]
    non_zero_concerned_hvac_array_sorted_diff = (np.diff(non_zero_concerned_hvac_array_sorted.T) != 0).astype(int)
    non_zero_concerned_hvac_array_sorted_diff = np.r_[0, non_zero_concerned_hvac_array_sorted_diff]
    non_zero_concerned_hvac_array_sorted_diff_idx = np.argwhere(non_zero_concerned_hvac_array_sorted_diff == 1)
    non_zero_concerned_hvac_array_sorted_diff_idx = np.r_[0, non_zero_concerned_hvac_array_sorted_diff_idx[:, 0]]

    # initializing guess for reduction
    guesses = []
    guesses_idx = []
    residues_from_hvac = []
    non_zero_array_length = len(non_zero_concerned_hvac_array_sorted)

    # getting guess array
    for idx in non_zero_concerned_hvac_array_sorted_diff_idx:
        guess = non_zero_concerned_hvac_array_sorted[idx]
        guesses.append(guess)
        guesses_idx.append(idx)

        residue_from_hvac = guess * (non_zero_array_length - idx) / Cgbdisagg.WH_IN_1_KWH
        residues_from_hvac.append(residue_from_hvac)

    distance_from_required_residue = np.array(residues_from_hvac) - required_reduction

    # getting best guess
    try:

        best_guess_bet = np.min(distance_from_required_residue[distance_from_required_residue > 0])
        best_guess_bet_location = np.argwhere(distance_from_required_residue == best_guess_bet)[0][0]

        best_guess_idx = guesses_idx[best_guess_bet_location]
        best_guess = non_zero_concerned_hvac_array_sorted[best_guess_idx]

        valid_consumption = appliance_df[hvac] >= best_guess
        residue_contender_in_hvac = appliance_df[hvac][hvac_contenders_in_month & valid_consumption]

        guess = (residual_to_meet - month_base_residue) * Cgbdisagg.WH_IN_1_KWH / residue_contender_in_hvac.shape[0]

    except (ValueError, IndexError, KeyError):

        # fail safe of best guess
        best_guess_bet = np.max(distance_from_required_residue[distance_from_required_residue < 0])
        best_guess_bet_location = np.argwhere(distance_from_required_residue == best_guess_bet)[0][0]

        best_guess_idx = guesses_idx[best_guess_bet_location]
        best_guess = non_zero_concerned_hvac_array_sorted[best_guess_idx]

        valid_consumption = appliance_df[hvac] >= best_guess

        guess = best_guess

    return valid_consumption, guess


def get_residue_from_hvac(overestimated_months, epoch_hvac_contenders, residual, hvac_months, hvac, info_carrier):
    """
    Function to extract residue out of hvac estimates in case of Overestimation

    Parameters:
        overestimated_months        (np.ndarray)      : Array containing boolean of underestimated months
        epoch_hvac_contenders       (np.ndarray)      : Array containing booleans of qualified entries w.r.t ac/sh
        residual                    (np.ndarray)      : Array containing epoch level residue information
        hvac_months                 (np.ndarray)      : Month boolean for valid AC/SH
        hvac                        (str)             : String to identify AC or SH
        info_carrier                (dict)            : Dictionary containing general data required for this function
        appliance                   (str)             : String identifier for AC/SH

    Returns:
        residue_from_hvac_array     (np.ndarray)      : Array containg residue extracted out of hvac appliance
    """

    # Read main components from info carrier
    appliance_df = info_carrier.get('appliance_df')
    epoch_input_data = info_carrier.get('epoch_input_data')
    residual_to_meet = info_carrier.get('residual_to_meet')
    month_epoch = info_carrier.get('month_epoch')
    if hvac == 'ac':
        dd_months = info_carrier.get('month_cdd')
    else:
        dd_months = info_carrier.get('month_hdd')

    # Initialise residue to be extracted out of hvac estimates at epoch level
    residue_from_hvac_array = np.zeros((len(appliance_df['residue']), 1))

    # Identify overestimated months
    hvac_overestimated_months_bool = overestimated_months & hvac_months

    # Avoid overestimation flag for peak summer month
    if hvac == 'ac':
        peak_month = np.argmax(dd_months)
        hvac_overestimated_months_bool[peak_month] = False

    hvac_overestimated_months = month_epoch[hvac_overestimated_months_bool]

    # looping on overestimated months to adjust epoch level hvac estimates
    for month in hvac_overestimated_months:

        # identifying hvac contenders in month
        month_rows = (epoch_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == month).astype(bool)
        hvac_contenders_in_month = month_rows & epoch_hvac_contenders

        # getting required reduction and max possible reduction of hvac in month
        month_base_residue = residual[month_epoch == month]
        required_reduction = residual_to_meet - month_base_residue
        max_possible_reduction_in_month = np.sum(appliance_df[hvac][hvac_contenders_in_month]) / Cgbdisagg.WH_IN_1_KWH

        # fail safe for max possible reduction less than zero
        if max_possible_reduction_in_month <= 0:
            guess = 0
            valid_consumption = appliance_df[hvac] >= guess
            residue_from_hvac_array[hvac_contenders_in_month & valid_consumption] = guess
            continue

        # adjusting hvac estimates when max reduction is less than required reduction
        if max_possible_reduction_in_month < required_reduction:

            residue_from_hvac_array[hvac_contenders_in_month] = appliance_df[hvac][
                hvac_contenders_in_month].values.reshape(residue_from_hvac_array[hvac_contenders_in_month].shape)
            appliance_df[hvac][hvac_contenders_in_month] = 0

        # adjusting hvac estimates when max reduction is more than required reduction
        else:

            valid_consumption, guess = get_guess_in_overestimation(hvac_contenders_in_month, appliance_df, hvac,
                                                                   required_reduction, residual_to_meet,
                                                                   month_base_residue)

            appliance_df[hvac][hvac_contenders_in_month & valid_consumption] = appliance_df[hvac][
                                                                                   hvac_contenders_in_month & valid_consumption] - guess

            residue_from_hvac_array[hvac_contenders_in_month & valid_consumption] = guess

    return residue_from_hvac_array


def get_month_hvac_lim(hvac_amplitude_limits, static_params, epochs_per_hour, appliance, low_consumption_hvac_flag):
    """
    Function to get monthly hvac lower limit for extraction from residue
    Arguments:
        hvac_amplitude_limits      (tuple)  : Tuple of lower and upper limits of AC/SH modes
        static_params              (dict)   : Dictionary of pre-defined post-processing thresholds
        epochs_per_hour            (int)    : Integer number of epochs per hour
        appliance                  (str)    : String identifier for ac/sh
        low_consumption_hvac_flag  (int)    : Integer flag for low consumption ac/sh appliance user
    Returns:
        month_reduction_lim        (float)  : Minimum monthly hvac amplitude to be extracted from residue
    """
    if appliance == 'ac':
        # Liberal threshold for maximum possible reduction for cooling to include range < lower limit of ac
        if low_consumption_hvac_flag and not np.isnan(hvac_amplitude_limits[0]):
            month_reduction_lim = np.max(
                [static_params.get('month_reduction_lim'), hvac_amplitude_limits[0] * epochs_per_hour *
                 static_params.get('min_total_hours_ac_low_amplitude')]) / Cgbdisagg.WH_IN_1_KWH
        else:
            lower_limit_hvac = hvac_amplitude_limits[0] * epochs_per_hour * static_params.get('min_total_hours_ac') / Cgbdisagg.WH_IN_1_KWH
            month_reduction_lim = np.nanmax([static_params.get('month_reduction_lim'), lower_limit_hvac])
    else:
        month_reduction_lim = static_params.get('month_reduction_lim')

    return month_reduction_lim


def get_hvac_from_residue(underestimated_months, epoch_hvac_contenders, residual, epoch_ao_hvac_res_net, hvac_months,
                          appliance, info_carrier):
    """
    Function to extract hvac component out of residue, in case of underestimation

    Parameters:

        underestimated_months (bool)            : Array containing boolean of underestimated months
        epoch_hvac_contenders (np.ndarray)      : Array containing booleans of qualified entries w.r.t ac/sh
        residual              (np.ndarray)      : Array containing epoch level residue information
        epoch_ao_hvac_res_net (np.ndarray)      : Array with epoch level AO, AC, SH, Residue, Net consumption estimates
        hvac_months           (np.ndarray)      : Month boolean for valid AC/SH
        appliance             (str)             : String identifier for AC/SH
        info_carrier          (dict)            : Dictionary containing general data required for this function

    Returns:
        hvac_from_residue     (np.ndarray)      : Array containing hvac at epoch level, extracted out of residue
    """

    static_params = hvac_static_params()

    # reading main components from info carrier
    column_index = info_carrier.get('column_index')
    epoch_residue = epoch_ao_hvac_res_net[:, column_index.get('residue')]
    epoch_input_data = info_carrier.get('epoch_input_data')
    residual_to_meet = info_carrier.get('residual_to_meet')
    month_epoch = info_carrier.get('month_epoch')
    epochs_per_hour = info_carrier.get('epochs_per_hour')
    low_consumption_hvac_flag = info_carrier['low_consumption_ac_flag']
    if appliance == 'ac':
        hvac_amplitude_limits = info_carrier['cooling_amplitude_limits']
    else:
        hvac_amplitude_limits = info_carrier['heating_amplitude_limits']

    # initializing hvac to be extracted out of residue
    hvac_from_residue = np.zeros((len(epoch_residue), 1))

    # identifying hvac underestimated months
    hvac_underestimated_months_bool = underestimated_months & hvac_months
    hvac_underestimated_months = month_epoch[hvac_underestimated_months_bool]

    # based on maximum possible reduction in hvac, determining reduction required
    month_reduction_lim = get_month_hvac_lim(hvac_amplitude_limits, static_params, epochs_per_hour, appliance, low_consumption_hvac_flag)

    # looping over underestimated months to adjust underestimated hvac estimates
    for _, month in enumerate(hvac_underestimated_months):
        # identifying month epochs and month hvac estimates
        month_rows = (epoch_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == month).astype(bool)
        hvac_contenders_in_month = month_rows & epoch_hvac_contenders

        # getting required reduction at month level
        month_base_residue = residual[month_epoch == month]
        required_reduction = month_base_residue - residual_to_meet
        max_possible_reduction_in_month = np.sum(epoch_residue[hvac_contenders_in_month]) / Cgbdisagg.WH_IN_1_KWH

        if (max_possible_reduction_in_month <= month_reduction_lim) | (required_reduction <= month_reduction_lim):
            required_reduction = 0

        # getting fail-safe cut levels
        if required_reduction <= 0:
            guess = np.max(epoch_residue) + 1
            consumption_more_than_guess = epoch_residue > guess

        elif max_possible_reduction_in_month < required_reduction:
            guess = 0
            consumption_more_than_guess = epoch_residue > guess

        else:
            # getting valid cut levels on consumption towers
            guess, consumption_more_than_guess = binary_search_for_cut_level(epoch_residue, hvac_contenders_in_month,
                                                                             month_base_residue, residual_to_meet)

        # modifying residual array
        epoch_residue[hvac_contenders_in_month & consumption_more_than_guess] = guess

        change_idx = hvac_contenders_in_month & consumption_more_than_guess
        hvac_from_residue[change_idx] = np.fmax(np.array(epoch_residue[change_idx]) - guess,
                                                0).reshape(hvac_from_residue[change_idx].shape)

    return hvac_from_residue


def remove_fp_hvac(info_carrier, column_index, epochs_per_hour, disagg_output_object, appliance):
    """
    Function to remove false positives of hvac

    Parameters:
        info_carrier          (dict)            : Dictionary containing general data required for this function
        column_index                (dict)          : Dictionary containing column identification indexes in hvac arrays
        epochs_per_hour             (int)           : number of epochs per hour
        disagg_output_object        (dict)          : Contains user specific key input payload
        appliance                   (str)           : String identifying ac or sh

    Returns:
        residue_from_hvac_fp        (np.ndarray)    : Array containing epoch level FP hvac values to be added back to residue
    """
    # Array containing epoch level hvac and ao consumptions
    epoch_ao_hvac_true = info_carrier.get('epoch_ao_hvac_true')
    # Array containing month level hvac and ao consumptions
    month_ao_hvac_res_net = info_carrier.get('month_ao_hvac_res_net')
    # Array containing epoch level input data consumptions
    epoch_input_data = info_carrier.get('epoch_input_data')
    # bill cycle start index
    month_idx = info_carrier.get('month_idx')
    # Array containing scaling factor for number of days in each BC
    days_in_bc_scaled = info_carrier.get('days_in_bc_scaled')
    # Boolean identifier of low consumption range user
    if appliance == 'ac':
        low_consumption_hvac_flag = info_carrier.get('low_consumption_ac_flag')
    else:
        low_consumption_hvac_flag = info_carrier.get('low_consumption_sh_flag')
    # reading hvac static parameters
    static_params = hvac_static_params()

    # Read kwh low limits in month for ac or sh

    # Subtract AO cooling and heating from AC/SH to apply monthly KWH total hvac limits only for on-demand
    if appliance == 'ac':
        # Check whether it is low consumption amplitude user and use corresponding limit
        if low_consumption_hvac_flag == 0:
            # The limits are also scaled for 30 days
            month_low_limit = static_params.get('month_low_kwh_ac') * days_in_bc_scaled - \
                              (disagg_output_object['ao_seasonality']['cooling'] / Cgbdisagg.WH_IN_1_KWH)
            month_low_limit[month_low_limit < 0] = 0
            min_total_hours = static_params.get('min_total_hours_ac') * epochs_per_hour
        else:
            # The limits are also scaled for 30 days
            month_low_limit = static_params.get('month_low_kwh_ac_low_amplitude') * days_in_bc_scaled
            min_total_hours = static_params.get('min_total_hours_ac_low_amplitude') * epochs_per_hour
        min_streaks = static_params.get('min_streaks_ac')
    else:
        month_low_limit = static_params.get('month_low_kwh_sh')
        min_total_hours = static_params.get('min_total_hours_sh') * epochs_per_hour
        min_streaks = static_params.get('min_streaks_sh')

    # identifying hvac false positive months
    fp_months = month_ao_hvac_res_net[:, 0][(month_ao_hvac_res_net[:, column_index[appliance]] > 0) &
                                            (month_ao_hvac_res_net[:, column_index[appliance]] <= month_low_limit)]

    # Calculate bill cycle level total consumption duration and number of streak (start, end) patterns
    # This is done to identify FP billing cycles
    app_valid_epochs = np.where((epoch_ao_hvac_true[:, column_index[appliance]] > 0), 1, 0)
    app_monthly_total = np.bincount(month_idx, app_valid_epochs)
    app_monthly_valid_idx = np.bincount(month_idx, app_valid_epochs) >= min_total_hours
    s = []
    for i, month in enumerate(np.unique(month_idx)):
        monthly_output_bool = np.where(epoch_ao_hvac_true[month_idx == month, column_index[appliance]] > 0, 1, 0)
        change_indices = np.nonzero(np.diff(monthly_output_bool) != 0)[0]
        segments = [segment for segment in np.split(monthly_output_bool, change_indices + 1) if segment[0] != 0]
        app_monthly_valid_idx[i] = app_monthly_valid_idx[i] & (len(segments) >= min_streaks)
        s.append(len(segments))

    # initializing residues to zero, for extracting fp hvac out of hvac consumption array at epoch level
    app_monthly_valid_idx = app_monthly_valid_idx[month_idx]
    app_monthly_total = app_monthly_total[month_idx]
    residue_from_hvac_fp = np.zeros((epoch_ao_hvac_true.shape[0], 1))

    # if any fp hvac month exists, suppress hvac consumption
    if np.any(fp_months) or (len(app_monthly_valid_idx) != np.sum(app_monthly_valid_idx)):
        # identifying false positive indexes
        fp_indexes = np.isin(epoch_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], fp_months)
        fp_indexes = fp_indexes | (~app_monthly_valid_idx & (app_monthly_total > 0))
        epoch_ao_hvac_true_backup = copy.deepcopy(epoch_ao_hvac_true)
        # assigning fp hvac to residue for suppression
        residue_from_hvac_fp[fp_indexes, 0] = epoch_ao_hvac_true_backup[fp_indexes, column_index.get(appliance)]

    return residue_from_hvac_fp


def remove_week_clashes(epoch_ao_hvac_true, month_ao_hvac_res_net, epoch_input_data, disagg_output_object,
                        column_index, month_idx):
    """
    Function to handle cooling-heating clashes in a week

    Parameters:
        epoch_ao_hvac_true      (np.ndarray)    : Contains epoch level hvac consumption
        month_ao_hvac_res_net   (np.ndarray)
        epoch_input_data        (np.ndarray)    : Contains input data of user
        disagg_output_object    (dict)          : Contains user specific key input payload
        column_index            (dict)

    Returns:
        epoch_ao_hvac_true      (np.ndarray)    : Contains epoch level hvac consumption
        disagg_output_object    (dict)          : Contains user specific key input payload
    """
    # making copy of input consumption data for processing
    input_data_copy = copy.deepcopy(epoch_input_data)

    # getting unique weeks
    unique_weeks = np.unique(input_data_copy[:, Cgbdisagg.INPUT_WEEK_IDX])

    # reading and making copy of hvac consumptions at epoch level
    ac_od_net = copy.deepcopy(epoch_ao_hvac_true[:, column_index.get('ac')])
    sh_od_net = copy.deepcopy(epoch_ao_hvac_true[:, column_index.get('sh')])
    ac_ao_net = copy.deepcopy(disagg_output_object.get('ao_seasonality').get('epoch_cooling'))
    sh_ao_net = copy.deepcopy(disagg_output_object.get('ao_seasonality').get('epoch_heating'))

    # initializing removal of hvac consumptions at epoch level
    ac_od_removed = ac_od_net * 0
    sh_od_removed = sh_od_net * 0
    ac_ao_removed = ac_ao_net * 0
    sh_ao_removed = sh_ao_net * 0

    # handling ac and sh conflicts happening in a week
    for week_idx in range(len(unique_weeks)):

        # identifying current week
        current_week = unique_weeks[week_idx]
        current_week_arr = (input_data_copy[:, Cgbdisagg.INPUT_WEEK_IDX] == current_week).astype(int)

        # identifying current week hvac consumptions
        current_week_ac_od = current_week_arr * ac_od_net
        current_week_sh_od = current_week_arr * sh_od_net
        current_week_ac_ao = current_week_arr * ac_ao_net
        current_week_sh_ao = current_week_arr * sh_ao_net
        current_week_ac = current_week_ac_od + current_week_ac_ao
        current_week_sh = current_week_sh_od + current_week_sh_ao

        # getting total hvac consumption happening in current week
        current_week_ac_total = np.nansum(current_week_ac)
        current_week_sh_total = np.nansum(current_week_sh)

        # handling current week conflict by comparing ac and sh estimates
        if current_week_ac_total >= current_week_sh_total:

            sh_od_net = sh_od_net - current_week_sh_od
            sh_ao_net = sh_ao_net - current_week_sh_ao
            sh_od_removed = sh_od_removed + current_week_sh_od
            sh_ao_removed = sh_ao_removed + current_week_sh_ao

        elif current_week_ac_total < current_week_sh_total:

            ac_od_net = ac_od_net - current_week_ac_od
            ac_ao_net = ac_ao_net - current_week_ac_ao
            ac_od_removed = ac_od_removed + current_week_ac_od
            ac_ao_removed = ac_ao_removed + current_week_ac_ao

        else:

            # move to next week's hvac comparison
            continue

    # assigning modified hvac estimates at epoch level to original hvac array carrier
    epoch_ao_hvac_true[:, column_index.get('ac')] = ac_od_net
    epoch_ao_hvac_true[:, column_index.get('sh')] = sh_od_net
    disagg_output_object['ao_seasonality']['epoch_cooling'] = ac_ao_net
    disagg_output_object['ao_seasonality']['epoch_heating'] = sh_ao_net

    # aggregating epoch level estimate to month level
    month_ao_cool = np.bincount(month_idx, ac_ao_net)
    month_ao_heat = np.bincount(month_idx, sh_ao_net)
    month_od_cool = np.bincount(month_idx, ac_od_net)
    month_od_cool = month_od_cool / Cgbdisagg.WH_IN_1_KWH
    month_od_heat = np.bincount(month_idx, sh_od_net)
    month_od_heat = month_od_heat / Cgbdisagg.WH_IN_1_KWH

    month_ac_od_removed = np.bincount(month_idx, ac_od_removed)
    month_ac_od_removed = month_ac_od_removed / Cgbdisagg.WH_IN_1_KWH
    month_sh_od_removed = np.bincount(month_idx, sh_od_removed)
    month_sh_od_removed = month_sh_od_removed / Cgbdisagg.WH_IN_1_KWH

    month_sh_ao_removed = np.bincount(month_idx, sh_ao_removed)
    month_sh_ao_removed = month_sh_ao_removed / Cgbdisagg.WH_IN_1_KWH
    month_ac_ao_removed = np.bincount(month_idx, ac_ao_removed)
    month_ac_ao_removed = month_ac_ao_removed / Cgbdisagg.WH_IN_1_KWH

    # assigning modified hvac estimates at month level to original hvac array carrier
    disagg_output_object['ao_seasonality']['cooling'] = month_ao_cool
    disagg_output_object['ao_seasonality']['heating'] = month_ao_heat
    month_ao_hvac_res_net[:, column_index.get('ac')] = month_od_cool
    month_ao_hvac_res_net[:, column_index.get('sh')] = month_od_heat

    # assigning modified residue at month level to original hvac array carrier
    std_residual_col = -2
    month_ao_hvac_res_net[:, std_residual_col] = month_ao_hvac_res_net[:, std_residual_col] + month_ac_od_removed + \
                                                 month_sh_od_removed + month_sh_ao_removed + month_ac_ao_removed

    return epoch_ao_hvac_true, month_ao_hvac_res_net, disagg_output_object


def impose_duty_cycle_on_hvac(hvac_consumption, net_consumption):
    """
    Function to impose duty cycle effect on hvac consumption based on net consumption
    Parameters:
        hvac_consumption            (np.ndarray): Array containing hvac consumption at epoch level
        net_consumption             (np.ndarray): Array containing net energy consumption at epoch level

    Returns:
        updated_hvac_consumption    (np.ndarray) : Array containing updated hvac consumption at epoch level
    """

    # finding run starts and their lengths
    run_values, run_starts, run_lengths = find_runs(hvac_consumption)

    # identifying valid length of runs
    valid_idx = (run_lengths > 1) & (run_values != 0)
    valid_start = run_starts[valid_idx]
    valid_lengths = run_lengths[valid_idx]

    # copy of hvac consumption for making update
    updated_hvac_consumption = copy.deepcopy(hvac_consumption)

    # looping over each valid starts to update hvac consumption
    for i in range(0, valid_start.shape[0]):
        start_index = valid_start[i]
        length = valid_lengths[i]
        end_index = start_index + length
        temp_net_consumption_array = net_consumption[start_index:end_index]

        # ensuring division by zero does not happen in update
        if np.nansum(temp_net_consumption_array) == 0:
            updated_hvac_consumption[start_index:end_index] = 0
        else:
            total_hvac = hvac_consumption[start_index:end_index].sum()
            temp_updated_hvac = (temp_net_consumption_array * total_hvac) / temp_net_consumption_array.sum()
            updated_hvac_consumption[start_index:end_index] = temp_updated_hvac

    return updated_hvac_consumption


def avoid_overshoot_net_consumption(epoch_ao_hvac_true, column_index, disagg_input_object,
                                    disagg_output_object):
    """
    Function to restrict HVAC from overshooting net consumption

    Parameters:

        epoch_ao_hvac_true      (np.ndarray)    : Array containing epoch level ao and hvac consumptions
        column_index            (dict)          : dictionary containing column mapping of ao, ac and sh in above arrays
        disagg_input_object     (dict)          : Dictionary containing user level input information
        disagg_output_object    (dict)          : Dictionary containing all output attributes

    Returns:
        epoch_ao_hvac_true      (np.ndarray)    : Array containing modified epoch level ao and hvac consumptions
        month_ao_hvac_true      (np.ndarray)    : Array containing modified month level ao and hvac consumptions
    """
    static_params = hvac_static_params()
    ao_ac = disagg_output_object['ao_seasonality']['epoch_cooling']
    ao_sh = disagg_output_object['ao_seasonality']['epoch_heating']

    ac_consumption = copy.deepcopy(epoch_ao_hvac_true[:, column_index.get('ac')])
    ac_consumption = ac_consumption + ao_ac
    ac_epochs = ac_consumption > 0

    sh_consumption = copy.deepcopy(epoch_ao_hvac_true[:, column_index.get('sh')])
    sh_consumption = sh_consumption + ao_sh
    sh_epochs = sh_consumption > 0

    ao_consumption = epoch_ao_hvac_true[:, column_index.get('ao')]

    input_data = copy.deepcopy(disagg_input_object.get('input_data'))
    net_consumption = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    overshoot_amount = np.round((ac_consumption + sh_consumption + ao_consumption) - net_consumption, 1)
    overshoot_epochs = overshoot_amount > 0

    # handle overshoot by ac
    ac_overshoot_epochs = ac_epochs & overshoot_epochs
    if np.any(ac_overshoot_epochs):
        ac_consumption[ac_overshoot_epochs] = ac_consumption[ac_overshoot_epochs] - overshoot_amount[
            ac_overshoot_epochs]

    # handle overshoot by sh
    sh_overshoot_epochs = sh_epochs & overshoot_epochs
    if np.any(sh_overshoot_epochs):
        sh_consumption[sh_overshoot_epochs] = sh_consumption[sh_overshoot_epochs] - overshoot_amount[
            sh_overshoot_epochs]

    ac_consumption = ac_consumption - ao_ac
    sh_consumption = sh_consumption - ao_sh

    ac_consumption[ac_consumption < 0] = 0
    sh_consumption[sh_consumption < 0] = 0
    ao_consumption[ao_consumption > net_consumption] = net_consumption[ao_consumption > net_consumption]

    month_epoch, _, month_idx = scipy.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                                             return_index=True, return_inverse=True)

    ao_grey = disagg_output_object['ao_seasonality']['grey'] / static_params['kilo']
    net_consumption = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    residual_consumption = net_consumption - (ao_consumption + ac_consumption + sh_consumption)
    month_net = np.bincount(month_idx, net_consumption) / static_params['kilo']
    month_residue = np.bincount(month_idx, residual_consumption) / static_params['kilo'] + ao_grey
    month_ac_consumption = np.bincount(month_idx, ac_consumption) / static_params['kilo']
    month_sh_consumption = np.bincount(month_idx, sh_consumption) / static_params['kilo']
    month_ao_consumption = np.bincount(month_idx, ao_consumption) / static_params['kilo']

    epoch_ao_hvac_true[:, column_index.get('ac')] = ac_consumption
    epoch_ao_hvac_true[:, column_index.get('sh')] = sh_consumption
    epoch_ao_hvac_true[:, column_index.get('ao')] = ao_consumption

    month_ao_hvac_res_net = np.c_[month_epoch, month_ao_consumption, month_ac_consumption, month_sh_consumption,
                                  month_residue, month_net]

    return epoch_ao_hvac_true, month_ao_hvac_res_net


def suppress_ao_seasonal(month_ao_hvac_res_net, disagg_output_object, epoch_input_data,
                         days_in_bc_scaled, month_idx, column_index, appliance, low_consumption_hvac_flag=0):
    """
    Suppress seasonal very low consumption always on component to zero if on demand component is zero
    Args:
        month_ao_hvac_res_net       (np.ndarray)    : Array containing month level ao and hvac consumptions
        disagg_output_object        (dict)          : Dictionary containing all output attributes
        epoch_input_data            (np.ndarray)    : Array containing input epoch level consumption data
        days_in_bc_scaled           (np.ndarray)    : Array containing scaling factor wr30 days for all bill cycles
        month_idx                   (np.ndarray)    : Index array containing month indices
        column_index                (dict)          : Dictionary containing column identifier indices of ao-ac-sh
        appliance                   (str)           : Appliance identifier string
        low_consumption_hvac_flag   (int)           : flag identifier for low consumption users

    returns:
        disagg_output_object        (dict)          : Dictionary containing all output attributes
    """
    static_params = hvac_static_params()
    if appliance == 'ac':
        # Check whether it is low consumption amplitude user and use corresponding scaled 30 days monthly KWH limit
        if low_consumption_hvac_flag == 0:
            month_low_limit = static_params.get('month_low_kwh_ao_ac') * days_in_bc_scaled
        else:
            month_low_limit = static_params.get('month_low_kwh_ao_ac_low_amplitude') * days_in_bc_scaled
    else:
        month_low_limit = static_params.get('month_low_kwh_ao_sh') * days_in_bc_scaled

    # Get epoch seasonal hvac in KWH
    if appliance == 'ac':
        ao_seasonal_hvac = disagg_output_object['ao_seasonality']['epoch_cooling'] / Cgbdisagg.WH_IN_1_KWH
    else:
        ao_seasonal_hvac = disagg_output_object['ao_seasonality']['epoch_heating'] / Cgbdisagg.WH_IN_1_KWH

    # Aggregate ao seasonal on bill cycle level
    ao_hvac_monthly = np.bincount(month_idx, ao_seasonal_hvac)

    # Identify months to suppress AO in : on demand component is zero and ao components is non-zero but less than limit
    no_hvac_months = month_ao_hvac_res_net[:, 0][
        (month_ao_hvac_res_net[:, column_index[appliance]] == 0) & (ao_hvac_monthly < month_low_limit) & (
                ao_hvac_monthly > 0)]

    # Get the epoch indices of identified bill cycles and suppress the epoch ao seasonal component
    if np.any(no_hvac_months):
        fp_indexes = np.isin(epoch_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], no_hvac_months)
        ao_seasonal_hvac[fp_indexes] = 0

    # Re-create the ao seasonal component aggregated on bill cycle level after converting back into WH
    ao_seasonal_hvac = ao_seasonal_hvac * Cgbdisagg.WH_IN_1_KWH
    ao_hvac_monthly = np.bincount(month_idx, ao_seasonal_hvac)

    # Update disagg output object
    if appliance == 'ac':
        disagg_output_object['ao_seasonality']['cooling'] = ao_hvac_monthly
        disagg_output_object['ao_seasonality']['epoch_cooling'] = ao_seasonal_hvac
    else:
        disagg_output_object['ao_seasonality']['heating'] = ao_hvac_monthly
        disagg_output_object['ao_seasonality']['epoch_heating'] = ao_seasonal_hvac

    return disagg_output_object


def postprocess_hvac(month_ao_hvac_res_net, epoch_ao_hvac_true, disagg_input_object, disagg_output_object, column_index,
                     hvac_tn, logger_hvac):
    """
    Function to post process hvac results in case of over/under estimation

    Parameters:
        month_ao_hvac_res_net   (np.ndarray)    : Array containing | month-ao-ac-sh-residue-net energies
        epoch_ao_hvac_true      (np.ndarray)    : Array containing | epoch-ao-ac-sh energies
        disagg_input_object     (dict)          : Dictionary containing all input attributes
        disagg_output_object    (dict)          : Dictionary containing all output attributes
        column_index            (dict)          : Dictionary containing column identifier indices of ao-ac-sh
        hvac_tn                 (dict)          : Dictionary flagging non-HVAC detected users
        logger_hvac             (logger)        : Logger to log stuff in code

    Returns:

        month_ao_hvac_res_net   (np.ndarray)     : Array containing | month-ao-ac-sh-residue-net energies (Processed)
        epoch_ao_hvac_true      (np.ndarray)     : Array containing | epoch-ao-ac-sh energies (Processed)
        disagg_output_object    (dict)           : Dictionary containing all output attributes
    """

    # reading hvac static parameters
    static_params = hvac_static_params()
    sampling_rate = disagg_input_object['config']['sampling_rate']
    epochs_per_hour = Cgbdisagg.SEC_IN_HOUR / sampling_rate
    month_ao_hvac_res_net[:, column_index['ao']] = disagg_output_object['ao_seasonality']['baseload']
    epoch_ao_hvac_true[:, column_index['ao']] = disagg_output_object['ao_seasonality']['epoch_baseload']

    # figuring out potential candidates for hvac underestimation based on residue distribution
    std_residual_col = -2
    try:

        # getting month level residuals and identifying qualified residuals for decision making of over/under estimation
        residual = month_ao_hvac_res_net[:, std_residual_col]
        concerned_residual = month_ao_hvac_res_net[(np.sum(month_ao_hvac_res_net[:,
                                                           [column_index.get('ac'), column_index.get('sh')]],
                                                           axis=1) == 0),
                                                   column_index.get('residue')]

        # if ample qualified residuals exist identify idea residuals to meet at month level
        if len(concerned_residual) > static_params.get('std_concerned_residual'):

            residual_to_meet = np.nanmedian(concerned_residual)
            # identifying potential underestimated hvac months
            underestimated_boolean = residual > (np.median(concerned_residual) +
                                                 static_params.get('under_estimate_arm') * np.std(concerned_residual))
        else:
            residual_to_meet = np.nanmedian(residual)

            # identifying potential underestimated hvac months
            underestimated_boolean = residual > (np.median(residual[1:-1]) +
                                                 static_params.get('under_estimate_arm') * np.std(residual[1:-1]))

    except IndexError:

        # failsafe method to extract representative residual to meet
        residual = month_ao_hvac_res_net[:, std_residual_col]
        concerned_residual = copy.deepcopy(residual)
        residual_to_meet = np.nanmedian(concerned_residual)

        # identifying potential underestimated hvac months
        underestimated_boolean = residual > (np.median(concerned_residual[1:-1]) +
                                             static_params.get('under_estimate_arm') * np.std(concerned_residual[1:-1]))

    epoch_input_data = copy.deepcopy(disagg_input_object.get('switch', {}).get('hvac_input_data_timed_removed'))

    # getting temperature and month identifiers at epoch level
    month_epoch, _, month_idx = scipy.unique(epoch_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                                             return_index=True, return_inverse=True)

    # Calculate the scaling factor from 30 days to  number of days each bill cycle
    days_in_bc = np.bincount(month_idx) * (sampling_rate / Cgbdisagg.SEC_IN_DAY)
    days_in_bc_scaled = days_in_bc / Cgbdisagg.DAYS_IN_MONTH
    days_in_bc_scaled = np.clip(days_in_bc_scaled, a_min=None, a_max=2)

    temperature = epoch_input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]

    # figuring out potential candidates for hvac overestimation based on residue distribution

    # getting overestimation metric to identify need to overestimation handling
    overestimation_metric = (
            np.nanmedian(residual[1:-1]) - static_params.get('over_estimate_arm') * np.std(residual[1:-1]))

    # if overestimation metric exists identify months overestimated for hvac estimates
    if not np.isnan(overestimation_metric) and overestimation_metric < 0:
        overestimated_boolean = residual < np.nanmedian(residual[1:-1])
    elif not np.isnan(overestimation_metric) and overestimation_metric > 0:
        overestimated_boolean = residual < (np.nanmedian(residual[1:-1]) -
                                            static_params.get('over_estimate_arm') * np.nanstd(residual[1:-1]))
    else:
        overestimated_boolean = np.zeros(residual.shape).astype(bool)

    # suppressing first and last months for overestimation candidates for sanity purpose
    overestimated_boolean[0] = False
    overestimated_boolean[-1] = False

    # getting heating and cooling amplitude limits from detection module

    hdd_info = disagg_output_object.get('hvac_debug', {}).get('detection', {}).get('hdd', {})
    hdd_limits_info = hdd_info.get('amplitude_cluster_info', {}).get('cluster_limits',
                                                                     [[np.inf, np.inf], [np.inf, np.inf]])
    hdd_amplitude_lim = (hdd_limits_info[0][0], hdd_limits_info[-1][-1])

    cdd_info = disagg_output_object.get('hvac_debug', {}).get('detection', {}).get('cdd', {})
    ac_setpoint = disagg_output_object.get('hvac_debug', {}).get('estimation', {}).get('cdd', {}).get('setpoint',
                                                                                                      np.nan)
    cooling_limits = cdd_info.get('amplitude_cluster_info', {}).get('cluster_limits',
                                                                    [[np.inf, np.inf], [np.inf, np.inf]])
    cooling_amplitude_limits = (
        max_min_without_inf_nan(cooling_limits, 'min'), max_min_without_inf_nan(cooling_limits, 'max'))

    # Get the low consumption cooling flag
    low_consumption_ac_flag = disagg_output_object.get('hvac_debug', {}).get('pre_pipeline', {}) \
        .get('all_flags', {}).get('adjust_ac_detection_range_flag', 0)

    # getting representative cdd/hdd at epoch level to identify severity in temperature

    # For cooling : If setpoint is already calculated, select the more conservative setpoint between default and current value
    cdd_setpoint = np.nanmax([static_params.get('post_process_cdd_ref'), ac_setpoint])
    cdd = np.fmax(temperature - cdd_setpoint, 0)
    hdd_setpoint = np.nanmin([static_params.get('post_process_hdd_ref')])
    hdd = np.fmax(0, hdd_setpoint - temperature)

    # aggregating epoch level degree days to get month level degree days for hvac
    month_cdd = np.bincount(month_idx, cdd)
    month_hdd = np.bincount(month_idx, hdd)
    month_cdd[month_cdd < month_hdd] = 0
    month_hdd[month_hdd < month_cdd] = 0
    cdd_months = (month_cdd > 0).astype(bool)
    hdd_months = (month_hdd > 0).astype(bool)

    # Read user input consumption and temperature data at epoch level
    epoch_ao = epoch_ao_hvac_true[:, column_index.get('ao')]
    epoch_ac = epoch_ao_hvac_true[:, column_index.get('ac')]
    epoch_sh = epoch_ao_hvac_true[:, column_index.get('sh')]
    epoch_grey_residue_component = disagg_output_object.get('ao_seasonality', {}).get('epoch_grey', 0)
    epoch_residue = epoch_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] - (epoch_ao + epoch_ac + epoch_sh) + epoch_grey_residue_component

    epoch_ao_hvac_res_net = np.c_[epoch_ao_hvac_true, epoch_residue]
    epoch_ao_hvac_res_net = np.c_[epoch_ao_hvac_res_net, epoch_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]]
    epoch_ao_hvac_res_net_copy = copy.deepcopy(epoch_ao_hvac_res_net)

    # Prepare appliance df for restructuring hvac consumption based on identified over/under estimations
    appliance_df = pd.DataFrame()
    appliance_df['ao'] = epoch_ao
    appliance_df['ac'] = epoch_ac
    appliance_df['sh'] = epoch_sh
    appliance_df['residue'] = epoch_residue
    appliance_df_deepcopy = copy.deepcopy(appliance_df)

    # hvac restructuring won't happen beyond identified hvac amplitude range (for SH only) in detection module
    hdd[(epoch_residue < hdd_amplitude_lim[0]) | (epoch_residue > hdd_amplitude_lim[1])] = 0
    epoch_cooling_contenders = (cdd > 0).astype(bool)
    epoch_heating_contenders = (hdd > 0).astype(bool)

    info_carrier = {'appliance_df': appliance_df_deepcopy,
                    'epoch_ao_hvac_res_net': epoch_ao_hvac_res_net,
                    'epoch_input_data': epoch_input_data,
                    'residual_to_meet': residual_to_meet,
                    'cooling_amplitude_limits': cooling_amplitude_limits,
                    'low_consumption_ac_flag': low_consumption_ac_flag,
                    'heating_amplitude_limits': hdd_amplitude_lim,
                    'epochs_per_hour': epochs_per_hour,
                    'month_cdd': month_cdd,
                    'month_hdd': month_hdd,
                    'month_epoch': month_epoch,
                    'column_index': column_index}

    # Block Handler 1 ----------------------------------- >

    # Handling Obvious Underestimation at month level if some estimation was made and detection was not rejected
    if (not hvac_tn['AC']) and np.sum(appliance_df['ac']) > 0:
        cooling_from_residue = get_hvac_from_residue(underestimated_boolean, epoch_cooling_contenders,
                                                     residual, epoch_ao_hvac_res_net_copy, cdd_months, 'ac',
                                                     info_carrier)
    else:
        cooling_from_residue = np.zeros(epoch_cooling_contenders.shape).reshape(-1, 1)

    heating_from_residue = get_hvac_from_residue(underestimated_boolean, epoch_heating_contenders,
                                                 residual, epoch_ao_hvac_res_net_copy, hdd_months, 'sh',
                                                 info_carrier)

    logger_hvac.info('Cooling from residue | %.3f', np.sum(cooling_from_residue))
    logger_hvac.info('Heating from residue | %.3f', np.sum(heating_from_residue))

    # Handling Obvious Overestimation at month level if some estimation was made
    residue_from_cooling = get_residue_from_hvac(overestimated_boolean, epoch_cooling_contenders, residual,
                                                 cdd_months, 'ac', info_carrier)
    residue_from_heating = get_residue_from_hvac(overestimated_boolean, epoch_heating_contenders, residual,
                                                 hdd_months, 'sh', info_carrier)

    logger_hvac.info('residue from Cooling | %.3f', np.sum(residue_from_cooling))
    logger_hvac.info('residue from Heating | %.3f', np.sum(residue_from_heating))

    logger_hvac.info('Cooling Detected: adjusting for estimation')
    delta_residue_to_add_cooling = cooling_from_residue[:, 0] - residue_from_cooling[:, 0]
    epoch_ao_hvac_true[:, column_index.get('ac')] = epoch_ao_hvac_true[:, column_index.get('ac')] + delta_residue_to_add_cooling

    logger_hvac.info('Cooling Detected: adjusting for estimation')
    delta_residue_to_add_heating = heating_from_residue[:, 0] - residue_from_heating[:, 0]
    epoch_ao_hvac_true[:, column_index.get('sh')] = epoch_ao_hvac_true[:, column_index.get('sh')] + delta_residue_to_add_heating

    # aggregating ac estimates at month level for monthly estimates
    month_cooling_from_residue = np.bincount(month_idx, cooling_from_residue[:, 0])
    month_cooling_from_residue = month_cooling_from_residue / Cgbdisagg.WH_IN_1_KWH
    month_residue_from_cooling = np.bincount(month_idx, residue_from_cooling[:, 0])
    month_residue_from_cooling = month_residue_from_cooling / Cgbdisagg.WH_IN_1_KWH

    month_heating_from_residue = np.bincount(month_idx, heating_from_residue[:, 0])
    month_heating_from_residue = month_heating_from_residue / Cgbdisagg.WH_IN_1_KWH
    month_residue_from_heating = np.bincount(month_idx, residue_from_heating[:, 0])
    month_residue_from_heating = month_residue_from_heating / Cgbdisagg.WH_IN_1_KWH

    delta_residue_to_add_cooling = month_cooling_from_residue - month_residue_from_cooling
    month_ao_hvac_res_net[:, column_index.get('ac')] = month_ao_hvac_res_net[:, column_index.get('ac')] + delta_residue_to_add_cooling

    delta_residue_to_add_heating = month_heating_from_residue - month_residue_from_heating
    month_ao_hvac_res_net[:, column_index.get('sh')] = month_ao_hvac_res_net[:, column_index.get('sh')] + delta_residue_to_add_heating

    delta_hvac_to_subtract_residue = delta_residue_to_add_cooling + delta_residue_to_add_heating
    month_ao_hvac_res_net[:, std_residual_col] = month_ao_hvac_res_net[:, std_residual_col] - delta_hvac_to_subtract_residue

    # Block Handler 2 ----------------------------------- >
    epoch_ao_hvac_true, month_ao_hvac_res_net, disagg_output_object = remove_week_clashes(epoch_ao_hvac_true,
                                                                                          month_ao_hvac_res_net,
                                                                                          epoch_input_data,
                                                                                          disagg_output_object,
                                                                                          column_index, month_idx)

    # Duty Cycle HVAC -----------------------
    net_consumption = epoch_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    ac_od_consumption = epoch_ao_hvac_true[:, column_index['ac']]
    sh_od_consumption = epoch_ao_hvac_true[:, column_index['sh']]
    epoch_ao_hvac_true[:, column_index['ac']] = impose_duty_cycle_on_hvac(ac_od_consumption, net_consumption)
    epoch_ao_hvac_true[:, column_index['sh']] = impose_duty_cycle_on_hvac(sh_od_consumption, net_consumption)

    # Block Handler 4 ----------------------------------- >
    # Ensuring net HVAC consumption do not exceed monthly total consumption
    epoch_ao_hvac_true, month_ao_hvac_res_net = avoid_overshoot_net_consumption(epoch_ao_hvac_true,
                                                                                column_index, disagg_input_object,
                                                                                disagg_output_object)

    # Block Handler 5 ----------------------------------- >
    info_carrier_fp_removal = {
        'epoch_ao_hvac_true': epoch_ao_hvac_true,
        'month_ao_hvac_res_net': month_ao_hvac_res_net,
        'epoch_input_data': epoch_input_data,
        'month_idx': month_idx,
        'days_in_bc_scaled': days_in_bc_scaled,
        'low_consumption_ac_flag': low_consumption_ac_flag,
        'low_consumption_sh_flag': 0
    }

    residue_from_cooling_fp = remove_fp_hvac(info_carrier_fp_removal, column_index, epochs_per_hour,
                                             disagg_output_object, 'ac')
    residue_from_heating_fp = remove_fp_hvac(info_carrier_fp_removal, column_index, epochs_per_hour,
                                             disagg_output_object, 'sh')

    epoch_ao_hvac_true[:, column_index.get('ac')] = epoch_ao_hvac_true[:, column_index.get('ac')] - residue_from_cooling_fp[:, 0]
    epoch_ao_hvac_true[:, column_index.get('sh')] = epoch_ao_hvac_true[:, column_index.get('sh')] - residue_from_heating_fp[:, 0]

    # aggregating ac estimates at month level for monthly estimates
    month_residue_from_cooling_fp = np.bincount(month_idx, residue_from_cooling_fp[:, 0])
    month_residue_from_cooling_fp = month_residue_from_cooling_fp / Cgbdisagg.WH_IN_1_KWH

    month_residue_from_heating_fp = np.bincount(month_idx, residue_from_heating_fp[:, 0])
    month_residue_from_heating_fp = month_residue_from_heating_fp / Cgbdisagg.WH_IN_1_KWH

    month_ao_hvac_res_net[:, column_index.get('ac')] = month_ao_hvac_res_net[:, column_index.get('ac')] - month_residue_from_cooling_fp
    month_ao_hvac_res_net[:, column_index.get('sh')] = month_ao_hvac_res_net[:, column_index.get('sh')] - month_residue_from_heating_fp

    month_ao_hvac_res_net[:, std_residual_col] = month_ao_hvac_res_net[:, std_residual_col] + month_residue_from_cooling_fp
    month_ao_hvac_res_net[:, std_residual_col] = month_ao_hvac_res_net[:, std_residual_col] + month_residue_from_heating_fp

    # Adjust AO for no OD HVAC consumption
    disagg_output_object = suppress_ao_seasonal(month_ao_hvac_res_net, disagg_output_object,
                                                epoch_input_data,
                                                days_in_bc_scaled, month_idx, column_index, 'ac',
                                                low_consumption_ac_flag)

    # Add AO_AC, OD_AC, AO_SH, OD_SH
    month_ao_hvac_res_net = np.c_[month_ao_hvac_res_net, disagg_output_object['ao_seasonality']['cooling']]
    month_ao_hvac_res_net = np.c_[month_ao_hvac_res_net, month_ao_hvac_res_net[:, column_index['ac']] * Cgbdisagg.WH_IN_1_KWH]
    month_ao_hvac_res_net = np.c_[month_ao_hvac_res_net, disagg_output_object['ao_seasonality']['heating']]
    month_ao_hvac_res_net = np.c_[month_ao_hvac_res_net, month_ao_hvac_res_net[:, column_index['sh']] * Cgbdisagg.WH_IN_1_KWH]

    epoch_ao_hvac_true = np.c_[epoch_ao_hvac_true, disagg_output_object['ao_seasonality']['epoch_cooling']]
    epoch_ao_hvac_true = np.c_[epoch_ao_hvac_true, epoch_ao_hvac_true[:, column_index['ac']]]
    epoch_ao_hvac_true = np.c_[epoch_ao_hvac_true, disagg_output_object['ao_seasonality']['epoch_heating']]
    epoch_ao_hvac_true = np.c_[epoch_ao_hvac_true, epoch_ao_hvac_true[:, column_index['sh']]]

    # Add AO HVAC into HVAC Columns
    month_ao_hvac_res_net[:, column_index['ac']] = \
        month_ao_hvac_res_net[:, column_index['ac']] * Cgbdisagg.WH_IN_1_KWH + disagg_output_object['ao_seasonality'][
            'cooling']

    epoch_ao_hvac_true[:, column_index['ac']] = \
        epoch_ao_hvac_true[:, column_index['ac']] + disagg_output_object['ao_seasonality']['epoch_cooling']

    month_ao_hvac_res_net[:, column_index['sh']] = \
        month_ao_hvac_res_net[:, column_index['sh']] * Cgbdisagg.WH_IN_1_KWH + disagg_output_object['ao_seasonality'][
            'heating']

    epoch_ao_hvac_true[:, column_index['sh']] = \
        epoch_ao_hvac_true[:, column_index['sh']] + disagg_output_object['ao_seasonality']['epoch_heating']

    return month_ao_hvac_res_net, epoch_ao_hvac_true, disagg_output_object
