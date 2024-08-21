"""
Author - Abhinav Srivastava
Date - 22nd Oct 2018
Call to fill hvac user profile
"""

# Import python packages

import copy
import logging
import traceback
import pandas as pd
import numpy as np

from sklearn import mixture

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.get_cooling_profile import get_cooling_profile
from python3.disaggregation.aer.hvac.get_heating_profile import get_heating_profile
from python3.utils.validate_output_schema import validate_appliance_profile_schema_for_billcycle


def get_largest_valid_mode(mid_mode_amplitude, sh_electric_threshold, valid_mode_amplitudes, valid_mode_hours):
    """
    Function to get the largest valid mode for SH fuel type

    Parameters:

        mid_mode_amplitude              (float)     : Mid mode amplitude value
        sh_electric_threshold           (int)       : Threshold for electric type classification
        valid_mode_amplitudes           (np.ndarray): Array of amplitudes
        valid_mode_hours                (np.ndarray): Array of hours for each amplitude

    Returns:
        largest_valid_mode_amplitude    (float)     : Largest valid mode amplitude modified
        largest_valid_mode_hours        (float)     : Largest mode's number of hours modified

    """

    if mid_mode_amplitude > sh_electric_threshold:
        numerator_weight = valid_mode_amplitudes[-2] * valid_mode_hours[-2] + valid_mode_amplitudes[-1] * \
                           valid_mode_hours[-1]
        denominator_hours = valid_mode_hours[-2] + valid_mode_hours[-1]
        largest_valid_mode_amplitude = round(numerator_weight / denominator_hours)
        largest_valid_mode_hours = round(denominator_hours)

    else:
        largest_valid_mode_amplitude = valid_mode_amplitudes[-1]
        largest_valid_mode_hours = valid_mode_hours[-1]

    return largest_valid_mode_amplitude, largest_valid_mode_hours


def get_smallest_valid_mode(mid_mode_amplitude, sigma_mid, sh_gas_margin_threshold, valid_mode_amplitudes, valid_mode_hours):
    """
    Function to get the smallest valid mode for SH fuel type

    Parameters:

        mid_mode_amplitude              (float)     : Mid mode amplitude value
        sigma_mid                       (float)     : Standard deviation of mid amplitude
        sh_gas_margin_threshold         (int)       : Threshold for gas type classification
        valid_mode_amplitudes           (np.ndarray): Array of amplitudes
        valid_mode_hours                (np.ndarray): Array of hours for each amplitude

    Returns:
        smallest_valid_mode_amplitude   (float)     : Smallest valid mode amplitude modified
        smallest_valid_mode_hours       (float)     : Smallest mode's number of hours modified

    """

    if mid_mode_amplitude + sigma_mid < sh_gas_margin_threshold:
        numerator_weight = valid_mode_amplitudes[0] * valid_mode_hours[0] + valid_mode_amplitudes[1] * valid_mode_hours[1]
        denominator_hours = valid_mode_hours[0] + valid_mode_hours[1]
        smallest_valid_mode_amplitude = round(numerator_weight / denominator_hours)
        smallest_valid_mode_hours = round(denominator_hours)
    else:
        smallest_valid_mode_amplitude = valid_mode_amplitudes[0]
        smallest_valid_mode_hours = valid_mode_hours[0]

    return smallest_valid_mode_amplitude, smallest_valid_mode_hours


def get_electric_dominant_sh_type(largest_valid_mode_hours, smallest_valid_mode_amplitude, sh_gas_threshold,
                                  smallest_valid_mode_hours):
    """
    Function to get general fuel types of  SH

    Parameters:

        largest_valid_mode_amplitude    (float)     : Largest valid mode amplitude modified
        smallest_valid_mode_amplitude   (float)     : Smallest valid mode amplitude modified
        sh_gas_threshold                (int)       : Threshold for gas type classification
        smallest_valid_mode_hours       (float)     : Smallest mode's number of hours modified

    Returns:
        primary_heating_type    (str)       : String containing primary SH fuel type tag identified
        secondary_heating_type  (str)       : String containing secondary SH fuel type tag identified

    """

    if largest_valid_mode_hours > 600:
        primary_heating_type = 'Electric'

        if (smallest_valid_mode_amplitude < sh_gas_threshold) & (smallest_valid_mode_hours > 3 * largest_valid_mode_hours):
            secondary_heating_type = 'Gas'
        else:
            secondary_heating_type = None


    elif (smallest_valid_mode_amplitude < sh_gas_threshold) & (smallest_valid_mode_hours > 3 * largest_valid_mode_hours):
        primary_heating_type = 'Gas'
        secondary_heating_type = 'Electric'

    else:
        primary_heating_type = 'Electric'
        secondary_heating_type = None

    return primary_heating_type, secondary_heating_type


def get_general_fuel_type(largest_valid_mode_amplitude, sh_gas_threshold, sigma_large, sh_gas_margin_threshold,
                          sh_electric_threshold, smallest_valid_mode_amplitude, smallest_valid_mode_hours,
                          largest_valid_mode_hours):
    """
    Function to get general fuel types of  SH

    Parameters:

        largest_valid_mode_amplitude    (float)     : Largest valid mode amplitude modified
        sh_gas_threshold                (int)       : Threshold for gas type classification
        sigma_large                     (float)     : Standard deviation of large amplitude
        sh_gas_margin_threshold         (int)       : Threshold for gas type classification
        sh_electric_threshold           (int)       : Threshold for electric type classification
        smallest_valid_mode_amplitude   (float)     : Smallest valid mode amplitude modified
        smallest_valid_mode_hours       (float)     : Smallest mode's number of hours modified
        largest_valid_mode_hours        (float)     : Largest mode's number of hours modified

    Returns:
        primary_heating_type    (str)       : String containing primary SH fuel type tag identified
        secondary_heating_type  (str)       : String containing secondary SH fuel type tag identified

    """

    if largest_valid_mode_amplitude <= sh_gas_threshold:

        primary_heating_type = 'Gas'
        secondary_heating_type = None

    elif largest_valid_mode_amplitude + sigma_large < sh_gas_margin_threshold:
        primary_heating_type = 'Gas'
        secondary_heating_type = None

    elif (largest_valid_mode_amplitude <= sh_electric_threshold):

        if (smallest_valid_mode_amplitude < sh_gas_threshold) & (smallest_valid_mode_hours > 2 * largest_valid_mode_hours):
            primary_heating_type = 'Gas'
            secondary_heating_type = None

        else:
            primary_heating_type = 'Electric'
            secondary_heating_type = 'Gas'

    else:

        primary_heating_type, secondary_heating_type = get_electric_dominant_sh_type(largest_valid_mode_hours,
                                                                                     smallest_valid_mode_amplitude,
                                                                                     sh_gas_threshold,
                                                                                     smallest_valid_mode_hours)

    return primary_heating_type, secondary_heating_type


def get_sh_fuel_type_gmm(disagg_input_object, disagg_output_object, month_estimates, col_idx):
    """
    Function to get the heating type appliance profile (None / "Electric" / "Gas")

    Parameters:
        disagg_input_object     (dict)      : Dictionary containing all input objects for disagg
        disagg_output_object    (dict)      : Dictionary containing all output objects for disagg
        month_estimates         (np.ndarray): Array containing all month level HVAC estimates
        col_idx                 (dict)      : Dictionary containing AC/SH column identifiers

    Returns:
        primary_heating_type    (str)       : String containing primary SH fuel type tag identified
        secondary_heating_type  (str)       : String containing secondary SH fuel type tag identified
    """

    # threshold for sh type Electric is 600Wh
    sh_electric_threshold = 600
    sh_gas_threshold = 270
    sh_gas_margin_threshold = 450

    sampling_rate = disagg_input_object.get('config').get('sampling_rate')

    # extracting epoch level heating estimates
    epoch_hvac_estimates = disagg_output_object.get('hvac_debug').get('write').get('epoch_ao_hvac_true')
    epoch_idx_id = disagg_output_object.get('hvac_debug').get('write').get('epoch_idx_dentify')
    epoch_ao_sh = epoch_hvac_estimates[:, epoch_idx_id.get('sh_ao')]
    epoch_od_sh = epoch_hvac_estimates[:, epoch_idx_id.get('sh_od')]

    # Getting extent of heating in run
    epoch_net_heating = epoch_ao_sh + epoch_od_sh
    epoch_net_heating = epoch_net_heating[epoch_net_heating > 0]
    num_heating_epochs = len(epoch_net_heating)
    total_heating_hours = num_heating_epochs * sampling_rate / Cgbdisagg.SEC_IN_HOUR

    # extracting total hvac consumption in current run
    total_ao_sh = np.round(month_estimates[:, col_idx.get('sh_ao_idx')])
    total_od_sh = np.round(month_estimates[:, col_idx.get('sh_od_idx')])
    month_total_sh = total_ao_sh + total_od_sh
    valid_heating = month_total_sh > 0
    has_heating = (np.nansum(valid_heating) > 0) & (num_heating_epochs > 3)

    if has_heating:

        n_estimators = np.arange(1, 4)
        clfs = [mixture.GaussianMixture(n).fit(epoch_net_heating.reshape(-1,1)) for n in n_estimators]

        # Analyzing model with three gaussians
        clf = clfs[2]
        sorted_model = np.argsort(np.squeeze(clf.means_))

        mu_small = clf.means_[sorted_model[0], 0]
        weight_small = clf.weights_[sorted_model[0]]
        heating_hrs_small = weight_small * total_heating_hours
        small_mode_valid = heating_hrs_small > Cgbdisagg.HRS_IN_DAY
        small_mode_amplitude = mu_small * Cgbdisagg.SEC_IN_HOUR / sampling_rate

        mu_mid = clf.means_[sorted_model[1], 0]
        sigma_mid = np.sqrt(clf.covariances_[sorted_model[1]])[0, 0]
        weight_mid = clf.weights_[sorted_model[1]]
        heating_hrs_mid = weight_mid * total_heating_hours
        mid_mode_valid = heating_hrs_mid > Cgbdisagg.HRS_IN_DAY
        mid_mode_amplitude = mu_mid * Cgbdisagg.SEC_IN_HOUR / sampling_rate

        mu_large = clf.means_[sorted_model[2], 0]
        sigma_large = np.sqrt(clf.covariances_[sorted_model[2]])[0, 0]
        weight_large = clf.weights_[sorted_model[2]]
        heating_hrs_large = weight_large * total_heating_hours
        large_mode_valid = heating_hrs_large > Cgbdisagg.HRS_IN_DAY
        large_mode_amplitude = mu_large * Cgbdisagg.SEC_IN_HOUR / sampling_rate

        validity_array = np.array([small_mode_valid, mid_mode_valid, large_mode_valid])
        amplitude_array = np.array([small_mode_amplitude, mid_mode_amplitude, large_mode_amplitude])
        hours_array = np.array([heating_hrs_small, heating_hrs_mid, heating_hrs_large])

        valid_mode_idxs = np.where(validity_array == True)[0]

        valid_mode_amplitudes = amplitude_array[valid_mode_idxs]
        valid_mode_hours = hours_array[valid_mode_idxs]

        # returning None types when no valid mode is found

        if len(valid_mode_amplitudes) == 0:

            primary_heating_type = None
            secondary_heating_type = None

            return primary_heating_type, secondary_heating_type

        if len(valid_mode_amplitudes) == 3:

            largest_valid_mode_amplitude, largest_valid_mode_hours = get_largest_valid_mode(mid_mode_amplitude,
                                                                                            sh_electric_threshold,
                                                                                            valid_mode_amplitudes,
                                                                                            valid_mode_hours)

        else:

            largest_valid_mode_amplitude = valid_mode_amplitudes[-1]
            largest_valid_mode_hours = valid_mode_hours[-1]

        if len(valid_mode_amplitudes) == 3:

            smallest_valid_mode_amplitude, smallest_valid_mode_hours = get_smallest_valid_mode(mid_mode_amplitude,
                                                                                               sigma_mid,
                                                                                               sh_gas_margin_threshold,
                                                                                               valid_mode_amplitudes,
                                                                                               valid_mode_hours)

        else:

            smallest_valid_mode_amplitude = valid_mode_amplitudes[0]
            smallest_valid_mode_hours = valid_mode_hours[0]

        primary_heating_type, secondary_heating_type = get_general_fuel_type(largest_valid_mode_amplitude,
                                                                             sh_gas_threshold, sigma_large,
                                                                             sh_gas_margin_threshold,
                                                                             sh_electric_threshold,
                                                                             smallest_valid_mode_amplitude,
                                                                             smallest_valid_mode_hours,
                                                                             largest_valid_mode_hours)

    else:

        primary_heating_type = None
        secondary_heating_type = None

    return primary_heating_type, secondary_heating_type


def get_cooling_type(disagg_input_object, disagg_output_object, month_estimates, col_idx):
    """
    Function to get the cooling type appliance profile (None / "Fan" / "AC")

    Parameters:
        disagg_input_object     (dict)      : Dictionary containing all input objects for disagg
        disagg_output_object    (dict)      : Dictionary containing all output objects for disagg
        month_estimates         (np.ndarray): Array containing all month level HVAC estimates
        col_idx                 (dict)      : Dictionary containing AC/SH column identifiers

    Returns:
        cooling_type            (str)       : String containing "Fan" or "AC" tag identified

    """

    sampling_rate = disagg_input_object.get('config').get('sampling_rate')

    # extracting epoch level Cooling estimates
    epoch_hvac_estimates = disagg_output_object.get('hvac_debug').get('write').get('epoch_ao_hvac_true')
    epoch_idx_id = disagg_output_object.get('hvac_debug').get('write').get('epoch_idx_dentify')
    epoch_ao_ac = epoch_hvac_estimates[:, epoch_idx_id.get('ac_ao')]
    epoch_od_ac = epoch_hvac_estimates[:, epoch_idx_id.get('ac_od')]

    # Getting extent of cooling in run
    epoch_net_cooling = epoch_ao_ac + epoch_od_ac
    epoch_net_cooling = epoch_net_cooling[epoch_net_cooling > 0]
    num_cooling_epochs = len(epoch_net_cooling)
    total_cooling_hours = num_cooling_epochs * sampling_rate / Cgbdisagg.SEC_IN_HOUR

    # extracting total hvac consumption in current run
    total_ao_ac = np.round(month_estimates[:, col_idx.get('ac_ao_idx')])
    total_od_ac = np.round(month_estimates[:, col_idx.get('ac_od_idx')])
    month_total_ac = total_ao_ac + total_od_ac
    valid_cooling = month_total_ac > 0
    has_cooling = np.nansum(valid_cooling) > 0

    if has_cooling:

        cooling_type = "Fan"

        n_estimators = np.arange(1, 4)
        clfs = [mixture.GaussianMixture(n).fit(epoch_net_cooling.reshape(-1,1)) for n in n_estimators]

        # Analyzing model with three gaussians
        clf = clfs[2]
        sorted_model = np.argsort(np.squeeze(clf.means_))

        mu_small = clf.means_[sorted_model[0], 0]
        weight_small = clf.weights_[sorted_model[0]]
        cooling_hrs_small = weight_small * total_cooling_hours
        small_mode_valid = cooling_hrs_small > Cgbdisagg.HRS_IN_DAY
        small_mode_amplitude = mu_small * Cgbdisagg.SEC_IN_HOUR / sampling_rate

        mu_mid = clf.means_[sorted_model[1], 0]
        weight_mid = clf.weights_[sorted_model[1]]
        cooling_hrs_mid = weight_mid * total_cooling_hours
        mid_mode_valid = cooling_hrs_mid > Cgbdisagg.HRS_IN_DAY
        mid_mode_amplitude = mu_mid * Cgbdisagg.SEC_IN_HOUR / sampling_rate

        mu_large = clf.means_[sorted_model[2], 0]
        weight_large = clf.weights_[sorted_model[2]]
        cooling_hrs_large = weight_large * total_cooling_hours
        large_mode_valid = cooling_hrs_large > Cgbdisagg.HRS_IN_DAY
        large_mode_amplitude = mu_large * Cgbdisagg.SEC_IN_HOUR / sampling_rate

        validity_array = np.array([small_mode_valid, mid_mode_valid, large_mode_valid])
        amplitude_array = np.array([small_mode_amplitude, mid_mode_amplitude, large_mode_amplitude])

        valid_mode_idxs = np.where(validity_array == True)[0]
        valid_mode_amplitudes = amplitude_array[valid_mode_idxs]

        if len(valid_mode_amplitudes) > 0:

            largest_valid_mode_amplitude = valid_mode_amplitudes[-1]

            # threshold for ac type amplitude is 700Wh
            ac_type_threshold = 700

            if largest_valid_mode_amplitude > ac_type_threshold:

                cooling_type = 'AC'

        else:
            cooling_type = None

    else:
        cooling_type = None

    return cooling_type


def fill_run_attributes(disagg_input_object, disagg_output_object, col_idx):

    """
    Function to prepare app profile attributes, constant for all BC in a run

    Parameters:
        disagg_input_object (dict)      : Contains user output attributes
        disagg_output_object (dict)     : Contains user output attributes
        col_idx (dict)                  : Dictionary containing appliance column identifier in month estimates

    Returns:
        run_attributes          (dict)     : Dictionary containing run related static appliance profile
    """

    sampling_rate = disagg_input_object.get('config').get('sampling_rate')

    # extracting run related hvac attributes
    info = disagg_output_object['created_hsm']['hvac']['attributes']
    month_estimates = disagg_output_object['hvac_debug']['write']['month_ao_hvac_res_net']

    # determining if hvac exists
    has_sh = bool(np.nansum(month_estimates[:, [col_idx.get('sh_ao_idx'), col_idx.get('sh_od_idx')]]) > 0)
    has_ac = bool(np.nansum(month_estimates[:, [col_idx.get('ac_ao_idx'), col_idx.get('ac_od_idx')]]) > 0)
    has_ao = bool(np.nansum(month_estimates[:, col_idx.get('ao_idx')]) > 0)

    # get heating fuel type
    primary_sh_fuel, secondary_sh_fuel = get_sh_fuel_type_gmm(disagg_input_object, disagg_output_object, month_estimates, col_idx)
    cooling_type = get_cooling_type(disagg_input_object, disagg_output_object, month_estimates, col_idx)

    # get cooling fuel type
    if has_ac:
        primary_ac_fuel = 'Electric'
    else:
        primary_ac_fuel = None

    # standardizing Means
    inf_handling = 123456789
    sh_means = np.array(info['sh_means'], dtype=float)
    sh_mode_amplitudes = np.array(info['sh_mode_limits_limits'], dtype=float)
    ac_means = np.array(info['ac_means'], dtype=float)
    ac_mode_amplitudes = np.array(info['ac_mode_limits_limits'])

    mult_factor = Cgbdisagg.SEC_IN_HOUR / sampling_rate
    const_handling = inf_handling * mult_factor

    if len(sh_means) > 0:
        sh_means = sh_means * mult_factor
        sh_means[sh_means == const_handling] = inf_handling
        sh_means = np.array(sh_means, dtype=float).tolist()

    if len(sh_mode_amplitudes) > 0:
        sh_mode_amplitudes = sh_mode_amplitudes * mult_factor
        sh_mode_amplitudes[sh_mode_amplitudes == const_handling] = inf_handling
        sh_mode_amplitudes = np.array(sh_mode_amplitudes, dtype=float).tolist()

    if len(ac_means) > 0:
        ac_means = ac_means * mult_factor
        ac_means[ac_means == const_handling] = inf_handling
        ac_means = np.array(ac_means, dtype=float).tolist()

    if len(ac_mode_amplitudes) > 0:
        ac_mode_amplitudes = ac_mode_amplitudes * mult_factor
        ac_mode_amplitudes[ac_mode_amplitudes == const_handling] = inf_handling
        ac_mode_amplitudes = np.array(ac_mode_amplitudes, dtype=float).tolist()

    run_attributes = {
        'sh': {
            'fuelType': primary_sh_fuel,
            'secondaryFuelType': secondary_sh_fuel,
            'usageModeCount': len(info['sh_means']),
            'usageModeCountConfidence': 1.0,
            'modeAmplitudes': sh_mode_amplitudes,
            'regressionCoeff': np.array(info['sh_cluster_info_keys_coefficient']).astype(float).tolist(),
            'regressionCoeffConfidence': np.ones(len(info['sh_cluster_info_keys_coefficient'])).astype((float)).tolist(),
            'regressionType': ['linear' if el == 1 else 'root' if el == 0.5 else None for el in info['sh_cluster_info_keys_kind']],
            'heatingMeans': sh_means,
            'heatingStd': np.array(info['sh_std'], dtype=float).tolist(),
            'isPresent': has_sh,
            'detectionConfidence': 1.0,
            'count': len(info['sh_means']),
            'timeOfUsage': None,
        },
        'ac': {
            'fuelType': primary_ac_fuel,
            'secondaryFuelType': None,
            'usageModeCount': len(info['ac_means']),
            'usageModeCountConfidence': 1.0,
            'modeAmplitudes': ac_mode_amplitudes,
            'regressionCoeff': np.array(info['ac_cluster_info_keys_coefficient']).astype(float).tolist(),
            'regressionCoeffConfidence': np.ones(len(info['ac_cluster_info_keys_coefficient'])).astype((float)).tolist(),
            'regressionType': ['linear' if el == 1 else 'root' if el == 0.5 else None for el in info['ac_cluster_info_keys_kind']],
            'coolingMeans': ac_means,
            'coolingStd': np.array(info['ac_std'], dtype=float).tolist(),
            'isPresent': has_ac,
            'detectionConfidence': 1.0,
            'count': len(info['ac_means']),
            'timeOfUsage': None,
            'coolingType': cooling_type
        },
        'ao':{
            'isPresent':has_ao,
            'detectionConfidence':1.0,
            'count':1,
            'fuelType':'Electric',
        }
    }

    return run_attributes


def get_tou_attributes(disagg_input_object, disagg_output_object):

    """
    Function to get the derived hvac tou signatures for each of the out billing cycles

    Parameters:
        disagg_input_object     (dict)     : Contains user input attributes
        disagg_output_object    (dict)     : Contains user output attributes

    Returns:
        tou_map                 (dict)     : Dictionary containing derived tou for ac and sh, for each out billing cycle
    """

    component_idx = disagg_output_object.get('hvac_debug').get('write').get('epoch_idx_dentify')

    epoch_energy = disagg_output_object.get('hvac_debug').get('write').get('epoch_ao_hvac_true')
    ac_od = epoch_energy[:, component_idx.get('ac_od')]
    sh_od = epoch_energy[:, component_idx.get('sh_od')]

    # getting input data
    input_data = disagg_input_object.get('input_data')
    month_arr = input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]
    input_data_raw = copy.deepcopy(pd.DataFrame(input_data))
    columns = Cgbdisagg.INPUT_COLUMN_NAMES
    input_data_raw.columns = columns

    # getting timestamps
    input_data_raw['timestamp'] = pd.to_datetime(input_data_raw['epoch'], unit='s')
    timezone = disagg_input_object['home_meta_data']['timezone']

    # noinspection PyBroadException
    try:
        input_data_raw['timestamp'] = input_data_raw.timestamp.dt.tz_localize('UTC', ambiguous='infer').dt.tz_convert(
            timezone)
    except (IndexError, KeyError, TypeError):
        input_data_raw['timestamp'] = input_data_raw.timestamp.dt.tz_localize('UTC', ambiguous='NaT').dt.tz_convert(
            timezone)

    # getting time attributes info
    input_data_raw['date'] = input_data_raw['timestamp'].dt.date
    input_data_raw['hour'] = input_data_raw['timestamp'].dt.hour
    input_data_raw['month'] = month_arr

    input_data_raw['ac_od'] = ac_od
    input_data_raw['sh_od'] = sh_od

    # getting ac map
    ac_od_map = input_data_raw.pivot_table(index='date', columns=['hour'], values='ac_od', aggfunc=np.nansum)
    ac_od_map = ac_od_map.fillna(0)
    ac_od_map = ac_od_map.astype(int)
    ac_od_map = ac_od_map.values

    # getting sh map
    sh_od_map = input_data_raw.pivot_table(index='date', columns=['hour'], values='sh_od', aggfunc=np.nansum)
    sh_od_map = sh_od_map.fillna(0)
    sh_od_map = sh_od_map.astype(int)
    sh_od_map = sh_od_map.values

    # getting month map
    month_map = input_data_raw.pivot_table(index='date', columns=['hour'], values='month', aggfunc=np.median)
    month_map = month_map.fillna(0)
    month_map = month_map.astype(int)
    month_map = month_map.values

    days_of_bc = np.median(month_map, axis=1)

    # getting out billing cycles
    out_bill_cycles = disagg_input_object.get('out_bill_cycles')

    # initializing derived tou object
    tou_map = {'ac':{}, 'sh':{}}

    # filling derived tou object for each out billing cycle
    for billcycle_start, _ in out_bill_cycles:

        # identifying days of current billing cycle
        bc_id = days_of_bc == billcycle_start

        # extracting hvac for current billing cycle
        bc_ac_od_map = ac_od_map[bc_id, :]
        bc_sh_od_map = sh_od_map[bc_id, :]

        # getting derived tou for ac
        if np.any(bc_ac_od_map):

            # summing ac consumption for each hour
            total_ac_tou = np.nansum(bc_ac_od_map, axis=0)

            # normalizing hour totals by total ac consumption
            bc_ac_od_tou = total_ac_tou / np.nansum(total_ac_tou)

            # rounding and getting derived tou signature in right format
            bc_ac_od_tou = np.round(bc_ac_od_tou, 4)
            bc_ac_od_tou = bc_ac_od_tou.astype(float).tolist()
        else:

            # failsafe ac derived signature
            bc_ac_od_tou = np.zeros(Cgbdisagg.HRS_IN_DAY).astype(float).tolist()

        # getting derived tou for sh
        if np.any(bc_sh_od_map):

            # summing sh consumption for each hour
            total_sh_tou = np.nansum(bc_sh_od_map, axis=0)

            # normalizing hour totals by total sh consumption
            bc_sh_od_tou = total_sh_tou / np.nansum(total_sh_tou)

            # rounding and getting derived tou signature in right format
            bc_sh_od_tou = np.round(bc_sh_od_tou, 4)
            bc_sh_od_tou = bc_sh_od_tou.astype(float).tolist()
        else:
            bc_sh_od_tou = np.zeros(Cgbdisagg.HRS_IN_DAY).astype(float).tolist()

        # assigning tou signature for current month, to tou carries dictionary
        tou_map['ac'][billcycle_start] = bc_ac_od_tou
        tou_map['sh'][billcycle_start] = bc_sh_od_tou

    return tou_map


def fill_user_profile(disagg_input_object, disagg_output_object, logger_base):

    """
    Function to fill user profile for hvac

    Parameters:
        disagg_input_object         (dict)              : Dictionary containing all inputs
        disagg_output_object        (dict)              : Dictionary containing all outputs
        logger_base                 (logger object)     : Logging object to log important steps and values in the run

    Returns:
        disagg_output_object        (dict)              : Dictionary containing all outputs
    """

    # initializing logging object
    logger_local = logger_base.get("logger").getChild("hvac_user_profile")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac_profile = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # getting out billing cycles
    out_bill_cycles = disagg_input_object.get('out_bill_cycles')

    # getting month estimates
    month_estimates = disagg_output_object['hvac_debug']['write']['month_ao_hvac_res_net']

    # column identifier in month estimates
    col_idx = {
        'ao_idx':1,
        'ac_ao_idx':6,
        'ac_od_idx':7,
        'sh_ao_idx':8,
        'sh_od_idx':9
    }

    # fill run level user profile
    run_attributes = fill_run_attributes(disagg_input_object, disagg_output_object, col_idx)
    tou_attributes = get_tou_attributes(disagg_input_object, disagg_output_object)

    # accessing user profiles object
    user_profile_object = disagg_output_object['appliance_profile']

    logger_hvac_profile.info(' Starting bill cycle profile writing |')

    # filling user profile for each billing cycle
    for billcycle_start, billcycle_end in out_bill_cycles:

        base = user_profile_object[billcycle_start]['profileList'][0]

        month_consumptions = month_estimates[month_estimates[:, 0] == billcycle_start]

        # Fill Heating Profile
        try:

            heating_profile = get_heating_profile(run_attributes, base, month_consumptions, billcycle_start,
                                                  billcycle_end, col_idx, tou_attributes)

            user_profile_object[billcycle_start]['profileList'][0]['3'][0] = heating_profile

            logger_hvac_profile.info(' Done heating profile for {} bill cycle |'.format(billcycle_start))

        except (ValueError, IndexError, KeyError):

            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger_hvac_profile.error('Heating Profile Fill for billcycle %d failed | %s' %
                                      (int(billcycle_start), error_str))
            logger_hvac_profile.info('Heating Profile Empty for billcycle | %d ', int(billcycle_start))

        # Fill Cooling profile
        try:

            cooling_profile = get_cooling_profile(run_attributes, base, month_consumptions, billcycle_start,
                                                  billcycle_end, col_idx, tou_attributes)

            user_profile_object[billcycle_start]['profileList'][0]['4'][0] = cooling_profile

            logger_hvac_profile.info(' Done cooling profile for {} bill cycle |'.format(billcycle_start))

        except (ValueError, IndexError, KeyError):

            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger_hvac_profile.error('Cooling Profile Fill for billcycle %d failed | %s' %
                                      (int(billcycle_start), error_str))
            logger_hvac_profile.info('Cooling Profile Empty for billcycle | %d ', int(billcycle_start))

        # Fill AO Profile
        try:
            ao_profile = base['8'][0]

            if len(month_consumptions) > 0:
                total_ao = month_consumptions[:, 1][0]
            else:
                total_ao = 0

            ao_profile['validity'] = dict()
            ao_profile['validity']['start'] = int(billcycle_start)
            ao_profile['validity']['end'] = int(billcycle_end)
            ao_profile['isPresent'] = run_attributes.get('ao').get('isPresent')
            ao_profile['detectionConfidence'] = run_attributes.get('ao').get('detectionConfidence')
            ao_profile['count'] = run_attributes.get('ao').get('count')
            ao_profile['attributes']['fuelType'] = run_attributes.get('ao').get('fuelType')
            ao_profile['attributes']['aoConsumption'] = float(total_ao)

            user_profile_object[billcycle_start]['profileList'][0]['8'][0] = ao_profile

            logger_hvac_profile.info(' Done AO profile for {} bill cycle |'.format(billcycle_start))

        except (ValueError, IndexError, KeyError):

            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger_hvac_profile.error('AO Profile Fill for billcycle %d failed | %s' %
                                      (int(billcycle_start), error_str))
            logger_hvac_profile.info('AO Profile Empty for billcycle | %d ', int(billcycle_start))

        disagg_output_object['appliance_profile'] = user_profile_object

        logger_hvac_profile.info(
            ' Updated AO-HVAC profile in disagg output object for {} bill cycle |'.format(billcycle_start))

        validate_appliance_profile_schema_for_billcycle(disagg_output_object, billcycle_start, logger_pass)

    return disagg_output_object


def fill_and_validate_user_profile(disagg_mode, disagg_input_object, disagg_output_object, logger_pass):

    """
    Function to fill and validate the user profile

    Parameters :
        disagg_mode             (str)               : Identifies disagg mode
        disagg_input_object     (dict)              : Contains key input information
        disagg_output_object    (dict)              : Contains key output information
        logger_pass             (logging object)    : Records log while running code

    Returns:
        disagg_output_object    (dict)              : Contains key output information
    """

    if disagg_mode != 'mtd':
        # Fill hvac appliance profile if disagg_mode is not mtd

        disagg_output_object = fill_user_profile(disagg_input_object, disagg_output_object, logger_pass)

    # Schema Validation for filled appliance profile
    out_bill_cycles = disagg_input_object.get('out_bill_cycles')

    for billcycle_start, _ in out_bill_cycles:

        # TODO(Abhinav): Write your code for filling appliance profile for this bill cycle here

        validate_appliance_profile_schema_for_billcycle(disagg_output_object, billcycle_start, logger_pass)

    return disagg_output_object
