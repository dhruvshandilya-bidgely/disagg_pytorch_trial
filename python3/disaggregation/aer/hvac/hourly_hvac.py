"""
Author - Abhinav
Date - 10/10/2018
Hourly HVAC Computation
"""

# Import python packages
import logging
import copy
import scipy
import numpy as np
from datetime import datetime
from python3.utils.time.get_time_diff import get_time_diff

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants
from python3.disaggregation.aer.hvac.estimate_hvac import estimate_hvac
from python3.disaggregation.aer.hvac.get_hsm_attributes import get_hsm_attributes
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import init_hvac_params
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import pre_detection_params
from python3.disaggregation.aer.hvac.detect_hvac_amplitude import detect_hvac_amplitude
from python3.disaggregation.aer.hvac.adjust_hvac_params import get_user_cooling_flags
from python3.disaggregation.aer.hvac.adjust_hvac_params import get_adjusted_hvac_parameters
from python3.disaggregation.aer.hvac.fetch_user_characteristics import get_user_characteristic

hvac_static_params = hvac_static_params()


def get_heat_count(app_profile):
    """
    Utility Function to get heating appliance count

    Parameters:
        app_profile (dict)  : Dictionary containing heating appliance count

    Returns:
        app_count   (int)   : Count number for heating
    """

    app_count = -1
    count = []

    app_name_list = ['sh', 'rh', 'ch', 'hp']

    for app_name in app_name_list:

        app_profile_name = app_profile.get(app_name)

        if app_profile_name is not None:
            if app_name == 'hp' and app_profile_name.get('number') < 1:
                count.append(0)
            else:
                count.append(app_profile_name.get('number'))

    count = np.array(count)

    pos_idx = count > 0
    neg_idx = count < 0
    zero_idx = count == 0

    if np.sum(pos_idx) > 0:
        app_count = np.sum(count[pos_idx])
    elif np.sum(neg_idx) == 0 and np.sum(zero_idx) > 0:
        app_count = 0

    return app_count


def get_cool_count(app_profile):
    """
    Utility Function to get cooling appliance count

    Parameters:
        app_profile (dict)  : Dictionary containing heating appliance count

    Returns:
        app_count   (int)   : Count number for heating
    """

    app_count = -1
    count = []

    app_name_list = ['cac', 'rac', 'hp']

    for app_name in app_name_list:

        app_profile_name = app_profile.get(app_name)

        if app_profile_name is not None:
            if app_name == 'hp' and app_profile_name.get('number') < 1:
                count.append(0)
            else:
                count.append(app_profile_name.get('number'))

    count = np.array(count)

    pos_idx = count > 0
    neg_idx = count < 0
    zero_idx = count == 0

    if np.sum(pos_idx) > 0:
        app_count = np.sum(count[pos_idx])
    elif np.sum(neg_idx) == 0 and np.sum(zero_idx) > 0:
        app_count = 0

    return app_count


def override_detection(disagg_input_object, appliance, base_found):
    """
    Function to override detection if appliance profile says no for AC/SH

    Parameters:
        disagg_input_object     (dict)  : Dictionary containing all inputs
        appliance               (str)   : Identifier for AC/SH
        base_found              (bool)  : Boolean indicating if AC/SH is found from HVAC algo in current run

    Returns:
        base_found              (bool)  : Boolean indicating if AC/SH is present as per appliance profile
    """

    app_count = -1

    if appliance == 'sh':
        app_count = get_heat_count(disagg_input_object['app_profile'])
    elif appliance == 'ac':
        app_count = get_cool_count(disagg_input_object['app_profile'])

    if app_count == 0:
        base_found = False
        return base_found
    else:
        return base_found


def add_detection_analytics(disagg_output_object, hvac_debug):
    """
    Function to populate hvac detection related parameters for 2nd order analytics

    Parameters:
        disagg_output_object    (dict)    : Dictionary containing all outputs
        hvac_debug              (dict)    : dictionary containing hvac detection debug

    Return:
        None
    """

    disagg_output_object['analytics']['values']['cooling'] = {
        'detection': hvac_debug['detection']['cdd']['amplitude_cluster_info']}
    disagg_output_object['analytics']['values']['heating'] = {
        'detection': hvac_debug['detection']['hdd']['amplitude_cluster_info']}


def restructure_atrributes_for_hsm(attributes):
    """
    Function restructures the attributes into format that is desired for hsm storage in cassandra
    Parameters:
        attributes  (dict)    : Dictionary containing attributes to be saved in hsm, but in nested form

    Returns:
         attributes (dict)    : Dictionary containing attributes to be saved in hsm, but in restructured form
    """

    restructured_attributes = copy.deepcopy(attributes)

    dict_to_structure = ['ac_cluster_info', 'sh_cluster_info']
    array_to_structure = ['ac_mode_limits', 'sh_mode_limits']

    ac_means = np.array(restructured_attributes['ac_means'])
    ac_means[ac_means == np.Inf] = 123456789
    restructured_attributes['ac_means'] = list(ac_means)

    sh_means = np.array(restructured_attributes['sh_means'])
    sh_means[sh_means == np.Inf] = 123456789
    restructured_attributes['sh_means'] = list(sh_means)

    ac_mode_limits = np.array(restructured_attributes['ac_mode_limits'])
    ac_mode_limits[ac_mode_limits == np.Inf] = 123456789
    restructured_attributes['ac_mode_limits'] = (
        (ac_mode_limits[0][0], ac_mode_limits[0][1]), (ac_mode_limits[1][0], ac_mode_limits[1][1]))

    sh_mode_limits = np.array(restructured_attributes['sh_mode_limits'])
    sh_mode_limits[sh_mode_limits == np.Inf] = 123456789
    restructured_attributes['sh_mode_limits'] = (
        (sh_mode_limits[0][0], sh_mode_limits[0][1]), (sh_mode_limits[1][0], sh_mode_limits[1][1]))

    for main_key in dict_to_structure:

        sub_keys = list(attributes[main_key].keys())

        restructured_attributes[main_key + '_keys'] = sub_keys

        kind_list = []
        validity_list = []
        coefficient_list = []
        intercept_list = []

        for cluster in sub_keys:

            if attributes[main_key][cluster]['regression_kind'] == 'root':
                kind_list.append(0.5)
            elif attributes[main_key][cluster]['regression_kind'] == 'linear':
                kind_list.append(1)

            validity_list.append(int(attributes[main_key][cluster]['validity']))

            try:
                coefficient_list.append(attributes[main_key][cluster]['coefficient'][0][0])
            except (IndexError, KeyError):
                coefficient_list.append(0)

            try:
                intercept_list.append(attributes[main_key][cluster]['intercept'][0])
            except (IndexError, KeyError):
                intercept_list.append(0)

        restructured_attributes[main_key + '_keys' + '_kind'] = kind_list
        restructured_attributes[main_key + '_keys' + '_validity'] = validity_list
        restructured_attributes[main_key + '_keys' + '_coefficient'] = coefficient_list
        restructured_attributes[main_key + '_keys' + '_intercept'] = intercept_list

        restructured_attributes.pop(main_key, None)

    for main_key in array_to_structure:
        limits = np.array(attributes[main_key])
        restructured_attributes[main_key + '_limits'] = list(limits.flatten())

        restructured_attributes.pop(main_key, None)

    ac_mode_limits_limits = np.array(restructured_attributes['ac_mode_limits_limits'])
    ac_mode_limits_limits[ac_mode_limits_limits == np.Inf] = 123456789
    restructured_attributes['ac_mode_limits_limits'] = list(ac_mode_limits_limits)

    sh_mode_limits_limits = np.array(restructured_attributes['sh_mode_limits_limits'])
    sh_mode_limits_limits[sh_mode_limits_limits == np.Inf] = 123456789
    restructured_attributes['sh_mode_limits_limits'] = list(sh_mode_limits_limits)

    return restructured_attributes


def setpoint_list(start, stop, step=1):
    """ Float list generation """
    return np.array(list(range(start, stop + step, step)))


def extract_pp_twh(disagg_output_object, pilot_id):
    """
    Extracts pool pump and timed water heater consumption form other modules to use in vacation disagg
    Parameters:
        disagg_output_object    (dict)              : Dictionary containing all outputs
        pilot_id                (int)               : The id for the pilot the user belongs to
    Returns:
        timed_disagg_output     (dict)              : Dict containing all timed appliance outputs
    """

    # Extract the pool pump output from source based on pilot id

    if pilot_id in PilotConstants.AUSTRALIA_PILOTS:

        # Extract pool pump output from special output for Origin

        cons_col = 1

        pp_disagg_output = copy.deepcopy(disagg_output_object.get('special_outputs').get('pp_consumption')[:, cons_col])

        if pp_disagg_output is None:
            pp_disagg_output = np.zeros_like(disagg_output_object.get('epoch_estimate').shape[0])

    else:

        # Extract timestamp level pool pump consumption data

        pp_out_idx = disagg_output_object.get('output_write_idx_map').get('pp')
        pp_disagg_output = copy.deepcopy(disagg_output_object.get('epoch_estimate')[:, pp_out_idx])

    # Set nans to 0 and if pp is absent set to empty array

    pp_disagg_output[np.isnan(pp_disagg_output)] = 0
    pp_cons_pts = np.sum(pp_disagg_output)

    # Extract timestamp level timed water heater consumption data

    twh_disagg_output = disagg_output_object.get("special_outputs").get("timed_water_heater")

    if twh_disagg_output is None:
        twh_disagg_output = np.zeros_like(disagg_output_object.get('epoch_estimate').shape[0])
    else:
        twh_disagg_output = twh_disagg_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Set nans to 0 and if twh is absent set to empty array

    twh_disagg_output[np.isnan(twh_disagg_output)] = 0
    twh_cons_pts = np.sum(twh_disagg_output)

    # Populate the timed disagg output dictionary

    timed_disagg_output = {
        'pp': {
            'cons': pp_disagg_output,
            'num_pts': pp_cons_pts,
        },
        'twh': {
            'cons': twh_disagg_output,
            'num_pts': twh_cons_pts,
        },
    }

    return timed_disagg_output


def remove_timed_appliance(disagg_output_object, hvac_input_data, logger_hvac, sampling_rate, disagg_input_object):
    """
        Function to remove timed appliances from entering HVAC module

        Parameters:
            disagg_output_object    (dict)              : Dictionary containing all outputs
            hvac_input_data         (np.ndarray)        : 2D array of epoch level consumption data
            logger_hvac             (logging object)    : Writes logs during code flow
            sampling_rate           (float)             : Float carrying sampling rate of user

        Returns:
            hvac_input_data         (np.ndarray)        : 2D array of epoch level consumption data
        """

    pilot_id = disagg_input_object['config']['pilot_id']
    timed_disagg_output = extract_pp_twh(disagg_output_object, pilot_id)

    pp_estimate = timed_disagg_output['pp']['cons']
    pp_num_pts = timed_disagg_output['pp']['num_pts']
    twh_estimate = timed_disagg_output['twh']['cons']
    twh_num_pts = timed_disagg_output['twh']['num_pts']

    disagg_input_object['switch']['timed_app'] = {'pp': {'removed': False,
                                                         'estimate': pp_estimate,
                                                         'pp_num_pts': pp_num_pts,
                                                         'confidence': 0
                                                         },

                                                  'twh': {'removed': False,
                                                          'estimate': twh_estimate,
                                                          'twh_num_pts': twh_num_pts,
                                                          'confidence': 0
                                                          }
                                                  }

    if (pp_num_pts > 0) & ('pp_confidence' in disagg_output_object['special_outputs'].keys()):

        pp_confidence = disagg_output_object['special_outputs']['pp_confidence']
        logger_hvac.info(' PP Confidence is {}%. |'.format(pp_confidence))

        disagg_input_object['switch']['timed_app']['pp']['confidence'] = pp_confidence

        if (pp_confidence is not None) and (pp_confidence >= 70):
            hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] - pp_estimate
            logger_hvac.info(' PP successfully removed before HVAC |')
            disagg_input_object['switch']['timed_app']['pp']['removed'] = True

    logger_hvac.info(' Sampling rate is {} . Time WH will be removed below 1800 only |'.format(sampling_rate))

    if (twh_num_pts > 0) & ('timed_wh_confidence' in disagg_output_object['special_outputs'].keys()):

        t_wh_confidence = disagg_output_object['special_outputs']['timed_wh_confidence']
        logger_hvac.info(' Timed WH Confidence is {}%. |'.format(t_wh_confidence * 100))

        disagg_input_object['switch']['timed_app']['twh']['confidence'] = t_wh_confidence

        if t_wh_confidence >= 0.5:
            hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] - twh_estimate
            logger_hvac.info(' TWH successfully removed before HVAC |')
            disagg_input_object['switch']['timed_app']['twh']['removed'] = True

    return hvac_input_data


def remove_ev(disagg_output_object, hvac_input_data, disagg_input_object):
    """
        Function to remove ev from entering HVAC module

        Parameters:
            disagg_output_object        (dict)              : Dictionary containing all outputs
            hvac_input_data             (np.ndarray)        : 2D array of epoch level consumption data
            disagg_input_object         (dict)              : Dictionary containing all useful inputs

        Returns:
            hvac_input_data             (np.ndarray)        : 2D array of epoch level consumption data
        """

    # Extract timestamp level ev consumption data
    ev_out_idx = disagg_output_object.get('output_write_idx_map').get('ev')
    ev_estimate = copy.deepcopy(disagg_output_object.get('epoch_estimate')[:, ev_out_idx])
    ev_estimate[np.isnan(ev_estimate)] = 0
    ev_net_output = np.sum(ev_estimate)

    disagg_input_object['switch']['timed_app']['ev'] = {'removed': False,
                                                        'estimate': ev_estimate,
                                                        'ev_num_pts': ev_net_output,
                                                        'confidence': 0
                                                        }

    hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] - ev_estimate
    disagg_input_object['switch']['timed_app']['ev']['removed'] = True

    return hvac_input_data


def hvac_module(disagg_input_object, disagg_output_object, logger_base, hvac_exit_status):
    """
    Function for detection of hvac amplitude and estimation of hvac epoch level consumption

    Parameters:
        disagg_input_object         (dict)              : Dictionary containing all inputs
        disagg_output_object        (dict)              : Dictionary containing all outputs
        logger_base                 (logging object)    : Writes logs during code flow
        hvac_exit_status            (dict)              : Dictionary containing hvac exit code and list of handled errors

    Returns:
        month_ao_hvac               (np.ndarray)        : 2D array of Month epoch and monthly cooling-heating estimates
        epoch_ao_hvac               (np.ndarray)        : 2D array of epoch time-stamp and epoch cooling-heating estimates
        hvac_debug                  (dict)              : Dictionary containing hvac stage related debugging information
        hsm_update                  (dict)              : Dictionary containing hsm parameters [historical/incremental run]
        hvac_exit_status            (dict)              : Dictionary containing hvac exit code and list of handled errors
    """
    t_start = datetime.now()

    logger_local = logger_base.get("logger").getChild("hvac_module")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    logger_hvac.info(' --------------------- HVAC Algo Start -------------------------- |')

    np.random.seed(12345)

    global_config = copy.deepcopy(disagg_input_object.get('config'))
    sampling_rate = disagg_input_object['config']['sampling_rate']

    # reading input data
    input_data = copy.deepcopy(disagg_input_object['input_data'])
    logger_hvac.info(' >> Describe Consumption : Sum {} Mean {}, Describe Temperature : Sum {} Mean {} |'.format(
        np.nansum(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]),
        np.nanmean(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]),
        np.nansum(input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]),
        np.nanmean(input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX])))

    epoch_data = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]

    # reading always on estimates, estimated in ao module
    algo_baseload = disagg_output_object['epoch_estimate'][:, disagg_output_object.get('output_write_idx_map').get('ao')]
    logger_hvac.info(' Got AO from algo. Describe : Sum {}, {} rows present |'.format(np.nansum(algo_baseload),
                                                                                      algo_baseload.shape[0]))

    hvac_input_data = copy.deepcopy(input_data)

    # Timed appliance removal
    hvac_input_data = remove_timed_appliance(disagg_output_object, hvac_input_data, logger_hvac, sampling_rate,
                                             disagg_input_object)
    hvac_input_data = remove_ev(disagg_output_object, hvac_input_data, disagg_input_object)

    hvac_input_data_timed_removed = copy.deepcopy(hvac_input_data)
    disagg_input_object['switch']['hvac_input_data_timed_removed'] = hvac_input_data_timed_removed
    logger_hvac.info(' Updated hvac input data consumption after removing >> timed appliances << |')

    # removing ao from net consumption, at epoch level and updating hvac_input_data array
    hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] - algo_baseload
    disagg_input_object['switch']['hvac_input_data_timed_ao_removed'] = hvac_input_data
    logger_hvac.info(' Removed algo-ao from Raw consumption |')

    # initializing hvac algo related key parameters
    hvac_params = init_hvac_params(sampling_rate, disagg_input_object, logger_hvac, logger_flag=True)
    logger_hvac.info(' HVAC parameters initiated for {} sampling rate |'.format(sampling_rate))

    # reading user profile related config

    global_config['switch'] = disagg_input_object['switch']

    if global_config['pilot_id'] in hvac_static_params['cold_pilots']:
        hvac_params['setpoint']['AC']['SETPOINTS'] = setpoint_list(80, 70, -1)

    # reading consumption and temperature related valid epochs
    valid_consumption = copy.deepcopy(disagg_input_object['data_quality_metrics']['is_valid_cons'])
    valid_temperature = copy.deepcopy(disagg_input_object['data_quality_metrics']['is_valid_temp'])
    valid_idx = valid_consumption & valid_temperature
    invalid_idx = np.logical_not(valid_idx)
    logger_hvac.info(' >> Total valid epoch points based on Energy and Temperature : {} |'.format(np.sum(valid_idx)))

    # initializing empty dictionary for storing hvac algo related key debugging information

    hvac_debug = {}
    if (global_config.get('disagg_mode') == 'historical') or (global_config.get('disagg_mode') == 'incremental'):
        logger_hvac.info(' --------------------- Pre-Pipeline -------------------------- |')
        sampling_rate = disagg_input_object['config']['sampling_rate']
        disagg_mode = disagg_input_object['config']['disagg_mode']
        timezone = disagg_input_object.get('home_meta_data').get('timezone')
        try:
            existing_hsm = disagg_input_object.get('appliances_hsm', {}).get('hvac', {}).get('attributes', {})
        except AttributeError:
            existing_hsm = {}

        # Get Initial Config object and get all user hvac consumption / overall temperature characteristics
        config = pre_detection_params(sampling_rate, hvac_params, hvac_input_data, timezone)
        user_parameters = get_user_characteristic(hvac_input_data, invalid_idx, config, hvac_params,
                                                  disagg_mode, existing_hsm, logger_pass)
        user_flags = get_user_cooling_flags(hvac_input_data, user_parameters, hvac_params, config)

        # Adjust cooling/heating parameters and update hvac_params object
        disagg_input_object, hvac_params = get_adjusted_hvac_parameters(disagg_input_object, user_flags,
                                                                        user_parameters, hvac_params, config, 'AC')
        disagg_input_object, hvac_params = get_adjusted_hvac_parameters(disagg_input_object, user_flags,
                                                                        user_parameters, hvac_params, config, 'SH')

        if 'pre_pipeline' not in hvac_debug.keys():
            hvac_debug.update({'pre_pipeline': {}})

        hvac_debug['pre_pipeline']['hvac'] = user_parameters
        hvac_debug['pre_pipeline']['all_flags'] = user_flags
        hvac_debug['pre_pipeline']['config'] = config
        hvac_debug['pre_pipeline']['all_indices'] = user_parameters['all_indices']

        t_end = datetime.now()
        logger_hvac.info('HVAC pre-detection pipeline complete | %.3f s |', get_time_diff(t_start, t_end))

        logger_hvac.info(' --------------------- Amplitude Estimation -------------------------- |')
        # detecting heating and cooling amplitudes
        hvac_debug['detection'] = detect_hvac_amplitude(hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                                                        hvac_input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX],
                                                        invalid_idx,
                                                        hvac_params, hvac_debug['pre_pipeline'],
                                                        logger_pass, global_config, hvac_exit_status)

        # if user profile confirms absence of ac or sh, enforcing cooling detection to be false
        base_ac_found = hvac_debug['detection']['cdd']['found']
        hvac_debug['detection']['cdd']['found'] = override_detection(disagg_input_object, 'ac', base_ac_found)

        # if user profile confirms absence of ac or sh, enforcing heating detection to be false
        base_sh_found = hvac_debug['detection']['hdd']['found']
        hvac_debug['detection']['hdd']['found'] = override_detection(disagg_input_object, 'sh', base_sh_found)

        if disagg_output_object['analytics']['required']:
            add_detection_analytics(disagg_output_object, hvac_debug)

    elif global_config.get('disagg_mode') == 'mtd':

        hvac_debug = get_hsm_attributes(disagg_input_object, disagg_output_object)

    logger_hvac.info(' --------------------- Estimation -------------------------- |')

    disagg_output_object['switch'] = disagg_input_object['switch']

    # estimating epoch level ac and sh consumption
    estimation_output = estimate_hvac(hvac_input_data, invalid_idx, hvac_debug, hvac_params,
                                      disagg_output_object, global_config, logger_pass, hvac_exit_status)

    x_hour_hvac_by_mode, epoch_hvac, estimation_debug, hvac_exit_status = estimation_output

    # assigning important attributes to hvac_debug dictionary
    hvac_debug['daily_output'] = x_hour_hvac_by_mode
    hvac_debug['hourly_output'] = epoch_hvac
    hvac_debug['estimation'] = estimation_debug

    # ensuring no nan estimates go into final appliance estimate array
    hvac_debug['hourly_output'][np.isnan(hvac_debug['hourly_output'])] = 0
    hvac_debug['hourly_output'][(hvac_debug['hourly_output']) < 0] = 0

    # getting month epoch identifiers, and epoch level markings of month epoch identifiers
    month_epoch, _, month_idx = scipy.unique(hvac_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                                             return_index=True, return_inverse=True)

    # aggregating ac estimates at month level for monthly estimates
    month_cool = np.bincount(month_idx, hvac_debug['hourly_output'][:, 0])
    logger_hvac.info('aggregated ac estimates at month level for monthly estimates |')

    # aggregating sh estimates at month level for monthly estimates
    month_heat = np.bincount(month_idx, hvac_debug['hourly_output'][:, 1])
    logger_hvac.info('aggregated sh estimates at month level for monthly estimates |')

    # aggregating always on estimates at month level for monthly estimates
    algo_baseload[np.isnan(algo_baseload)] = 0
    month_bl = np.bincount(month_idx, algo_baseload)
    logger_hvac.info('aggregated always on estimates at month level for monthly estimates |')

    # concatenating month level appliance estimates, to form consumption array
    month_ao_hvac = np.c_[month_epoch, month_bl, month_cool, month_heat]
    logger_hvac.info('concatenated month level appliance estimates, to form consumption array |')

    # concatenating epoch level appliance estimates, to form consumption array
    epoch_ao_hvac = np.c_[epoch_data, algo_baseload, hvac_debug['hourly_output']]
    logger_hvac.info('concatenated epoch level appliance estimates, to form consumption array |')

    if (global_config.get('disagg_mode') == 'historical') or (global_config.get('disagg_mode') == 'incremental'):

        logger_hvac.info('>> attempting to prepare dictionary for updating hsm |')

        attributes = {
            'hot_cold_normal_user_flag': [hvac_debug['pre_pipeline']['all_flags']['hot_cold_normal_user_flag']],
            'ac_low_consumption_user': [hvac_debug['pre_pipeline']['all_flags']['low_summer_consumption_user_flag']],
            'sh_low_consumption_user': [hvac_debug['pre_pipeline']['all_flags']['low_winter_consumption_user_flag']],
            'no_ac_user': [int(hvac_debug['pre_pipeline']['all_flags']['is_not_ac'])],
            'night_ac_user': [int(hvac_debug['pre_pipeline']['all_flags']['is_night_ac'])],
            'ac_setpoint': [hvac_debug['estimation']['cdd']['setpoint']],
            'sh_setpoint': [hvac_debug['estimation']['hdd']['setpoint']],
            'ac_aggregation_factor': [hvac_debug['estimation']['cdd']['aggregation_factor']],
            'sh_aggregation_factor': [hvac_debug['estimation']['hdd']['aggregation_factor']],
            'ac_cluster_info': hvac_debug['estimation']['cdd']['cluster_info']['hvac'],
            'sh_cluster_info': hvac_debug['estimation']['hdd']['cluster_info']['hvac'],
            'ac_setpoint_exist': [int(hvac_debug['estimation']['cdd']['exist'])],
            'sh_setpoint_exist': [int(hvac_debug['estimation']['hdd']['exist'])],
            'ac_found': [int(hvac_debug['detection']['cdd']['found'])],
            'sh_found': [int(hvac_debug['detection']['hdd']['found'])],
            'ac_means': hvac_debug['detection']['cdd']['amplitude_cluster_info']['means'],
            'sh_means': hvac_debug['detection']['hdd']['amplitude_cluster_info']['means'],
            'ac_std': hvac_debug['detection']['cdd']['amplitude_cluster_info']['std'],
            'sh_std': hvac_debug['detection']['hdd']['amplitude_cluster_info']['std'],
            'ac_mode_limits': hvac_debug['detection']['cdd']['amplitude_cluster_info']['cluster_limits'],
            'sh_mode_limits': hvac_debug['detection']['hdd']['amplitude_cluster_info']['cluster_limits'],
            'ac_mu': [float(hvac_debug['detection']['cdd']['mu'])],
            'sh_mu': [float(hvac_debug['detection']['hdd']['mu'])],
            'ac_number_of_modes': [hvac_debug['detection']['cdd']['amplitude_cluster_info']['number_of_modes']],
            'sh_number_of_modes': [hvac_debug['detection']['hdd']['amplitude_cluster_info']['number_of_modes']],
        }

        restructured_attributes = restructure_atrributes_for_hsm(attributes)

        hsm_update = dict({'timestamp': hvac_input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX]})
        hsm_update['attributes'] = restructured_attributes
        logger_hvac.info('prepared dictionary for updating with new hsm |')
    else:
        logger_hvac.info('monthly mode, no hsm attribute to be updated in hvac module |')

        # monthly mode, no hsm attribute to be updated in hvac module
        hsm_update = np.array([])

    # hvac module finished as expected, exiting with exit code : 1
    hvac_exit_status['exit_code'] = 1

    return month_ao_hvac, epoch_ao_hvac, hvac_debug, hsm_update, hvac_exit_status
