"""
Author - Kris Duszak
Date - 3/25/2019
Hourly HVAC Computation for Japan
"""

# Import python packages

import copy
import scipy
import logging
import numpy as np
from scipy.stats import mode

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.hvac.estimate_hvac import estimate_hvac
from python3.disaggregation.aer.hvac.adjust_baseload import adjust_baseload
from python3.disaggregation.aer.hvac.get_hsm_attributes import get_hsm_attributes

from python3.disaggregation.aer.hvac.hvac_japan_functions.init_hourly_hvac_params_jp import init_hvac_params_jp
from python3.disaggregation.aer.hvac.hvac_japan_functions.detect_hvac_amplitude_jp import detect_hvac_amplitude_jp
from python3.disaggregation.aer.hvac.hvac_japan_functions.compute_daily_baseload import compute_daily_baseload


def stabilize_setpoint_range(global_config, hvac_params, logger_base):

    """
    Function to reduce room for change of setpoint with respect to last detected setpoint, read from hsm

    Parameters:
        global_config (dict)                    : Dictionary containing user profile related information
        hvac_params(dict)                       : Dictionary containing hvac algo related initialized parameters
        logger_base(logging object)             : Writes logs during code flow

    Returns:
        hvac_params (dict)                      : Dictionary containing hvac algo related initialized parameters
    """

    logger_local = logger_base.get("logger").getChild("stabilize_setpoint")
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # reading ac setpoint from previously saved hsm

    previous_ac_setpoint = global_config['previous_hsm']['setPoint'][0]
    logger_hvac.debug('>> ac setpoint read from hsm is {}F |'.format(previous_ac_setpoint))

    # limiting setpoint variability of 5F around last saved setpoint

    min_ac_setpoint_possible = max((previous_ac_setpoint - 5), min(hvac_params['setpoint']['AC']['SETPOINTS']))
    max_ac_setpoint_possible = min((previous_ac_setpoint + 5), max(hvac_params['setpoint']['AC']['SETPOINTS']))

    # assigning adjusted setpoint range to hvac initializing parameters

    hvac_params['setpoint']['AC']['SETPOINTS'] = np.arange(min_ac_setpoint_possible, max_ac_setpoint_possible)
    logger_hvac.debug('>> new stabilized ac setpoint range is : | {}F - {}F '.format(min_ac_setpoint_possible,
                                                                                     max_ac_setpoint_possible))

    # reading sh setpoint from previously saved hsm

    previous_sh_setpoint = global_config['previous_hsm']['setPoint'][1]
    logger_hvac.debug('>> sh setpoint read from hsm is | {}F'.format(previous_sh_setpoint))

    # limiting setpoint variability of 5F around last saved setpoint

    min_sh_setpoint_possible = max((previous_sh_setpoint - 5), min(hvac_params['setpoint']['SH']['SETPOINTS']))
    max_sh_setpoint_possible = min((previous_sh_setpoint + 5), max(hvac_params['setpoint']['SH']['SETPOINTS']))

    # assigining adjusted setpoint range to hvac initializing parameters

    hvac_params['setpoint']['SH']['SETPOINTS'] = np.arange(min_sh_setpoint_possible, max_sh_setpoint_possible)
    logger_hvac.debug('>> new stabilized sh setpoint range is : | {}F - {}F'.format(min_sh_setpoint_possible,
                                                                                    max_sh_setpoint_possible))

    return hvac_params


def get_heat_count(app_profile):

    """Utility to get heating appliance count"""

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

    """Utility to get cooling appliance count"""

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

    """Overrides detection if appliance profile says no"""

    app_count = -1

    if appliance == 'sh':
        app_count = get_heat_count(disagg_input_object['app_profile'])
    elif appliance == 'ac':
        app_count = get_cool_count(disagg_input_object['app_profile'])

    if app_count == 0:
        return False
    else:
        return base_found


def hvac_module_jp(disagg_input_object, disagg_output_object, logger_base, hvac_exit_status):

    """
    Function for detection of hvac amplitude and estimation of hvac epoch level consumption

    Parameters:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
        logger_base(logging object)             : Writes logs during code flow
        hvac_exit_status(dict)                  : Dictionary containing hvac exit code and list of handled errors

    Returns:
        month_ao_hvac (numpy array)             : 2D array of Month epoch and monthly cooling-heating estimates
        epoch_ao_hvac (numpy array)             : 2D array of epoch time-stamp and epoch cooling-heating estimates
        hvac_debug (dict)                       : Dictionary containing hvac stage related debugging information
        hsm_update (dict)                       : Dictionary containing hsm parameters [historical/incremental run]
        hvac_exit_status (dict)                 : Dictionary containing hvac exit code and list of handled errors
    """

    logger_local = logger_base.get("logger").getChild("hvac_module")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_base.get("logging_dict")}

    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    logger_hvac.info(' --------------------- HVAC Algo Start -------------------------- |')

    np.random.seed(12345)

    # reading input data
    input_data = copy.deepcopy(disagg_input_object['input_data'])

    # reading epoch level consumption from input data
    epoch_data = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]

    # reading always on estimates, estimated in ao module
    algo_baseload = disagg_output_object['epoch_estimate'][:, disagg_output_object.get('output_write_idx_map').get('ao')]
    logger_hvac.info('got ao from ao-algo. {} rows present |'.format(algo_baseload.shape[0]))

    # reading always on estimates, estimated in ao module
    algo_daily_baseload = compute_daily_baseload(input_data)
    logger_hvac.info('got daily ao from daily ao-algo. {} rows present |'.format(algo_daily_baseload.shape[0]))

    # impute missing baseload with mode
    algo_daily_baseload[np.isnan(algo_daily_baseload)] = float(mode(algo_daily_baseload)[0])

    hvac_input_data = copy.deepcopy(input_data)

    # removing daily ao from net consumption at epoch level
    hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] - algo_daily_baseload
    logger_hvac.info('removed daily ao from raw |')

    sampling_rate = disagg_input_object['config']['sampling_rate']

    # initializing Japan hvac algo related key parameters
    hvac_params = init_hvac_params_jp(sampling_rate)
    logger_hvac.info('hvac parameters for Japan initiated |')

    # reading user profile related config
    global_config = copy.deepcopy(disagg_input_object.get('config'))

    # reduce room for change of setpoint in incremental run, using last stored setpoint info from last hsm
    if global_config.get('disagg_mode') == 'incremental':
        try:
            logger_hvac.info('incremental mode: stabilizing setpoint using hsm |')
            global_config['previous_hsm'] = disagg_input_object['appliances_hsm']['hvac']['attributes']
            hvac_params = stabilize_setpoint_range(global_config, hvac_params, logger_pass)
            logger_hvac.info('incremental mode: setpoint stabilized |')
        except Exception as exc:
            logger_hvac.debug('incremental mode: setpoint range not stabilized' + str(exc) + ' |')
            hvac_exit_status['error_list'].append('incremental mode: setpoint range not stabilized')

    # reading consumption related valid epochs
    valid_consumption = copy.deepcopy(disagg_input_object['data_quality_metrics']['is_valid_cons'])

    # reading temperature related valid epochs
    valid_temperature = copy.deepcopy(disagg_input_object['data_quality_metrics']['is_valid_temp'])

    # overall valid epoch has to be both valid consumption and valid temperature
    valid_idx = valid_consumption & valid_temperature
    logger_hvac.info('>> {} valid ET epoch points identified, for hvac disagg |'.format(np.sum(valid_idx)))

    invalid_idx = np.logical_not(valid_idx)

    # initializing empty dictionary for storing hvac algo related key debugging information

    hvac_debug = {}
    if (global_config.get('disagg_mode') == 'historical') or (global_config.get('disagg_mode') == 'incremental'):

        # if user profile doesnt have hsm field, initializing it to avoid crash
        if not hasattr(global_config, 'hsm'):
            logger_hvac.info('initializing hvac hsm |')

        logger_hvac.info(' --------------------- Amplitude Estimation -------------------------- |')
        logger_hvac.info('attempting to detect heating and cooling amplitudes |')

        vacation_periods = disagg_output_object.get('special_outputs').get('vacation_periods')

        # detecting heating and cooling amplitudes
        hvac_debug['detection'] = detect_hvac_amplitude_jp(hvac_input_data,
                                                           invalid_idx,
                                                           hvac_params,
                                                           vacation_periods,
                                                           logger_pass,
                                                           hvac_exit_status)

        logger_hvac.info(' --------------------- Adjust Baseload -------------------------- |')

        # modifying always on epoch level estimates, so that hvac doesn't creep in ao. ao stabilizing feature.
        [baseload, past_baseload, min_baseload] = adjust_baseload(hvac_input_data, algo_baseload,
                                                                  hvac_params, logger_pass)

        # adding daily ao from ao daily algo to get original epoch level consumption
        hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = \
            hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] + algo_daily_baseload

        # removing new ao from original consumption, at epoch level and storing in hvac_input_data
        hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = \
            hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] - baseload

        # assigning important attributes to hvac_debug dictionary
        hvac_debug['pastBaseLoad'] = past_baseload
        hvac_debug['minBaseLoad'] = min_baseload

        logger_hvac.info(' --------------------- Re: Amplitude Estimation -------------------------- |')

        # re: detecting heating and cooling amplitudes, after always on stabilization
        temp = detect_hvac_amplitude_jp(hvac_input_data,
                                        invalid_idx,
                                        hvac_params,
                                        vacation_periods,
                                        logger_pass,
                                        hvac_exit_status)

        # assigning important attributes to hvac_debug dictionary
        hvac_debug['detection']['hdd'] = temp['hdd']
        hvac_debug['detection']['cdd'] = temp['cdd']

        # if user profile confirms absence of ac or sh, enforcing cooling detection to be false
        base_ac_found = hvac_debug['detection']['cdd']['found']
        hvac_debug['detection']['cdd']['found'] = override_detection(disagg_input_object,
                                                                     'ac', base_ac_found)

        # if user profile confirms absence of ac or sh, enforcing heating detection to be false
        base_sh_found = hvac_debug['detection']['hdd']['found']
        hvac_debug['detection']['hdd']['found'] = override_detection(disagg_input_object,
                                                                     'sh', base_sh_found)

    elif global_config.get('disagg_mode') == 'mtd':

        # in mtd mode, hvac algo related parameters are read from last saved hsm
        # HSM parameters for debug.detection
        hvac_debug = get_hsm_attributes(disagg_input_object, disagg_output_object)

        # ensuring always on doesnt have creeped in hvac in it
        baseload, past_baseload, min_baseload = adjust_baseload(hvac_input_data, algo_baseload, hvac_params,
                                                                logger_pass, hvac_debug['detection']['pastBaseLoad'],
                                                                hvac_debug['detection']['minBaseLoad'])
    else:
        # ensuring appropriate fail-safe ao is always available for further code flow
        baseload = algo_baseload

    logger_hvac.info(' --------------------- Estimation -------------------------- |')

    # estimating epoch level ac and sh consumption if sh or ac detected
    daily_output, hourly_output, estimation_debug, regression_debug, hvac_exit_status = \
    estimate_hvac(hvac_input_data, invalid_idx, hvac_debug['detection'], hvac_params, global_config, logger_pass,
                  hvac_exit_status)

    # assigning important attributes to hvac_debug dictionary
    hvac_debug['daily_output'] = daily_output
    hvac_debug['hourly_output'] = hourly_output
    hvac_debug['setpoint'] = estimation_debug
    hvac_debug['global_regression'] = regression_debug

    # ensuring no nan estimates go into final appliance estimate array
    hvac_debug['hourly_output'][np.isnan(hvac_debug['hourly_output'])] = 0

    # getting month epoch identifiers, and epoch level markings of month epoch identifiers
    month_epoch, idx_2, month_idx = scipy.unique(hvac_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_index = True,
                                                 return_inverse = True)

    # aggregating ac estimates at month level for monthly estimates
    month_cool = np.bincount(month_idx, hvac_debug['hourly_output'][:, 0])
    logger_hvac.info('aggregated ac estimates at month level for monthly estimates |')

    # aggregating sh estimates at month level for monthly estimates
    month_heat = np.bincount(month_idx, hvac_debug['hourly_output'][:, 1])
    logger_hvac.info('aggregated sh estimates at month level for monthly estimates |')

    # aggregating always on estimates at month level for monthly estimates

    baseload[np.isnan(baseload)] = 0

    month_bl = np.bincount(month_idx, baseload)
    logger_hvac.info('aggregated always on estimates at month level for monthly estimates |')

    # concatenating month level appliance estimates, to form consumption array
    month_ao_hvac = np.c_[month_epoch, month_bl, month_cool, month_heat]
    logger_hvac.info('concatenated month level appliance estimates, to form consumption array |')

    # concatenating epoch level appliance estimates, to form consumption array
    epoch_ao_hvac = np.c_[epoch_data, baseload, hvac_debug['hourly_output']]
    logger_hvac.info('concatenated epoch level appliance estimates, to form consumption array |')

    if (global_config.get('disagg_mode') == 'historical') or (global_config.get('disagg_mode') == 'incremental'):
        logger_hvac.info('>> attempting to prepare dictionary for updating hsm |')

        # attempting to prepare dictionary for updating hsm
        attributes = {
            'setPoint': [hvac_debug['setpoint']['cdd']['setpoint'], hvac_debug['setpoint']['hdd']['setpoint']],
            'coeff': [hvac_debug['global_regression']['ccoeff'], hvac_debug['global_regression']['hcoeff']],
            'coeffAO': np.array([None, None]),
            'found': [hvac_debug['detection']['cdd']['found'], hvac_debug['detection']['hdd']['found']],
            'inactiveConsumption': None,
        }

        hsm_update = dict({'timestamp': hvac_input_data[-1, Cgbdisagg.INPUT_BILL_CYCLE_IDX]})
        hsm_update['attributes'] = attributes
        logger_hvac.info('prepared dictionary for updating with new hsm |')
    else:
        logger_hvac.info('monthly mode, no hsm attribute to be updated in hvac module |')

        # monthly mode, no hsm attribute to be updated in hvac module
        hsm_update = np.array([])

    # hvac module finished as expected, exiting with exit code : 1
    hvac_exit_status['exit_code'] = 1

    return month_ao_hvac, epoch_ao_hvac, hvac_debug, hsm_update, hvac_exit_status
