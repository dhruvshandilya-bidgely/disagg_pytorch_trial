"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to run smb hvac module
"""

# Import python packages
import copy
import scipy
import logging
import numpy as np
import pandas as pd

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aes.hvac.estimate_smb_hvac import estimate_smb_hvac
from python3.disaggregation.aer.hvac.get_hsm_attributes import get_hsm_attributes
from python3.disaggregation.aes.hvac.read_hvac_from_user import override_detection
from python3.disaggregation.aes.hvac.init_hourly_smb_hvac_params import init_hvac_params
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params
from python3.disaggregation.aes.hvac.restructure_hsm import restructure_atrributes_for_hsm
from python3.disaggregation.aes.hvac.detect_smb_hvac_amplitude import detect_hvac_amplitude
from python3.disaggregation.aes.work_hours.operational_and_xao.smb_utility import prepare_input_df


def add_detection_analytics(disagg_output_object, hvac_debug):
    """
    Function to populate hvac detection related parameters for 2nd order analytics

    Parameters:

        disagg_output_object        (dict)                          : Dictionary containing all outputs
        hvac_debug                  (dict)                          : dictionary containing hvac detection debug

    Return:
        None
    """

    # populating analytics attributes
    disagg_output_object['analytics']['values']['cooling'] = {'detection': hvac_debug.get('detection').get('cdd').get('amplitude_cluster_info')}
    disagg_output_object['analytics']['values']['heating'] = {'detection': hvac_debug.get('detection').get('hdd').get('amplitude_cluster_info')}


def remove_operational_load(disagg_input_object, hvac_input_data, parameters):
    """
    Function to remove a "provisional" Operational Load from raw data before OD HVAC estimations
    Parameters:
        disagg_input_object (dict)     : Dictionary containing all inputs from the pipeline
        hvac_input_data     (np.array) : 2D Array containing extracted info for HVAC disagg
        parameters          (dict)     : dictionary with SMB HVAC specific constants

    Returns:
        hvac_input_data     (np.array) : 2D Array containing extracted info for HVAC disagg
    """
    func_params = parameters.get('remove_operational_load')

    epoch_input_data = copy.deepcopy(disagg_input_object.get('input_data'))
    input_df = prepare_input_df(epoch_input_data, disagg_input_object)

    appliance_df = pd.DataFrame()
    appliance_df['consumption'] = hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    appliance_df['consumption'][appliance_df['consumption'] < 0] = 0
    appliance_df['temperature'] = input_df['temperature']
    appliance_df['date'] = input_df['date']
    appliance_df['time'] = input_df['time']
    energy_heatmap = appliance_df.pivot_table(index='date', columns=['time'], values='consumption', aggfunc=sum)
    energy_heatmap = energy_heatmap.fillna(0)

    open_close = copy.deepcopy(disagg_input_object.get('switch').get('smb').get('open_close_table'))

    # If open close was detected for the user, identify the bare minimum operational load for those work hours and
    # remove it before estimating HVAC. This prevents HVAC overestimation as well as provide enough residue for
    # Operational Load estimation later.
    if np.nansum(open_close) > 0:

        work_hour_proportion = np.sum(open_close, axis=1) / open_close.shape[1]
        median_work_hour_prop = np.nanmedian(work_hour_proportion[work_hour_proportion > 0])
        daily_cons = np.sum(energy_heatmap, axis=1)
        median_daily_cons = np.nanmedian(daily_cons)

        valid_work_hour_days_idx1 = np.where(np.logical_and(work_hour_proportion > 0, daily_cons < median_daily_cons))

        max_work_hour_prop = median_work_hour_prop / func_params.get('work_hour_prop_divisor')
        valid_work_hour_days_idx2 = np.where(abs(work_hour_proportion - median_work_hour_prop) < max_work_hour_prop)

        valid_work_hour_days_idx = [x for x in valid_work_hour_days_idx1[0] if x in valid_work_hour_days_idx2[0]]

        close_hours = np.where(open_close == 0, 1, 0)

        # removing the operational load before calculating HVAC
        total_consumption_open = energy_heatmap * open_close

        total_consumption_close = energy_heatmap * close_hours

        min_non_operational_load = np.tile(np.ma.median(np.ma.masked_equal(total_consumption_close, 0),
                                                        axis=1).data.reshape(-1, 1), total_consumption_open.shape[1])

        # good days are ideally NON_HVAC days
        good_days = np.take(total_consumption_close, valid_work_hour_days_idx, axis=0)

        # Check if epochs with 0 consumption is more than the allowed max_limit
        if np.nansum(good_days == 0) >= good_days.shape[0] * good_days.shape[1] * func_params.get('max_zero_data_days'):
            min_non_operational_load_overall = 0

        else:
            min_non_operational_load_overall = np.nanpercentile(good_days[good_days >=
                                                                          func_params.get('min_good_days')], 75)

        min_operational_load = np.nanpercentile(total_consumption_open[total_consumption_open >=
                                                                       func_params.get('min_epoch_cons')], 25)
        max_operational_load = np.nanpercentile(total_consumption_open[total_consumption_open >=
                                                                       func_params.get('min_epoch_cons')], 35)

        total_consumption_open = np.where(total_consumption_open > max_operational_load,
                                          total_consumption_open - max_operational_load,
                                          np.where(total_consumption_open <= min_operational_load, 0,
                                                   np.minimum(np.maximum(min_non_operational_load,
                                                                         min_non_operational_load_overall),
                                                              total_consumption_open)))

        energy_heatmap_n = np.where(open_close == 1, total_consumption_open, energy_heatmap)

        epoch_df = input_df.pivot_table(index='date', columns=['time'], values='epoch', aggfunc=np.min)
        epochs = epoch_df.values.flatten()
        energy_heatmap = energy_heatmap_n.flatten()
        _, idx_mem_1, idx_mem_2 = np.intersect1d(epochs, input_df['epoch'], return_indices=True)
        hvac_input_data[idx_mem_2, Cgbdisagg.INPUT_CONSUMPTION_IDX] = energy_heatmap[idx_mem_1]

    else:

        epoch_df = input_df.pivot_table(index='date', columns=['time'], values='epoch', aggfunc=np.min)
        epochs = epoch_df.values.flatten()
        energy_heatmap = energy_heatmap.values.flatten()
        _, idx_mem_1, idx_mem_2 = np.intersect1d(epochs, input_df['epoch'], return_indices=True)
        hvac_input_data[idx_mem_2, Cgbdisagg.INPUT_CONSUMPTION_IDX] = energy_heatmap[idx_mem_1]

    disagg_input_object['switch']['hvac']['operational_removed'] = copy.deepcopy(hvac_input_data
                                                                                 [:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    return hvac_input_data


def setpoint_list(start, stop, step=1):
    """
    Float list generation

    Parameters :
        start           (int)           : start temperature
        stop            (int)           : end temperature
        step            (int)           : steps of temperature
    Return :
        setpoint_list   (np.ndarray)    : Array containing setpoint temperatures
    """
    # creating setpoint list
    setpoint_list = np.array(list(range(start, stop + step, step)))

    return setpoint_list


def smb_hvac(disagg_input_object, disagg_output_object, logger_base, hvac_exit_status):
    """
    Function for detection of hvac amplitude and estimation of hvac epoch level consumption

    Parameters:

        disagg_input_object     (dict)              : Dictionary containing all inputs
        disagg_output_object    (dict)              : Dictionary containing all outputs
        logger_base             (logging object)    : Writes logs during code flow
        hvac_exit_status        (dict)              : Dictionary containing hvac exit code and list of handled errors

    Returns:

        month_ao_hvac       (numpy array)          : 2D array of Month epoch and monthly cooling-heating estimates
        epoch_ao_hvac       (numpy array)          : 2D array of epoch time-stamp and epoch cooling-heating estimates
        hvac_debug          (dict)                 : Dictionary containing hvac stage related debugging information
        hsm_update          (dict)                 : Dictionary containing hsm parameters [historical/incremental run]
        hvac_exit_status    (dict)                 : Dictionary containing hvac exit code and list of handled errors
    """

    static_params = hvac_static_params()

    logger_local = logger_base.get("logger").getChild("hvac_module")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    logger_hvac.info(' --------------------- HVAC Algo Start -------------------------- |')

    np.random.seed(12345)

    # reading input data
    input_data = copy.deepcopy(disagg_input_object.get('input_data'))
    logger_hvac.info(' >> Describe Consumption : Sum {} Mean {}, Describe Temperature : Sum {} Mean {} |'.format(
        np.nansum(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]), np.nanmean(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]),
        np.nansum(input_data[:, Cgbdisagg.INPUT_FEELS_LIKE_IDX]), np.nanmean(input_data[:, Cgbdisagg.INPUT_FEELS_LIKE_IDX])))

    input_data[np.isnan(input_data)] = 0

    epoch_data = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]

    # reading always on estimates, estimated in ao module
    algo_baseload = disagg_output_object.get('epoch_estimate')[:, disagg_output_object.get('output_write_idx_map').get('ao_smb')]

    logger_hvac.info(' Got AO from algo. Describe : Sum {}, {} rows present | '.format(np.nansum(algo_baseload),
                                                                                       algo_baseload.shape[0]))

    # reading External Lighting estimated earlier
    external_lighting = disagg_output_object.get('epoch_estimate')[:, disagg_output_object.get('output_write_idx_map').get('li_smb')]

    hvac_input_data = copy.deepcopy(input_data)

    # removing ao from net consumption, at epoch level and updating hvac_input_data array
    hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] -= algo_baseload
    logger_hvac.info(' Removed algo-ao from Raw consumption |')

    # removing external lighting from net consumption, at epoch level and updating hvac_input_data array
    hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] -= np.nan_to_num(external_lighting)
    logger_hvac.info(' Removed algo-external lighting from Raw consumption |')

    ####
    # SMB v2.0 improvement
    hvac_input_data = remove_operational_load(disagg_input_object, hvac_input_data, static_params)
    ####

    # Carrying hvac input data in switch for hvac internal functions
    disagg_input_object['switch']['hvac_input_data'] = hvac_input_data

    # initializing hvac algo related key parameters
    sampling_rate = disagg_input_object['config']['sampling_rate']
    hvac_params = init_hvac_params(sampling_rate, disagg_input_object, logger_hvac, logger_flag=True)
    logger_hvac.info(' HVAC parameters initiated for {} sampling rate |'.format(sampling_rate))

    # reading user profile related config
    global_config = copy.deepcopy(disagg_input_object.get('config'))
    global_config['switch'] = disagg_input_object.get('switch')

    if global_config['pilot_id'] in static_params['cold_pilots']:
        hvac_params['setpoint']['AC']['SETPOINTS'] = setpoint_list(static_params['cold_setpoint_high_lim'],
                                                                   static_params['cold_setpoint_low_lim'], -1)

    # reading consumption and temperature related valid epochs
    valid_consumption = copy.deepcopy(disagg_input_object.get('data_quality_metrics').get('is_valid_cons'))
    valid_temperature = copy.deepcopy(disagg_input_object.get('data_quality_metrics').get('is_valid_temp'))
    valid_idx = valid_consumption & valid_temperature
    invalid_idx = np.logical_not(valid_idx)
    logger_hvac.info(' >> Total valid epoch points based on Energy and Temperature : {} |'.format(np.sum(valid_idx)))

    # initializing empty dictionary for storing hvac algo related key debugging information

    hvac_debug = {}
    if (global_config.get('disagg_mode') == 'historical') or (global_config.get('disagg_mode') == 'incremental'):

        # if user profile doesnt have hsm field, initializing it to avoid crash
        if not hasattr(global_config, 'hsm'):
            logger_hvac.info('initializing hvac hsm |')

        logger_hvac.info(' --------------------- Amplitude Estimation -------------------------- |')

        # detecting heating and cooling amplitudes
        hvac_debug['detection'] = detect_hvac_amplitude(hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                                                        hvac_input_data[:, Cgbdisagg.INPUT_FEELS_LIKE_IDX],
                                                        invalid_idx,
                                                        hvac_params, logger_pass, global_config, hvac_exit_status)

        # if user profile confirms absence of ac or sh, enforcing cooling detection to be false
        base_ac_found = hvac_debug.get('detection').get('cdd').get('found')
        hvac_debug['detection']['cdd']['found'] = override_detection(disagg_input_object, 'ac', base_ac_found)

        # if user profile confirms absence of ac or sh, enforcing heating detection to be false
        base_sh_found = hvac_debug.get('detection').get('hdd').get('found')
        hvac_debug['detection']['hdd']['found'] = override_detection(disagg_input_object, 'sh', base_sh_found)

        if disagg_output_object['analytics']['required']:
            add_detection_analytics(disagg_output_object, hvac_debug)

    elif global_config.get('disagg_mode') == 'mtd':

        hvac_debug = get_hsm_attributes(disagg_input_object, disagg_output_object)

    logger_hvac.info(' --------------------- Estimation -------------------------- |')

    disagg_output_object['switch'] = disagg_input_object.get('switch')

    # estimating epoch level ac and sh consumption
    estimation_output = estimate_smb_hvac(hvac_input_data, invalid_idx, hvac_debug, hvac_params, global_config,
                                          logger_pass, hvac_exit_status, disagg_output_object)

    x_hour_hvac_by_mode, epoch_hvac, estimation_debug, hvac_exit_status = estimation_output

    # assigning important attributes to hvac_debug dictionary
    hvac_debug['daily_output'] = x_hour_hvac_by_mode
    hvac_debug['hourly_output'] = epoch_hvac
    hvac_debug['estimation'] = estimation_debug

    # ensuring no nan estimates go into final appliance estimate array
    hvac_debug['hourly_output'][np.isnan(hvac_debug['hourly_output'])] = 0
    hvac_debug['hourly_output'][(hvac_debug['hourly_output']) < 0] = 0

    # getting month epoch identifiers, and epoch level markings of month epoch identifiers
    month_epoch, idx_2, month_idx = scipy.unique(hvac_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_index=True,
                                                 return_inverse=True)

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

    # creating attributes dictionary
    if (global_config.get('disagg_mode') == 'historical') or (global_config.get('disagg_mode') == 'incremental'):

        logger_hvac.info('>> attempting to prepare dictionary for updating hsm |')

        attributes = {
            'ac_setpoint': [hvac_debug['estimation']['cdd']['setpoint']],
            'sh_setpoint': [hvac_debug['estimation']['hdd']['setpoint']],
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

        # restructuring attributes for hsm
        restructured_attributes = restructure_atrributes_for_hsm(attributes)

        # assigning timestamp to hsm
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
