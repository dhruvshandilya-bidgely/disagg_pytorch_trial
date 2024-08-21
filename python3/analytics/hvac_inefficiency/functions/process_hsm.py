"""
Author - Anand Kumar Singh
Date - 26th June 2021
HVAC inefficiency HSM related operations
"""

# Import python packages

import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.mappings.get_app_id import get_app_id
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params
from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile as super_percentile


def get_fcc(output_inefficiency_object):

    """
    Function to get fcc

    Parameters:
        output_inefficiency_object  (dict)                      : Dictionary of inefficiency
    Returns:
        ac_fcc                      (float)                      : AC FCC
        sh_fcc                      (float)                      : SH FCC
    """

    ac_fcc = output_inefficiency_object.get('ac', {}).get('cycling_debug_dictionary', {}).get('full_cycle_consumption')
    if bool(ac_fcc):
        ac_fcc = float(ac_fcc)

    sh_fcc = output_inefficiency_object.get('sh', {}).get('cycling_debug_dictionary', {}).get('full_cycle_consumption')
    if bool(sh_fcc):
        sh_fcc = float(sh_fcc)

    return ac_fcc, sh_fcc


def get_pre_sat_fraction(output_inefficiency_object):

    """
    Function to get pre saturation fraction

    Parameters:
        output_inefficiency_object  (dict)                          : Dictionary of inefficiency
    Returns:
        ac_pre_sat_frac             (float)                         : AC Pre sat fraction
        sh_pre_sat_frac             (float)                         : SH Pre sat fraction
    """

    ac_pre_sat_frac = \
        output_inefficiency_object.get('ac', {}).get('cycling_debug_dictionary', {}).get('pre_saturation_fraction')
    if bool(ac_pre_sat_frac):
        ac_pre_sat_frac = float(ac_pre_sat_frac)

    sh_pre_sat_frac = \
        output_inefficiency_object.get('sh', {}).get('cycling_debug_dictionary', {}).get('pre_saturation_fraction')
    if bool(sh_pre_sat_frac):
        sh_pre_sat_frac = float(sh_pre_sat_frac)

    return ac_pre_sat_frac, sh_pre_sat_frac


def get_sat_fraction(output_inefficiency_object):

    """
        Function to get saturation fraction

        Parameters:
            output_inefficiency_object  (dict)                              : Dictionary of inefficiency
        Returns:
            ac_sat_frac                 (float)                             : AC sat fraction
            sh_sat_frac                 (float)                             : SH sat fraction
        """

    ac_sat_frac = \
        output_inefficiency_object.get('ac', {}).get('cycling_debug_dictionary', {}).get('saturation_fraction')
    if bool(ac_sat_frac):
        ac_sat_frac = float(ac_sat_frac)

    sh_sat_frac = \
        output_inefficiency_object.get('sh', {}).get('cycling_debug_dictionary', {}).get('saturation_fraction')
    if bool(sh_sat_frac):
        sh_sat_frac = float(sh_sat_frac)

    return ac_sat_frac, sh_sat_frac


def prepare_hsm(disagg_input_object, input_inefficiency_object, output_inefficiency_object):
    """
        Parameters:
            disagg_input_object                 (dict)          Dictionary containing all inputs
            input_hvac_inefficiency_object      (dict)          Dictionary containing all input information
            output_hvac_inefficiency_object     (dict)          Dictionary containing all output information

        Returns:
            hvac_inefficiency_hsm               (dict)          Dictionary containing hvac inefficiency HSM
    """

    static_params = hvac_static_params()

    # Prepare HVAC inefficiency HSM

    ac_relationship =\
        output_inefficiency_object.get('ac', {}).get('cycling_debug_dictionary', {}).get('duty_cycle_relationship')

    sh_relationship = \
        output_inefficiency_object.get('sh', {}).get('cycling_debug_dictionary', {}).get('duty_cycle_relationship')

    ac_fcc, sh_fcc = get_fcc(output_inefficiency_object)

    ac_pre_sat_frac, sh_pre_sat_frac = get_pre_sat_fraction(output_inefficiency_object)

    ac_pre_sat_temp = \
        output_inefficiency_object.get('ac', {}).get('cycling_debug_dictionary', {}).get('pre_saturation_temperature')
    sh_pre_sat_temp = \
        output_inefficiency_object.get('sh', {}).get('cycling_debug_dictionary', {}).get('pre_saturation_temperature')

    ac_sat_temp = \
        output_inefficiency_object.get('ac', {}).get('cycling_debug_dictionary', {}).get('saturation_temperature')
    sh_sat_temp = \
        output_inefficiency_object.get('sh', {}).get('cycling_debug_dictionary', {}).get('saturation_temperature')

    ac_sat_frac, sh_sat_frac = get_sat_fraction(output_inefficiency_object)

    ac_high_cons = \
        output_inefficiency_object.get('ac', {}).get('cycling_debug_dictionary', {}).get('compressor')

    sh_high_cons = \
        output_inefficiency_object.get('sh', {}).get('cycling_debug_dictionary', {}).get('compressor')

    if np.any(ac_high_cons):
        ac_high_cons = super_percentile(ac_high_cons, static_params.get('ineff').get('hsm_high_cons_quant') * 100)
        ac_high_cons = float(ac_high_cons)
    else:
        ac_high_cons = None

    if np.any(sh_high_cons):
        sh_high_cons = super_percentile(sh_high_cons, static_params.get('ineff').get('hsm_high_cons_quant') * 100)
        sh_high_cons = float(sh_high_cons)
    else:
        sh_high_cons = None

    duty_cycle_col = 1
    temperature_col = 0

    if np.any(ac_relationship):
        ac_duty_cycle = ac_relationship[:, duty_cycle_col]
        ac_duty_cycle = ac_duty_cycle.tolist()
        ac_temp = ac_relationship[:, temperature_col]
        ac_temp = ac_temp.tolist()
    else:
        ac_temp = None
        ac_duty_cycle = None

    if np.any(sh_relationship):
        sh_duty_cycle = sh_relationship[:, duty_cycle_col]
        sh_duty_cycle = sh_duty_cycle.tolist()
        sh_temp = sh_relationship[:, temperature_col]
        sh_temp = sh_temp.tolist()
    else:
        sh_temp = None
        sh_duty_cycle = None

    if (type(ac_sat_temp) == str) | (ac_sat_temp == None):
        ac_sat_temp = None
    else:
        ac_sat_temp = float(ac_sat_temp)

    if (type(ac_pre_sat_temp) == str) | (ac_pre_sat_temp == None):
        ac_pre_sat_temp = None
    else:
        ac_pre_sat_temp = float(ac_pre_sat_temp)

    if (type(sh_pre_sat_temp) == str) | (sh_pre_sat_temp == None):
        sh_pre_sat_temp = None
    else:
        sh_pre_sat_temp = float(sh_pre_sat_temp)

    if (type(sh_sat_temp) == str) | (sh_sat_temp == None):
        sh_sat_temp = None
    else:
        sh_sat_temp = float(sh_sat_temp)

    season_summary = input_inefficiency_object.get('seasons')

    # Creating HSM dictionary

    hvac_inefficiency_hsm = {
        'ac_fcc': ac_fcc,
        'ac_temp': ac_temp,
        'ac_duty_cycle': ac_duty_cycle,
        'ac_pre_sat_frac': ac_pre_sat_frac,
        'ac_pre_sat_temp': ac_pre_sat_temp,
        'ac_sat_temp': ac_sat_temp,
        'ac_sat_frac': ac_sat_frac,
        'ac_high_cons': ac_high_cons,
        'sh_fcc': sh_fcc,
        'sh_temp': sh_temp,
        'sh_duty_cycle': sh_duty_cycle,
        'sh_pre_sat_frac': sh_pre_sat_frac,
        'sh_pre_sat_temp': sh_pre_sat_temp,
        'sh_sat_temp': sh_sat_temp,
        'sh_sat_frac': sh_sat_frac,
        'sh_high_cons': sh_high_cons}

    # concatenating season summary and hvac inefficiency hsm
    hvac_inefficiency_hsm.update(season_summary)

    last_timestamp = disagg_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX]

    # Updating the required format

    hvac_inefficiency_hsm = {'timestamp': last_timestamp,
                             'attributes': hvac_inefficiency_hsm}

    return hvac_inefficiency_hsm


def use_previous_hsm(disagg_input_object):

    """
    This function extract information from previously stored HSMs

        Parameters:
            disagg_input_object     (dict)              Dictionary containing all inputs

        Returns:
            hsm_basic_info          (numpy.ndarray)     Numpy Array containing HSMs information
            hsm_basic_info          (dict)              Dictionary containing Duty Cycle Relations
    """

    # Extract app id and all hsms
    app_name = 'hvac'
    app_id = get_app_id(app_name)
    all_hsms = disagg_input_object.get('all_hsms_appliance', dict({}))

    ineff_hsm = all_hsms.get(str(app_id))

    hsm_basic_info = list([])
    hsm_duty_cycle_info = dict({})

    for timestamp in ineff_hsm:
        attributes = ineff_hsm.get(timestamp)[0].get('attributes')
        full_summer = attributes.get('full_summer', [None])[0]
        full_winter = attributes.get('full_winter', [None])[0]
        enough_summer = attributes.get('enough_summer', [None])[0]
        enough_winter = attributes.get('enough_winter', [None])[0]
        sh_high_cons = attributes.get('sh_high_cons', [None])[0]
        ac_high_cons = attributes.get('ac_high_cons', [None])[0]
        ac_fcc = attributes.get('ac_fcc', [None])[0]
        sh_fcc = attributes.get('sh_fcc', [None])[0]
        sh_pre_sat_frac = attributes.get('sh_pre_sat_frac', [None])[0]
        ac_pre_sat_frac = attributes.get('ac_pre_sat_frac', [None])[0]
        ac_temp = attributes.get('ac_temp', [None])
        sh_temp = attributes.get('sh_temp', [None])
        ac_duty_cycle = attributes.get('ac_duty_cycle', [None])
        sh_duty_cycle = attributes.get('sh_duty_cycle', [None])

        hsm_time = int(timestamp)

        single_hsm = list([hsm_time, full_summer, enough_summer, full_winter, enough_winter, ac_high_cons,
                           sh_high_cons, ac_fcc, sh_fcc, ac_pre_sat_frac, sh_pre_sat_frac])
        hsm_basic_info.append(single_hsm)

        single_hsm_dc_relation = {hsm_time:{'ac_duty_cycle': ac_duty_cycle,
                                            'sh_duty_cycle': sh_duty_cycle,
                                            'ac_temp': ac_temp,
                                            'sh_temp': sh_temp}}

        hsm_duty_cycle_info.update(single_hsm_dc_relation)

    hsm_basic_info = np.array(hsm_basic_info, dtype=np.float)

    col_map = {'hsm_time': 0,
               'full_summer': 1,
               'enough_summer': 2,
               'full_winter': 3,
               'enough_winter': 4,
               'ac_high_cons': 5,
               'sh_high_cons': 6,
               'ac_fcc': 7,
               'sh_fcc': 8,
               'ac_pre_sat_frac': 9,
               'sh_pre_sat_frac': 10}

    return hsm_basic_info, col_map, hsm_duty_cycle_info


def find_valid_hsm(input_inefficiency_object, device, logger_pass):

    """
        This function extract valid HSMs for

        Parameters:
            input_inefficiency_object       (dict)              Dictionary containing all inputs
            device                          (str)               Device for which HSMs are being captured
            logger_pass                     (object)            Logger object

        Returns:
            input_inefficiency_object       (dict)              Dictionary containing all inputs
    """

    static_params = hvac_static_params()

    logger_local = logger_pass.get("logger").getChild("find_valid_hsm")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    logger.debug('Processing HSMs to find valid representations |')

    # Map season to device

    if device == 'ac':
        season = 'summer'

    else:
        season = 'winter'

    allowed_buffer_months = static_params.get('ineff').get('hsm_allowed_buffer_months')

    hsm_dict = input_inefficiency_object.get('hsm_information')

    col_map = hsm_dict.get('col_map')
    hsm_basic = hsm_dict.get('basic')

    previous_year = None
    two_year_old = None

    if hsm_basic is None:
        valid_hsm_time = {'previous_year': previous_year, 'two_year_old': two_year_old}
        input_inefficiency_object[device]['valid_hsm_time'] = valid_hsm_time
        return input_inefficiency_object

    if hsm_basic.shape[0] == 0:
        valid_hsm_time = {'previous_year': previous_year, 'two_year_old': two_year_old}
        input_inefficiency_object[device]['valid_hsm_time'] = valid_hsm_time
        return input_inefficiency_object

    time_col = col_map.get('hsm_time')
    full_season_col = col_map.get('full_{}'.format(season))
    enough_season_col = col_map.get('enough_{}'.format(season))

    current_timestamp = input_inefficiency_object.get('raw_input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX]
    buffer_time = Cgbdisagg.SEC_IN_DAY * Cgbdisagg.DAYS_IN_MONTH * allowed_buffer_months
    epochs_in_a_year = Cgbdisagg.SEC_IN_DAY * Cgbdisagg.DAYS_IN_YEAR

    # Prepare min and time times for HSM filtering
    min_hsm_time = current_timestamp - epochs_in_a_year - buffer_time
    max_hsm_time = current_timestamp - epochs_in_a_year + buffer_time

    valid_time_hsm_idx = (hsm_basic[:, time_col] >= min_hsm_time) & (hsm_basic[:, time_col] <= max_hsm_time)
    full_season_idx = (hsm_basic[:, full_season_col] == 1)
    enough_season_idx = (hsm_basic[:, enough_season_col] == 1)

    full_season_check = full_season_idx & valid_time_hsm_idx
    enough_season_check = enough_season_idx & valid_time_hsm_idx

    if hsm_basic[full_season_check, :].shape[0] == 0:
        logger.debug('No HSM had full season last year, moving with decent sized {} |'.format(season))

        if hsm_basic[enough_season_check, :].shape[0] == 0:
            logger.debug('No HSM had decent sized {}, skipping |'.format(season))
        else:
            valid_hsm_list = hsm_basic[enough_season_check, time_col]
            previous_year = np.nanmax(valid_hsm_list)
    else:
        valid_hsm_list = hsm_basic[full_season_check, time_col]
        previous_year = np.nanmax(valid_hsm_list)

    # Prepare min and time times for HSM filtering

    epochs_in_two_year = Cgbdisagg.SEC_IN_DAY * Cgbdisagg.DAYS_IN_YEAR * 2
    min_hsm_time = current_timestamp - epochs_in_two_year - buffer_time
    max_hsm_time = current_timestamp - epochs_in_two_year + buffer_time

    valid_time_hsm_idx = (hsm_basic[:, time_col] >= min_hsm_time) & (hsm_basic[:, time_col] <= max_hsm_time)
    full_season_idx = (hsm_basic[:, full_season_col] == 1)
    enough_season_idx = (hsm_basic[:, enough_season_col] == 1)

    full_season_check = full_season_idx & valid_time_hsm_idx
    enough_season_check = enough_season_idx & valid_time_hsm_idx

    if hsm_basic[full_season_check, :].shape[0] == 0:
        logger.debug('No HSM had full {} 2 years ago, moving with decent sized {} |'.format(season, season))

        if hsm_basic[enough_season_check, :].shape[0] == 0:
            logger.debug('No HSM 2 years ago had decent sized {}, skipping |'.format(season))
        else:
            valid_hsm_list = hsm_basic[enough_season_check, time_col]
            two_year_old = valid_hsm_list[-1]
    else:
        valid_hsm_list = hsm_basic[full_season_check, time_col]
        two_year_old = valid_hsm_list[-1]

    valid_hsm_time = {'previous_year': previous_year, 'two_year_old': two_year_old}
    input_inefficiency_object[device]['valid_hsm_time'] = valid_hsm_time

    return input_inefficiency_object
