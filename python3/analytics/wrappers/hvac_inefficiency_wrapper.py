"""
Author - Anand Kumar Singh
Date - 19th Feb 2021
Call the HVAC inefficiency module and get output
"""

# Import python packages

import copy
import logging
import datetime
import traceback
import numpy as np

# Import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff
from python3.analytics.hvac_inefficiency.functions.process_hsm import prepare_hsm
from python3.analytics.hvac_inefficiency.functions.process_hsm import find_valid_hsm
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params
from python3.analytics.hvac_inefficiency.functions.hvac_change import detect_hvac_change
from python3.analytics.hvac_inefficiency.functions.abrupt_jump_hvac_tou import get_tou_outliers
from python3.analytics.hvac_inefficiency.functions.hvac_degradation import detect_hvac_degradation
from python3.analytics.hvac_inefficiency.functions.hvac_interference import detect_hvac_interference
from python3.analytics.hvac_inefficiency.functions.behavior_change import detect_hvac_behavior_change
from python3.analytics.hvac_inefficiency.functions.data_preprocessing_functions import preprocess_data
from python3.analytics.hvac_inefficiency.functions.abrupt_jump_in_ao_hvac import abrupt_jump_in_ao_hvac
from python3.analytics.hvac_inefficiency.functions.peer_comparison import prepare_peer_comparison_vector
from python3.analytics.hvac_inefficiency.functions.cycling_based_inefficiency import cycling_based_ineff
from python3.analytics.hvac_inefficiency.functions.abrupt_jump_in_net_consumption import get_cons_outlier
from python3.analytics.hvac_inefficiency.plotting_functions.plot_relevant_plots import plot_relevant_plots
from python3.analytics.hvac_inefficiency.functions.abrupt_jump_in_hvac_amp import abrupt_change_in_hvac_amp
from python3.analytics.hvac_inefficiency.functions.hvac_sanity_checks import pre_inefficiency_sanity_checks
from python3.analytics.hvac_inefficiency.functions.abrupt_jump_in_net_consumption import get_ao_cons_outlier
from python3.analytics.hvac_inefficiency.functions.hvac_sanity_checks import post_inefficiency_sanity_checks
from python3.analytics.hvac_inefficiency.functions.abrupt_jump_in_hvac_hours import abrupt_change_in_hvac_hours

from python3.analytics.hvac_inefficiency.functions.init_ineff_attributes import initialize_ineff_attributes

from python3.utils.validate_output_schema import validate_appliance_profile_schema_for_billcycle


def get_saturation_attributes(inefficiency_data):

    """
    Function to return saturation related attributes

    Parameters:
        inefficiency_data   (dict)    : Dictionary containing inefficiency data of either ac or sh

    Returns:
        saturation_temp     (int)       : Saturation temperature or None
        saturation_ineff    (bool)     : Presence of saturation inefficiency
        saturation_frac     (float)     : percentage of hvac points in saturation
    """

    saturation_temp = inefficiency_data.get('cycling_debug_dictionary',{}).get('saturation_temp')
    if (type(saturation_temp) == str) | (saturation_temp == None):
        saturation_temp = None
    else:
        saturation_temp = int(saturation_temp)

    saturation_ineff = False
    if bool(saturation_temp):
        saturation_ineff = True

    saturation_frac = 0.0
    if bool(saturation_temp):
        saturation_frac = float(inefficiency_data.get('cycling_debug_dictionary').get('saturation_fraction'))

    return saturation_temp, saturation_ineff, saturation_frac


def get_pre_saturation_attributes(inefficiency_data):

    """
    Function to return pre saturation related attributes

    Parameters:
        inefficiency_data (dict)    : Dictionary containing inefficiency data of either ac or sh

    Returns:
        pre_saturation_temp (int)   : pre saturation temperature or None
        pre_saturation_ineff (bool) : Boolean of pre saturation
    """

    pre_saturation_temp = inefficiency_data.get('cycling_debug_dictionary',{}).get('pre_saturation_temperature')
    if (type(pre_saturation_temp) == str) | (pre_saturation_temp == None):
        pre_saturation_temp = None
    else:
        pre_saturation_temp = int(pre_saturation_temp)

    pre_saturation_ineff = False
    if bool(pre_saturation_temp):
        pre_saturation_ineff = True

    return pre_saturation_temp, pre_saturation_ineff


def get_abrupt_attributes(inefficiency_data):

    """
    Function to get inefficiency related to abrupt hvac usage

    Parameters:
        inefficiency_data               (dict)  : Dictionary containing hvac inefficiency related data from sub modules

    Returns:
        has_abrupt_amplitude            (bool)  : Boolean of has abrupt amplitude
        outlier_amplitude_severity      (list)  : Amplitude severity
        outlier_amplitude               (float) : Outlier amplitude
        abrupt_hvac_hours               (list)  : Abrupt days
        abrupt_tou_starts               (list)  : Abrupt start
        abrupt_tou_ends                 (list)  : Abrupt end
        has_abrupt_ao                   (bool)  : Boolean of has abrupt ao
        outlier_ao_amplitude_severity   (list)  : AO amplitude severity

    """

    static_params = hvac_static_params()

    has_abrupt_hvac_hours = len(inefficiency_data.get('abrupt_hvac_hours', {}).get('final_outlier_days', [])) > 0
    has_abrupt_amplitude = len(inefficiency_data.get('abrupt_amplitude', {}).get('final_outlier_days', [])) > 0
    has_abrupt_ao = len(inefficiency_data.get('abrupt_ao_hvac', {}).get('final_outlier_days', [])) > 0

    outlier_amplitude = None
    outlier_amplitude_severity = None
    outlier_amplitude_validity = bool(inefficiency_data.get('abrupt_amplitude',{}).get('return_dictionary',{}))

    has_abrupt_amplitude = has_abrupt_amplitude & outlier_amplitude_validity

    if has_abrupt_amplitude & outlier_amplitude_validity:
        outlier_amplitude_bool_1 = bool(np.any(inefficiency_data.get('abrupt_amplitude',{}).get('return_dictionary',{}).get('high_outliers',{}).get('quad',{})[1]))
        outlier_amplitude_bool_2 = bool(np.any(inefficiency_data.get('abrupt_amplitude',{}).get('return_dictionary',{}).get('high_outliers',{}).get('ransac',{})[1]))
        outlier_amplitude_validity = outlier_amplitude_bool_1 | outlier_amplitude_bool_2
        has_abrupt_amplitude = outlier_amplitude_validity

        ransac_outlier_amplitudes = inefficiency_data.get('abrupt_amplitude',{}).get('return_dictionary',{}).get('high_outliers',{}).get('ransac',{})[1]
        ransac_scores = inefficiency_data['abrupt_amplitude']['return_dictionary']['high_outliers']['ransac'][2]

        quad_outlier_amplitudes = inefficiency_data.get('abrupt_amplitude',{}).get('return_dictionary',{}).get('high_outliers',{}).get('quad',{})[1]
        quad_scores = inefficiency_data.get('abrupt_amplitude',{}).get('return_dictionary',{}).get('high_outliers',{}).get('quad',{})[2]

        outlier_amplitude_severity = [float(np.nanmedian(np.r_[ransac_scores, quad_scores]))]
        outlier_amplitude = float(np.nanmean(np.r_[ransac_outlier_amplitudes, quad_outlier_amplitudes]))

    outlier_ao_amplitudes = None
    outlier_ao_amplitude_severity = None
    outlier_ao_amplitude_validity = bool(inefficiency_data.get('abrupt_ao_hvac',{}).get('return_dictionary',{}))

    has_abrupt_ao = has_abrupt_ao & outlier_ao_amplitude_validity

    if has_abrupt_ao & (outlier_ao_amplitude_validity):

        outlier_ao_amplitude_bool_1 = bool(np.any(inefficiency_data.get('abrupt_ao_hvac',{}).get('return_dictionary',{}).get('high_outliers',{}).get('quad',{})[1]))
        outlier_ao_amplitude_bool_2 = bool(np.any(inefficiency_data.get('abrupt_ao_hvac',{}).get('return_dictionary',{}).get('high_outliers',{}).get('ransac',{})[1]))
        outlier_ao_amplitude_validity = outlier_ao_amplitude_bool_1 | outlier_ao_amplitude_bool_2
        has_abrupt_ao = outlier_ao_amplitude_validity

        ransac_outlier_ao_amplitudes = inefficiency_data.get('abrupt_ao_hvac',{}).get('return_dictionary',{}).get('high_outliers',{}).get('ransac',{})[1]
        ransac_ao_scores = inefficiency_data.get('abrupt_ao_hvac',{}).get('return_dictionary',{}).get('high_outliers',{}).get('ransac',{})[2]

        quad_outlier_ao_amplitudes = inefficiency_data.get('abrupt_ao_hvac',{}).get('return_dictionary',{}).get('high_outliers',{}).get('quad',{})[1]
        quad_ao_scores = inefficiency_data.get('abrupt_ao_hvac',{}).get('return_dictionary',{}).get('high_outliers',{}).get('quad',{})[2]

        outlier_ao_amplitude_severity = [float(np.nanmedian(np.r_[ransac_ao_scores, quad_ao_scores]))]
        outlier_ao_amplitudes = float(np.nanmean(np.r_[ransac_outlier_ao_amplitudes, quad_outlier_ao_amplitudes]))

    if bool(has_abrupt_amplitude) | bool(has_abrupt_ao):
        if bool(has_abrupt_amplitude) & bool(has_abrupt_ao):
            outlier_amplitude = [float(np.nanmedian([outlier_amplitude, outlier_ao_amplitudes]))]
        elif bool(has_abrupt_amplitude):
            outlier_amplitude = [outlier_amplitude]
        else:
            has_abrupt_amplitude = True
            outlier_amplitude_severity = outlier_ao_amplitude_severity
            outlier_amplitude = [outlier_ao_amplitudes]
    else:
        outlier_amplitude = None

    abrupt_hvac_hours = None
    abrupt_tou_starts = None
    abrupt_tou_ends = None
    if has_abrupt_hvac_hours:
        abrupt_hvac_hours = inefficiency_data.get('abrupt_hvac_hours',{}).get('final_outlier_days',{}).tolist()
        if len(abrupt_hvac_hours) > static_params.get('ineff',{}).get('max_abrupt_hour_len',{}):
            abrupt_hvac_hours = abrupt_hvac_hours[:10]

        abrupt_tou_starts = [abrupt_hvac_hours[0]]
        abrupt_tou_ends = [abrupt_hvac_hours[-1]]

    return (has_abrupt_amplitude, outlier_amplitude_severity, outlier_amplitude, abrupt_hvac_hours, abrupt_tou_starts,
            abrupt_tou_ends, has_abrupt_ao, outlier_ao_amplitude_severity)


def get_short_cycling_attributes(inefficiency_data, analytics_output_object):

    """
    Function to return short cycling related attributes

    Parameters:

        inefficiency_data       (dict)  : Dictionary containing inefficiency data
        analytics_output_object (dict)  : Dictionary containing output payload of analytics modules

    Returns:
        has_short_cycling   (bool) : Boolean indicating presence of short cycling
        short_cycling_start (list) : Short cycling start time
        short_cycling_end   (list) : Short cycling end time
    """

    short_cycling_arr = inefficiency_data.get('cycling_debug_dictionary', {}).get('short_cycling', [0])
    has_short_cycling = bool(np.nansum(short_cycling_arr) > 0)

    epoch_times = analytics_output_object['epoch_estimate'][:, 0]
    short_cycling_times = epoch_times[short_cycling_arr > 0].tolist()
    short_cycling_start = None
    short_cycling_end = None
    if has_short_cycling:
        short_cycling_start = [short_cycling_times[0]]
        short_cycling_end = [short_cycling_times[-1]]

    return has_short_cycling, short_cycling_start, short_cycling_end


def get_behavior_change_attributes(inefficiency_data):

    """
    Function to return behavior change related attributes

    Parameters:

        inefficiency_data (dict)    : Dictionary containing hvac inefficiency data

    Returns:
        has_behavior_change (bool)  : Boolean indicating change in behavior detected
        change_dir          (str)   : Indicates change direction to high or low consumption
        divergence_score    (list)  : Indicates severity of change
    """

    change_high = bool(np.any(inefficiency_data.get('behavior_change',{}).get('change_date_high',{})))
    change_low = bool(np.any(inefficiency_data.get('behavior_change',{}).get('change_date_low',{})))
    divergence_arr = inefficiency_data.get('behavior_change',{}).get('divergence_array',{})

    has_behavior_change = None
    change_dir = None
    divergence_score = None
    if change_high | change_low:

        divergence_days = divergence_arr[:, 0]
        has_behavior_change = True

        if change_high and not change_low:
            change_dir = 'high'

            change_days = inefficiency_data.get('behavior_change').get('change_date_high')
            change_idx = np.isin(divergence_days, change_days)

        elif change_low and not change_high:
            change_dir = 'low'

            change_days = inefficiency_data.get('behavior_change',{}).get('change_date_low',{})
            change_idx = np.isin(divergence_days, change_days)

        else:
            change_dir = 'high-low'

            change_days_high = inefficiency_data.get('behavior_change').get('change_date_high')
            change_days_low = inefficiency_data.get('behavior_change').get('change_date_low')

            change_high_idx = np.isin(divergence_days, change_days_high)
            change_low_idx = np.isin(divergence_days, change_days_low)

            change_idx = change_high_idx | change_low_idx

        divergence_score = np.abs(divergence_arr[change_idx, 1])/10
        divergence_score = np.fmin(divergence_score,[1])
        divergence_score = [float(np.nanmedian(divergence_score))]

    return has_behavior_change, change_dir, divergence_score


def get_appliance_change_attributes(inefficiency_data):

    """
    Function to return appliance change related attributes

    Parameters:

        inefficiency_data (dict)    : Dictionary containing hvac inefficiency data

    Returns:
        has_app_change      (bool)  : Boolean indicating change in appliance detected
        app_change_dir      (str)   : Indicates change direction to high or low consumption
        app_change_prob     (float) : Indicates probability of appliance change
    """

    has_app_change = bool(inefficiency_data.get('app_change', {}).get('app_change', 0))
    app_change_dir = None
    app_change_prob = None

    if has_app_change:
        previous_fcc = inefficiency_data.get('app_change', {}).get('previous_fcc', 0)
        current_fcc = inefficiency_data.get('app_change', {}).get('current_fcc', 0)
        if previous_fcc < current_fcc:
            app_change_dir = 'high'
        else:
            app_change_dir = 'low'

        app_change_prob = float(inefficiency_data.get('app_change', {}).get('probability', 0))

    return has_app_change, app_change_dir, app_change_prob


def get_degradation_attributes(inefficiency_data):

    """
    Function to return the hvac degradation related attributes

    Parameters:

        inefficiency_data (dict)    : Dictionary containing hvac inefficiency data

    Returns:
        has_degradation     (bool)  : Boolean indicating presence of degradation
        degradation_conf    (float) : Degradation detection confidence
    """

    has_degradation = bool(inefficiency_data.get('app_degradation', {}).get('degradation', 0))
    degradation_conf = None
    if has_degradation:
        degradation_conf = float(inefficiency_data.get('app_degradation', {}).get('probability')[0])

    return has_degradation, degradation_conf


def populate_sh_run_attributes(sh_inefficiency_data, analytics_output_object, sh_attributes):
    """
    Function to populate sh related inefficiency attributes for a run

    Parameters:
        sh_inefficiency_data        (dict)  : Dictionary containing sh ineff related data
        output_inefficiency_object  (dict)  : Dictionary containing output os inefficiency sub modules
        analytics_output_object     (dict)  : Dictionary containing analytics output payload
        sh_attributes               (dict)  : Dictionary containing initialized sh attributes

    Returns:
        sh_attributes               (dict)  : Dictionary containing filled sh attributes
    """

    # saturation attributes

    saturation_temp, saturation_ineff, saturation_frac = get_saturation_attributes(sh_inefficiency_data)

    # pre saturation attributes

    pre_saturation_temp, pre_saturation_ineff = get_pre_saturation_attributes(sh_inefficiency_data)

    # abrupt consumption attributes

    abrupt_attributes = get_abrupt_attributes(sh_inefficiency_data)

    # short cycling

    has_short_cycling, short_cycling_start, short_cycling_end = get_short_cycling_attributes(sh_inefficiency_data,
                                                                                             analytics_output_object)

    # behavior change

    has_behavior_change, change_dir, divergence_score = get_behavior_change_attributes(sh_inefficiency_data)

    # appliance change

    has_app_change, app_change_dir, app_change_prob = get_appliance_change_attributes(sh_inefficiency_data)

    # degradation

    has_degradation, degradation_conf = get_degradation_attributes(sh_inefficiency_data)

    sh_attributes["hasSaturationIneff"] = saturation_ineff
    sh_attributes["saturationTemperature"] = saturation_temp
    sh_attributes["fractionOfSaturatedPts"] = saturation_frac
    sh_attributes["hasPreSaturation"] = pre_saturation_ineff
    sh_attributes["preSaturationTemperature"] = pre_saturation_temp

    sh_attributes["hasAbruptConsumption"] = abrupt_attributes[0]
    sh_attributes["abruptConsumptionSeverity"] = abrupt_attributes[1]
    sh_attributes["abruptAmplitude"] = abrupt_attributes[2]
    sh_attributes["abruptHours"] = abrupt_attributes[3]
    sh_attributes["abruptTouStarts"] = abrupt_attributes[4]
    sh_attributes["abruptTouEnds"] = abrupt_attributes[5]

    sh_attributes["hasAbruptAoHvac"] = abrupt_attributes[6]
    sh_attributes["abruptAoHvacSeverity"] = abrupt_attributes[7]

    sh_attributes["hasShortCycling"] = has_short_cycling
    sh_attributes["shortCyclingStarts"] = short_cycling_start
    sh_attributes["shortCyclingEnds"] = short_cycling_end

    sh_attributes["hasBehaviorChange"] = has_behavior_change
    sh_attributes["behaviorChangeType"] = change_dir
    sh_attributes["behaviorChangeSeverity"] = divergence_score

    sh_attributes["hasApplianceChange"] = has_app_change
    sh_attributes["applianceChangeDirection"] = app_change_dir
    sh_attributes["applianceChangeConfidence"] = app_change_prob

    sh_attributes["hasApplianceDegradation"] = has_degradation
    sh_attributes["applianceDegradeConfidence"] = degradation_conf

    sh_attributes["applianceOfficeTimeOverlap"] = None
    sh_attributes["officeTimeOverlapSeverity"] = None

    sh_attributes["hasSmartThermostat"] = None
    sh_attributes["hasSmartThermostatConf"] = None

    sh_attributes["applianceUsageShape"] = None
    sh_attributes["applianceUsageHours"] = None

    return sh_attributes


def populate_ac_run_attributes(ac_inefficiency_data, analytics_output_object, ac_attributes):

    """
    Function to populate ac related inefficiency attributes for a run

    Parameters:
        ac_inefficiency_data        (dict)  : Dictionary containing ac ineff related data
        output_inefficiency_object  (dict)  : Dictionary containing output os inefficiency sub modules
        analytics_output_object     (dict)  : Dictionary containing analytics output payload
        ac_attributes               (dict)  : Dictionary containing initialized ac attributes

    Returns:
        ac_attributes               (dict)  : Dictionary containing filled ac attributes
    """

    # saturation attributes

    saturation_temp, saturation_ineff, saturation_frac = get_saturation_attributes(ac_inefficiency_data)

    # pre saturation attributes

    pre_saturation_temp, pre_saturation_ineff = get_pre_saturation_attributes(ac_inefficiency_data)

    # abrupt consumption attributes

    abrupt_attributes = get_abrupt_attributes(ac_inefficiency_data)

    # short cycling

    has_short_cycling, short_cycling_start, short_cycling_end = get_short_cycling_attributes(ac_inefficiency_data,
                                                                                             analytics_output_object)

    # behavior change

    has_behavior_change, change_dir, divergence_score = get_behavior_change_attributes(ac_inefficiency_data)

    # appliance change

    has_app_change, app_change_dir, app_change_prob = get_appliance_change_attributes(ac_inefficiency_data)

    # degradation

    has_degradation, degradation_conf = get_degradation_attributes(ac_inefficiency_data)

    # Office time overlap present only for ac device and not sh

    has_overlap = bool(ac_inefficiency_data.get('interfering_hvac', {}).get('overlap_score', 0))
    overlap_conf = None
    if has_overlap:
        overlap_conf = [float(ac_inefficiency_data.get('interfering_hvac', {}).get('overlap_score', 0))]

    ac_attributes["hasSaturationIneff"] = saturation_ineff
    ac_attributes["saturationTemperature"] = saturation_temp
    ac_attributes["fractionOfSaturatedPts"] = saturation_frac
    ac_attributes["hasPreSaturation"] = pre_saturation_ineff
    ac_attributes["preSaturationTemperature"] = pre_saturation_temp

    ac_attributes["hasAbruptConsumption"] = abrupt_attributes[0]
    ac_attributes["abruptConsumptionSeverity"] = abrupt_attributes[1]
    ac_attributes["abruptAmplitude"] = abrupt_attributes[2]
    ac_attributes["abruptHours"] = abrupt_attributes[3]
    ac_attributes["abruptTouStarts"] = abrupt_attributes[4]
    ac_attributes["abruptTouEnds"] = abrupt_attributes[5]

    ac_attributes["hasAbruptAoHvac"] = abrupt_attributes[6]
    ac_attributes["abruptAoHvacSeverity"] = abrupt_attributes[7]

    ac_attributes["hasShortCycling"] = has_short_cycling
    ac_attributes["shortCyclingStarts"] = short_cycling_start
    ac_attributes["shortCyclingEnds"] = short_cycling_end

    ac_attributes["hasBehaviorChange"] = has_behavior_change
    ac_attributes["behaviorChangeType"] = change_dir
    ac_attributes["behaviorChangeSeverity"] = divergence_score

    ac_attributes["hasApplianceChange"] = has_app_change
    ac_attributes["applianceChangeDirection"] = app_change_dir
    ac_attributes["applianceChangeConfidence"] = app_change_prob

    ac_attributes["hasApplianceDegradation"] = has_degradation
    ac_attributes["applianceDegradeConfidence"] = degradation_conf

    ac_attributes["applianceOfficeTimeOverlap"] = has_overlap
    ac_attributes["officeTimeOverlapSeverity"] = overlap_conf

    ac_attributes["hasSmartThermostat"] = None
    ac_attributes["hasSmartThermostatConf"] = None

    ac_attributes["applianceUsageShape"] = None
    ac_attributes["applianceUsageHours"] = None

    return ac_attributes


def fill_run_attributes(analytics_output_object, output_inefficiency_object):

    """
    Function to get run attributes to be filled in hvac appliance profile for inefficiency

    Parameters:
        analytics_output_object     (dict): Dictionary containing analytics output payload
        output_inefficiency_object  (dict): Dictionary containing inefficiency output payload

    Returns:
        run_attributes              (dict): Dictionary containing hvac inefficiency attributes
    """

    # accessing available inefficiency attributes
    ac_inefficiency_data = output_inefficiency_object['ac']
    sh_inefficiency_data = output_inefficiency_object['sh']

    # checking which device inefficiency exists
    ac_inefficiency_exists = ac_inefficiency_data != {}
    sh_inefficiency_exists = sh_inefficiency_data != {}

    # initializing ac and sh inefficiency attributes
    sh_attributes, ac_attributes = initialize_ineff_attributes()

    # getting inefficiency attributes to be populated

    if sh_inefficiency_exists:

        sh_attributes = populate_sh_run_attributes(sh_inefficiency_data, analytics_output_object, sh_attributes)

    if ac_inefficiency_exists:

        ac_attributes = populate_ac_run_attributes(ac_inefficiency_data, analytics_output_object, ac_attributes)

    run_attributes = {
        'sh': {
            'ineff_exists':sh_inefficiency_exists,
            'attributes':sh_attributes
        },
        'ac': {
            'ineff_exists':ac_inefficiency_exists,
            'attributes':ac_attributes
        }
    }

    return run_attributes


def fill_appliance_profile(analytics_input_object, analytics_output_object, output_inefficiency_object):

    """
    Function to fill appliance profile for hvac inefficiency attributes

    Parameters:
        analytics_input_object      (dict): Dictionary containing analytics input payload
        analytics_output_object     (dict): Dictionary containing analytics output payload
        output_inefficiency_object  (dict): Dictionary containing inefficiency output payload

    Returns:
        analytics_output_object     (dict): Dictionary containing updated analytics output payload
    """

    # initializing logger
    logger_inefficiency_base = analytics_input_object.get("logger").getChild("fill_appliance_profile")
    logger_inefficiency_pass = {"logger": logger_inefficiency_base,
                                "logging_dict": analytics_input_object.get("logging_dict")}

    logger_inefficiency = logging.LoggerAdapter(logger_inefficiency_base, analytics_input_object.get("logging_dict"))

    out_bill_cycles = analytics_input_object.get('out_bill_cycles')

    # getting run attributes to be filled in hvac appliance profile for inefficiency
    run_attributes = fill_run_attributes(analytics_output_object, output_inefficiency_object)

    # accessing user profiles object
    user_profile_object = analytics_output_object['appliance_profile']

    for billcycle_start, billcycle_end in out_bill_cycles:
        base = user_profile_object[billcycle_start]['profileList'][0]

        try:
            heating_profile = copy.deepcopy(base['3'][0])
            run_attributes_sh = run_attributes.get('sh').get('attributes')

            heating_profile['attributes']["hasSaturationIneff"] = run_attributes_sh['hasSaturationIneff']
            heating_profile['attributes']["saturationTemperature"] = run_attributes_sh['saturationTemperature']
            heating_profile['attributes']["fractionOfSaturatedPts"] = run_attributes_sh['fractionOfSaturatedPts']
            heating_profile['attributes']["hasPreSaturation"] = run_attributes_sh['hasPreSaturation']
            heating_profile['attributes']["preSaturationTemperature"] = run_attributes_sh['preSaturationTemperature']

            heating_profile['attributes']["hasAbruptConsumption"] = run_attributes_sh['hasAbruptConsumption']
            heating_profile['attributes']["abruptConsumptionSeverity"] = run_attributes_sh['abruptConsumptionSeverity']
            heating_profile['attributes']["abruptAmplitude"] = run_attributes_sh['abruptAmplitude']
            heating_profile['attributes']["abruptHours"] = run_attributes_sh['abruptHours']
            heating_profile['attributes']["abruptTouStarts"] = run_attributes_sh['abruptTouStarts']
            heating_profile['attributes']["abruptTouEnds"] = run_attributes_sh['abruptTouEnds']

            heating_profile['attributes']["hasAbruptAoHvac"] = run_attributes_sh['hasAbruptAoHvac']
            heating_profile['attributes']["abruptAoHvacSeverity"] = run_attributes_sh['abruptAoHvacSeverity']

            heating_profile['attributes']["hasShortCycling"] = run_attributes_sh['hasShortCycling']
            heating_profile['attributes']["shortCyclingStarts"] = run_attributes_sh['shortCyclingStarts']
            heating_profile['attributes']["shortCyclingEnds"] = run_attributes_sh['shortCyclingEnds']

            heating_profile['attributes']["hasBehaviorChange"] = run_attributes_sh['hasBehaviorChange']
            heating_profile['attributes']["behaviorChangeType"] = run_attributes_sh['behaviorChangeType']
            heating_profile['attributes']["behaviorChangeSeverity"] = run_attributes_sh['behaviorChangeSeverity']

            heating_profile['attributes']["hasApplianceChange"] = run_attributes_sh['hasApplianceChange']
            heating_profile['attributes']["applianceChangeDirection"] = run_attributes_sh['applianceChangeDirection']
            heating_profile['attributes']["applianceChangeConfidence"] = run_attributes_sh['applianceChangeConfidence']

            heating_profile['attributes']["hasApplianceDegradation"] = run_attributes_sh['hasApplianceDegradation']
            heating_profile['attributes']["applianceDegradeConfidence"] = run_attributes_sh['applianceDegradeConfidence']

            heating_profile['attributes']["applianceOfficeTimeOverlap"] = run_attributes_sh['applianceOfficeTimeOverlap']
            heating_profile['attributes']["officeTimeOverlapSeverity"] = run_attributes_sh['officeTimeOverlapSeverity']

            heating_profile['attributes']["hasSmartThermostat"] = run_attributes_sh['hasSmartThermostat']
            heating_profile['attributes']["hasSmartThermostatConf"] = run_attributes_sh['hasSmartThermostatConf']

            heating_profile['attributes']["applianceUsageShape"] = run_attributes_sh['applianceUsageShape']
            heating_profile['attributes']["applianceUsageHours"] = run_attributes_sh['applianceUsageHours']

            user_profile_object[billcycle_start]['profileList'][0]['3'][0] = heating_profile

            logger_inefficiency.info(' Done heating inefficiency profile for {} bill cycle |'.format(billcycle_start))

        except (ValueError, IndexError, KeyError):

            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger_inefficiency.error('Heating inefficiency Profile Fill for billcycle %d failed | %s' %
                                      (int(billcycle_start), error_str))
            logger_inefficiency.info('Heating inefficiency Profile Empty for billcycle | %d ', int(billcycle_start))

        try:
            cooling_profile = copy.deepcopy(base['4'][0])
            run_attributes_ac = run_attributes.get('ac').get('attributes')

            cooling_profile['attributes']["hasSaturationIneff"] = run_attributes_ac['hasSaturationIneff']
            cooling_profile['attributes']["saturationTemperature"] = run_attributes_ac['saturationTemperature']
            cooling_profile['attributes']["fractionOfSaturatedPts"] = run_attributes_ac['fractionOfSaturatedPts']
            cooling_profile['attributes']["hasPreSaturation"] = run_attributes_ac['hasPreSaturation']
            cooling_profile['attributes']["preSaturationTemperature"] = run_attributes_ac['preSaturationTemperature']

            cooling_profile['attributes']["hasAbruptConsumption"] = run_attributes_ac['hasAbruptConsumption']
            cooling_profile['attributes']["abruptConsumptionSeverity"] = run_attributes_ac['abruptConsumptionSeverity']
            cooling_profile['attributes']["abruptAmplitude"] = run_attributes_ac['abruptAmplitude']
            cooling_profile['attributes']["abruptHours"] = run_attributes_ac['abruptHours']
            cooling_profile['attributes']["abruptTouStarts"] = run_attributes_ac['abruptTouStarts']
            cooling_profile['attributes']["abruptTouEnds"] = run_attributes_ac['abruptTouEnds']

            cooling_profile['attributes']["hasAbruptAoHvac"] = run_attributes_ac['hasAbruptAoHvac']
            cooling_profile['attributes']["abruptAoHvacSeverity"] = run_attributes_ac['abruptAoHvacSeverity']

            cooling_profile['attributes']["hasShortCycling"] = run_attributes_ac['hasShortCycling']
            cooling_profile['attributes']["shortCyclingStarts"] = run_attributes_ac['shortCyclingStarts']
            cooling_profile['attributes']["shortCyclingEnds"] = run_attributes_ac['shortCyclingEnds']

            cooling_profile['attributes']["hasBehaviorChange"] = run_attributes_ac['hasBehaviorChange']
            cooling_profile['attributes']["behaviorChangeType"] = run_attributes_ac['behaviorChangeType']
            cooling_profile['attributes']["behaviorChangeSeverity"] = run_attributes_ac['behaviorChangeSeverity']

            cooling_profile['attributes']["hasApplianceChange"] = run_attributes_ac['hasApplianceChange']
            cooling_profile['attributes']["applianceChangeDirection"] = run_attributes_ac['applianceChangeDirection']
            cooling_profile['attributes']["applianceChangeConfidence"] = run_attributes_ac['applianceChangeConfidence']

            cooling_profile['attributes']["hasApplianceDegradation"] = run_attributes_ac['hasApplianceDegradation']
            cooling_profile['attributes']["applianceDegradeConfidence"] = run_attributes_ac[
                'applianceDegradeConfidence']

            cooling_profile['attributes']["applianceOfficeTimeOverlap"] = run_attributes_ac[
                'applianceOfficeTimeOverlap']
            cooling_profile['attributes']["officeTimeOverlapSeverity"] = run_attributes_ac['officeTimeOverlapSeverity']

            cooling_profile['attributes']["hasSmartThermostat"] = run_attributes_ac['hasSmartThermostat']
            cooling_profile['attributes']["hasSmartThermostatConf"] = run_attributes_ac['hasSmartThermostatConf']

            cooling_profile['attributes']["applianceUsageShape"] = run_attributes_ac['applianceUsageShape']
            cooling_profile['attributes']["applianceUsageHours"] = run_attributes_ac['applianceUsageHours']

            user_profile_object[billcycle_start]['profileList'][0]['4'][0] = cooling_profile

            logger_inefficiency.info(' Done cooling inefficiency profile for {} bill cycle |'.format(billcycle_start))

        except:

            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger_inefficiency.error('Cooling inefficiency Profile Fill for billcycle %d failed | %s' %
                                      (int(billcycle_start), error_str))
            logger_inefficiency.info('Cooling inefficiency Profile Empty for billcycle | %d ', int(billcycle_start))

        analytics_output_object['appliance_profile'] = user_profile_object

        validate_appliance_profile_schema_for_billcycle(analytics_output_object, billcycle_start, logger_inefficiency_pass)

    return analytics_output_object


def fill_and_validate_appliance_profiile(analytics_input_object, analytics_output_object, output_inefficiency_object):

    """
    Function to fill the appliance profile in non mtd modes

    Parameters:
        analytics_input_object      (dict): Dictionary containing analytics input payload
        analytics_output_object     (dict): Dictionary containing analytics output payload
        output_inefficiency_object  (dict): Dictionary containing inefficiency output payload

    Returns:
        analytics_output_object     (dict): Dictionary containing updated analytics input payload
    """

    disagg_mode = analytics_input_object.get('config').get('disagg_mode')

    if not disagg_mode == 'mtd':

        analytics_output_object = fill_appliance_profile(analytics_input_object, analytics_output_object,
                                                         output_inefficiency_object)

    return analytics_output_object


def hvac_inefficiency_wrapper(analytics_input_object, analytics_output_object):

    """
    Parameters:
        analytics_input_object (dict)              : Dictionary containing all inputs
        analytics_output_object(dict)              : Dictionary containing all outputs

    Returns:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    static_params = hvac_static_params()

    logger_inefficiency_base = analytics_input_object.get("logger").getChild("hvac_inefficiency_wrapper")
    logger_inefficiency_pass = {"logger": logger_inefficiency_base,
                                "logging_dict": analytics_input_object.get("logging_dict")}

    logger_inefficiency = logging.LoggerAdapter(logger_inefficiency_base, analytics_input_object.get("logging_dict"))

    master_start_time = datetime.datetime.now()

    # hvac inefficiency module won't run for mtd mode
    run_type = analytics_input_object['config']['disagg_mode']
    if run_type == 'mtd':
        time_taken = get_time_diff(master_start_time, datetime.datetime.now())
        logger_inefficiency.debug('>>Time taken for HVAC Inefficiency in mtd mode | {} | {}'.format('all combined', time_taken))

        return analytics_input_object, analytics_output_object

    hvac_epoch_estimate = analytics_output_object.get('hvac_debug', {}).get('write', {}).get('epoch_ao_hvac_true', {})

    if len(hvac_epoch_estimate) == 0:

        logger_inefficiency.warning('Epoch level HVAC not found')

        return analytics_input_object, analytics_output_object

    input_inefficiency_object, output_inefficiency_object =\
        preprocess_data(analytics_input_object, analytics_output_object, logger_inefficiency_pass)

    column_idx_dict = analytics_output_object.get('hvac_debug').get('write').get('epoch_idx_dentify')

    ac_demand_column = column_idx_dict.get('ac_od')
    sh_demand_column = column_idx_dict.get('sh_od')
    ac_demand = copy.deepcopy(hvac_epoch_estimate[:, ac_demand_column])
    sh_demand = copy.deepcopy(hvac_epoch_estimate[:, sh_demand_column])

    device_list = ['ac', 'sh']

    for device in device_list:

        try:

            if device == 'ac':
                input_inefficiency_object['hvac_consumption'] = copy.deepcopy(ac_demand.reshape(-1, 1))

            elif device == 'sh':
                input_inefficiency_object['hvac_consumption'] = copy.deepcopy(sh_demand.reshape(-1, 1))

            temp_time = datetime.datetime.now()
            time_taken = get_time_diff(temp_time, datetime.datetime.now())

            run_inefficiency =\
                pre_inefficiency_sanity_checks(input_inefficiency_object, logger_inefficiency_pass, device)

            logger_inefficiency.info('Time taken for Pre processing checks | {} | {}'.format(device, time_taken))

            if not run_inefficiency:
                continue

            temp_time = datetime.datetime.now()
            input_inefficiency_object, output_inefficiency_object = \
                cycling_based_ineff(input_inefficiency_object, output_inefficiency_object,
                                    logger_inefficiency_pass, device)

            time_taken = get_time_diff(temp_time, datetime.datetime.now())
            logger_inefficiency.info('Time taken for cycling | {} | {}'.format(device, time_taken))

            temp_time = datetime.datetime.now()
            input_inefficiency_object, output_inefficiency_object = \
                abrupt_change_in_hvac_hours(input_inefficiency_object, output_inefficiency_object,
                                            logger_inefficiency_pass, device)
            time_taken = get_time_diff(temp_time, datetime.datetime.now())
            logger_inefficiency.info('Time taken for abrupt_change_in_hvac_hours | {} | {}'.format(device,
                                                                                                   time_taken))

            temp_time = datetime.datetime.now()
            input_inefficiency_object, output_inefficiency_object =\
                abrupt_change_in_hvac_amp(input_inefficiency_object, output_inefficiency_object,
                                          logger_inefficiency_pass, device)
            time_taken = get_time_diff(temp_time, datetime.datetime.now())
            logger_inefficiency.info('Time taken for abrupt_change_in_hvac_amp | {} | {}'.format(device,
                                                                                                 time_taken))

            temp_time = datetime.datetime.now()
            input_inefficiency_object, output_inefficiency_object =\
                abrupt_jump_in_ao_hvac(input_inefficiency_object, output_inefficiency_object,
                                       logger_inefficiency_pass, device)
            time_taken = get_time_diff(temp_time, datetime.datetime.now())
            logger_inefficiency.info('Time taken for abrupt_change_in_hvac_ao | {} | {}'.format(device, time_taken))

            temp_time = datetime.datetime.now()
            input_inefficiency_object, output_inefficiency_object =\
                get_ao_cons_outlier(input_inefficiency_object, output_inefficiency_object, device)
            time_taken = get_time_diff(temp_time, datetime.datetime.now())
            logger_inefficiency.info('Time taken for consumption ao | {} | {}'.format(device, time_taken))

            temp_time = datetime.datetime.now()
            input_inefficiency_object, output_inefficiency_object = \
                get_cons_outlier(input_inefficiency_object, output_inefficiency_object, device)
            time_taken = get_time_diff(temp_time, datetime.datetime.now())
            logger_inefficiency.info('Time taken for abrupt_change_in_consumption | {} | {}'.format(device,
                                                                                                    time_taken))

            temp_time = datetime.datetime.now()
            input_inefficiency_object = find_valid_hsm(input_inefficiency_object, device, logger_inefficiency_pass)
            time_taken = get_time_diff(temp_time, datetime.datetime.now())
            logger_inefficiency.debug('Time taken for Preparing HSMs | {} | {}'.format(device, time_taken))

            temp_time = datetime.datetime.now()
            input_inefficiency_object, output_inefficiency_object = \
                detect_hvac_degradation(input_inefficiency_object, output_inefficiency_object,
                                        logger_inefficiency_pass, device)

            time_taken = get_time_diff(temp_time, datetime.datetime.now())
            logger_inefficiency.debug('Time taken for Degradation detection | {} | {}'.format(device, time_taken))

            temp_time = datetime.datetime.now()
            input_inefficiency_object, output_inefficiency_object =\
                detect_hvac_change(input_inefficiency_object, output_inefficiency_object,
                                   logger_inefficiency_pass, device)

            time_taken = get_time_diff(temp_time, datetime.datetime.now())
            logger_inefficiency.debug('Time taken for App Change detection | {} | {}'.format(device, time_taken))

            temp_time = datetime.datetime.now()
            input_inefficiency_object, output_inefficiency_object =\
                detect_hvac_behavior_change(input_inefficiency_object, output_inefficiency_object,
                                            logger_inefficiency_pass, device)
            time_taken = get_time_diff(temp_time, datetime.datetime.now())
            logger_inefficiency.debug('Time taken for Behavior Change detection | {} | {}'.format(device, time_taken))

            temp_time = datetime.datetime.now()
            input_inefficiency_object, output_inefficiency_object =\
                detect_hvac_interference(input_inefficiency_object, output_inefficiency_object,
                                         logger_inefficiency_pass, device)
            time_taken = get_time_diff(temp_time, datetime.datetime.now())
            logger_inefficiency.debug('Time taken for HVAC interference | {} | {}'.format(device, time_taken))

            temp_time = datetime.datetime.now()
            input_inefficiency_object, output_inefficiency_object = \
                prepare_peer_comparison_vector(input_inefficiency_object, output_inefficiency_object,
                                               logger_inefficiency_pass, device)
            time_taken = get_time_diff(temp_time, datetime.datetime.now())
            logger_inefficiency.debug('Time taken for Peer Comparison detection | {} | {}'.format(device, time_taken))

            temp_time = datetime.datetime.now()
            input_inefficiency_object, output_inefficiency_object =\
                post_inefficiency_sanity_checks(input_inefficiency_object, output_inefficiency_object,
                                                logger_inefficiency_pass, device)
            time_taken = get_time_diff(temp_time, datetime.datetime.now())
            logger_inefficiency.info('Time taken for Post processing checks | {} | {}'.format(device, time_taken))
        except:
            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger_inefficiency.error("{} inefficiency failed | %s".format(device), error_str)

    try:
        temp_time = datetime.datetime.now()
        input_inefficiency_object, output_inefficiency_object =\
            get_tou_outliers(input_inefficiency_object, output_inefficiency_object)
        time_taken = get_time_diff(temp_time, datetime.datetime.now())
        logger_inefficiency.info('Time taken for TOU outlier detection | {} | {}'.format("both combined", time_taken))

        time_taken = get_time_diff(master_start_time, datetime.datetime.now())
        logger_inefficiency.debug('Time taken for Combined Timed | {} | {}'.format('all combined', time_taken))

    except:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger_inefficiency.error("Common inefficiency failed | %s", error_str)

    # Creating HSM

    hvac_inefficiency_hsm = prepare_hsm(analytics_input_object, input_inefficiency_object, output_inefficiency_object)
    analytics_output_object['created_hsm_analytics']['hvac_ineff'] = hvac_inefficiency_hsm

    analytics_output_object = fill_and_validate_appliance_profiile(analytics_input_object, analytics_output_object,
                                                                   output_inefficiency_object)

    time_taken = get_time_diff(master_start_time, datetime.datetime.now())
    logger_inefficiency.debug('>>Time taken for HVAC Inefficiency | {} | {}'.format('all combined', time_taken))

    if static_params.get('ineff').get('enable_plotting'):
        try:
            plot_relevant_plots(input_inefficiency_object, output_inefficiency_object)
        except:
            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger_inefficiency.error("Plotting function failed | %s", error_str)

    return analytics_input_object, analytics_output_object
