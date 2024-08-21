"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to fill smb profile
"""

# Import python packages

import logging
import numpy as np

# Import functions from within the project

from python3.utils.validate_output_schema import validate_appliance_profile_schema_for_billcycle
from python3.utils.validate_output_schema import validate_lifestyle_profile_schema_for_billcycle


def get_heating_consumption(month_consumptions, month_idx_identify):

    """
    Function to get heating consumptions

    Parameters:

        month_consumptions (np.ndarray) : Array of month heating consumptions
        month_idx_identify (dict)       : Dictionary containing key identifiers of heating consumptions

    Returns:
        total_ao           (float)      : Total ao consumption
        total_heating      (float)      : Total sh consumption
        open_heating       (float)      : Total sh consumption in open hours
        close_heating      (float)      : Total sh consumption in close hours
    """

    if len(month_consumptions) > 0:
        total_ao = month_consumptions[:, month_idx_identify.get('ao')][0]
        total_heating = month_consumptions[:, month_idx_identify.get('sh')][0]
        open_heating = month_consumptions[:, month_idx_identify.get('sh_open')][0]
        close_heating = month_consumptions[:, month_idx_identify.get('sh_close')][0]
    else:
        total_ao = 0
        total_heating = 0
        open_heating = 0
        close_heating = 0

    return total_ao, total_heating, open_heating, close_heating


def get_cooling_consumption(month_consumptions, month_idx_identify):

    """
    Function to get cooling consumptions

    Parameters:

        month_consumptions (np.ndarray) : Array of month cooling consumptions
        month_idx_identify (dict)       : Dictionary containing key identifiers of cooling consumptions

    Returns:
        total_cooling      (float)      : Total ac consumption
        open_cooling       (float)      : Total ac consumption in open hours
        close_cooling      (float)      : Total ac consumption in close hours
    """

    if len(month_consumptions) > 0:
        total_cooling = month_consumptions[:, month_idx_identify.get('ac')][0]
        open_cooling = month_consumptions[:, month_idx_identify.get('ac_open')][0]
        close_cooling = month_consumptions[:, month_idx_identify.get('ac_close')][0]
    else:
        total_cooling = 0
        open_cooling = 0
        close_cooling = 0

    return total_cooling, open_cooling, close_cooling


def get_op_consumption(month_consumptions, month_idx_identify):

    """
    Function to get op consumptions

    Parameters:

        month_consumptions (np.ndarray) : Array of month op consumptions
        month_idx_identify (dict)       : Dictionary containing key identifiers of op consumptions

    Returns:
        op_consumption     (float)      : Total op consumption in month
    """

    if len(month_consumptions) > 0:
        op_consumption = month_consumptions[:, month_idx_identify.get('op')][0]
    else:
        op_consumption = 0

    return op_consumption


def get_xao_consumption(month_consumptions, month_idx_identify):

    """
    Function to get x-ao consumptions

    Parameters:

        month_consumptions (np.ndarray) : Array of month x-ao consumptions
        month_idx_identify (dict)       : Dictionary containing key identifiers of x-ao consumptions

    Returns:
        x_ao_consumption   (float)      : Total x-ao consumption in month
    """

    if len(month_consumptions) > 0:
        x_ao_consumption = month_consumptions[:, month_idx_identify.get('x-ao')][0]
    else:
        x_ao_consumption = 0

    return x_ao_consumption


def fill_user_profile_smb(disagg_input_object, disagg_output_object, logger_base):
    """
        Function to populate smb user profile

        Parameters:
            disagg_input_object     (dict)              : Dictionary containing all inputs
            disagg_output_object    (dict)              : Dictionary containing all outputs
            logger_base             (logging object)    : Logs code progress

        Returns:
            disagg_output_object    (dict)              : Dictionary containing all outputs
        """

    # initializing logger object
    logger_local = logger_base.get("logger").getChild("hvac_user_profile")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac_profile = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # getting out billing-cycles
    out_bill_cycles = disagg_input_object.get('out_bill_cycles')

    month_idx_identify = disagg_output_object.get('hvac_debug').get('write').get('column_dentify')
    month_estimates = disagg_output_object.get('hvac_debug').get('write').get('month_ao_hvac_true')

    # Extracting, if appliances exist for a user
    has_ac = bool(np.nansum(month_estimates[:, month_idx_identify.get('ac')]) > 0)
    has_sh = bool(np.nansum(month_estimates[:, month_idx_identify.get('sh')]) > 0)
    has_ao = bool(np.nansum(month_estimates[:, month_idx_identify.get('ao')]) > 0)
    has_op = bool(np.nansum(month_estimates[:, month_idx_identify.get('op')]) > 0)
    has_x_ao = bool(np.nansum(month_estimates[:, month_idx_identify.get('x-ao')]) > 0)

    # getting epoch estimates to be filled
    hvac_idx_map = disagg_output_object.get('output_write_idx_map').get('hvac_smb')
    ac_idx = hvac_idx_map[0]
    sh_idx = hvac_idx_map[1]
    epoch_estimates = disagg_output_object.get('epoch_estimate')

    # getting base appliance profile and lifestyle profile for user
    user_profile_object = disagg_output_object.get('appliance_profile')
    user_lifestyle_object = disagg_output_object.get('lifestyle_profile')

    logger_hvac_profile.info(' Starting bill cycle profile writing |')

    # filling profiles for each billing cycles
    for billcycle_start, billcycle_end in out_bill_cycles:

        base = user_profile_object[billcycle_start].get('profileList')[0]
        info = disagg_output_object.get('created_hsm').get('hvac').get('attributes')

        # Fill Heating Profile -----------------------------------------
        heating_profile = base['3'][0]
        month_consumptions = month_estimates[month_estimates[:,0] == billcycle_start]

        # getting heating consumption
        total_ao, total_heating, open_heating, close_heating = get_heating_consumption(month_consumptions, month_idx_identify)

        # getting epoch level ac and sh consumption for current billing cycle
        month_epoch_consumptions = epoch_estimates[disagg_input_object.get('input_data')[:, 0] == billcycle_start]
        month_epoch_ac = month_epoch_consumptions[:, ac_idx]
        month_epoch_sh = month_epoch_consumptions[:, sh_idx]

        # populating heating profiles
        heating_profile['validity'] = dict()
        heating_profile['validity']['start'] = int(billcycle_start)
        heating_profile['validity']['end'] = int(billcycle_end)
        heating_profile['isPresent'] = has_sh
        heating_profile['detectionConfidence'] = 1.0
        heating_profile['count'] = len(info['sh_means'])
        heating_profile['attributes']['heatingConsumption'] = float(total_heating)
        heating_profile['attributes']['inWorkingHoursAtMonth'] = float(open_heating)
        heating_profile['attributes']['inOffHoursAtMonth'] = float(close_heating)
        heating_profile['attributes']['fuelType'] = None
        heating_profile['attributes']['secondaryFuelType'] = None
        heating_profile['attributes']['usageModeCount'] = len(info.get('sh_means'))
        heating_profile['attributes']['usageModeCountConfidence'] = 1.0
        heating_profile['attributes']['modeAmplitudes'] = np.array(info.get('sh_mode_limits_limits')).tolist()
        heating_profile['attributes']['regressionCoeff'] = np.array(info.get('sh_cluster_info_keys_coefficient')).astype(float).tolist()
        heating_profile['attributes']['regressionCoeffConfidence'] = np.ones(len(info.get('sh_cluster_info_keys_coefficient'))).astype((float)).tolist()
        heating_profile['attributes']['regressionType'] = ['linear' if el == 1 else 'root' if el==0.5 else None for el in info['sh_cluster_info_keys_kind']]
        heating_profile['attributes']['aoHeating'] = None
        heating_profile['attributes']['onDemandHeating'] = None
        heating_profile['attributes']['heatingMeans'] = np.array(info.get('sh_means')).astype(float).tolist()
        heating_profile['attributes']['heatingStd'] = np.array(info.get('sh_std')).astype(float).tolist()
        heating_profile['attributes']['timeOfUsage'] = np.around(month_epoch_sh,2).astype(float).tolist()
        user_profile_object[billcycle_start]['profileList'][0]['3'][0] = heating_profile
        logger_hvac_profile.info(' Done heating profile for {} bill cycle |'.format(billcycle_start))

        # Fill Cooling profile -----------------------------------------
        cooling_profile = base['4'][0]

        # getting cooling consumption
        total_cooling, open_cooling, close_cooling = get_cooling_consumption(month_consumptions, month_idx_identify)

        # populating cooling profiles
        cooling_profile['validity'] = dict()
        cooling_profile['validity']['start'] = int(billcycle_start)
        cooling_profile['validity']['end'] = int(billcycle_end)
        cooling_profile['isPresent'] = has_ac
        cooling_profile['detectionConfidence'] = 1.0
        cooling_profile['count'] = len(info['ac_means'])
        cooling_profile['attributes']['coolingConsumption'] = float(total_cooling)
        cooling_profile['attributes']['inWorkingHoursAtMonth'] = float(open_cooling)
        cooling_profile['attributes']['inOffHoursAtMonth'] = float(close_cooling)
        cooling_profile['attributes']['fuelType'] = None
        cooling_profile['attributes']['secondaryFuelType'] = None
        cooling_profile['attributes']['usageModeCount'] = len(info.get('ac_means'))
        cooling_profile['attributes']['usageModeCountConfidence'] = 1.0
        cooling_profile['attributes']['modeAmplitudes'] = np.array(info.get('ac_mode_limits_limits')).tolist()
        cooling_profile['attributes']['regressionCoeff'] = np.array(info.get('ac_cluster_info_keys_coefficient')).astype(float).tolist()
        cooling_profile['attributes']['regressionCoeffConfidence'] = np.ones(len(info.get('ac_cluster_info_keys_coefficient'))).astype((float)).tolist()
        cooling_profile['attributes']['regressionType'] = ['linear' if el == 1 else 'root' if el == 0.5 else None for el in info.get('ac_cluster_info_keys_kind')]
        cooling_profile['attributes']['aoCooling'] = None
        cooling_profile['attributes']['onDemandCooling'] = None
        cooling_profile['attributes']['coolingMeans'] = np.array(info.get('ac_means')).astype(float).tolist()
        cooling_profile['attributes']['coolingStd'] = np.array(info.get('ac_std')).astype(float).tolist()
        cooling_profile['attributes']['coolingType'] = None
        cooling_profile['attributes']['timeOfUsage'] = np.around(month_epoch_ac, 2).astype(float).tolist()

        user_profile_object[billcycle_start]['profileList'][0]['4'][0] = cooling_profile
        logger_hvac_profile.info(' Done cooling profile for {} bill cycle |'.format(billcycle_start))

        # Fill AO Profile -----------------------------------------
        ao_profile = base['8'][0]
        ao_profile['validity'] = dict()
        ao_profile['validity']['start'] = int(billcycle_start)
        ao_profile['validity']['end'] = int(billcycle_end)
        ao_profile['isPresent'] = has_ao
        ao_profile['detectionConfidence'] = 1.0
        ao_profile['count'] = len(info.get('ac_means'))
        ao_profile['attributes']['fuelType'] = "Electric"
        ao_profile['attributes']['aoConsumption'] = float(total_ao)
        user_profile_object[billcycle_start]['profileList'][0]['8'][0] = ao_profile
        logger_hvac_profile.info(' Done AO profile for {} bill cycle |'.format(billcycle_start))

        # Fill Operational Load Profile -----------------------------------------
        op_consumption = get_op_consumption(month_consumptions, month_idx_identify)

        op_profile = base['81'][0]
        op_profile['validity'] = dict()
        op_profile['validity']['start'] = int(billcycle_start)
        op_profile['validity']['end'] = int(billcycle_end)
        op_profile['isPresent'] = has_op
        op_profile['detectionConfidence'] = 1.0
        op_profile['count'] = 1
        op_profile['attributes']['operationalLoadConsumption'] = float(op_consumption)
        op_profile['attributes']['samplingRate'] = int(disagg_input_object.get('config').get('sampling_rate'))
        op_profile['attributes']['operationalLoadAtHour'] = None
        user_profile_object[billcycle_start]['profileList'][0]['81'][0] = op_profile
        logger_hvac_profile.info(' Done OP profile for {} bill cycle |'.format(billcycle_start))

        # Fill Extra AO Profile -----------------------------------------
        x_ao_consumption = get_xao_consumption(month_consumptions, month_idx_identify)

        x_ao_profile = base['82'][0]
        x_ao_profile['validity'] = dict()
        x_ao_profile['validity']['start'] = int(billcycle_start)
        x_ao_profile['validity']['end'] = int(billcycle_end)
        x_ao_profile['isPresent'] = has_x_ao
        x_ao_profile['detectionConfidence'] = 1.0
        x_ao_profile['count'] = 1
        x_ao_profile['attributes']['anomalousLoadConsumption'] = float(x_ao_consumption)
        x_ao_profile['attributes']['dayOfUsage'] = None
        user_profile_object[billcycle_start]['profileList'][0]['82'][0] = x_ao_profile
        logger_hvac_profile.info(' Done X-AO profile for {} bill cycle |'.format(billcycle_start))

        disagg_output_object['appliance_profile'] = user_profile_object
        logger_hvac_profile.info(' Updated AO-HVAC profile in disagg output object for {} bill cycle |'.format(billcycle_start))

        # Fill Lifestyle-11 Profile
        month_lifestyle = disagg_output_object.get('special_outputs').get('smb_outputs').get('lifestyle_11')[billcycle_start]
        user_lifestyle_object[billcycle_start]['profileList'][0]['lifestyleid_11'] = month_lifestyle
        logger_hvac_profile.info(' Done lifestyle profile for {} bill cycle |'.format(billcycle_start))
        disagg_output_object['lifestyle_profile'] = user_lifestyle_object

        validate_appliance_profile_schema_for_billcycle(disagg_output_object, billcycle_start, logger_pass)
        validate_lifestyle_profile_schema_for_billcycle(disagg_output_object, billcycle_start, logger_pass)

    return disagg_output_object
