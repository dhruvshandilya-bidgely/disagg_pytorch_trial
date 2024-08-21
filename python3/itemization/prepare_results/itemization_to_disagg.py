"""
Author - Sahana M
Date - 01 September 2021
After the completion of the pipeline this function arranges the output in the format needed and returns as a dictionary
"""

# Import python packages
import logging
import numpy as np
from copy import deepcopy

# Import packages from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.find_seq import find_seq
from python3.config.mappings.get_app_id import get_app_id

from python3.config.pilot_constants import PilotConstants
from python3.config.pipeline_constants import GlobalConfigParams

from python3.utils.validate_output_schema import validate_appliance_profile_schema_for_billcycle


def prepare_appid_dict(app_name_list, output_write_idx_map, app_id_dict):
    """
    Function to return the appid dictionary

    Parameters:
        app_name_list           (list): list of appliances
        output_write_idx_map    (dict): contains writing indexes corresponding to app ids
        app_id_dict             (dict): contains mapping of app name to app id

    Returns:
        app_id_dict             (dict): contains mapping of app name to app id
    """

    for app_name in app_name_list:
        if app_name == 'hvac' or app_name == 'hvac_smb':
            col_idx = output_write_idx_map.get(app_name)
            app_id_dict[col_idx[0]] = get_app_id('ac')
            app_id_dict[col_idx[1]] = get_app_id('sh')
        elif app_name == 'va':
            col_idx = output_write_idx_map.get(app_name)
            app_id_dict[col_idx[0]] = get_app_id('va')
            app_id_dict[col_idx[1]] = get_app_id('vad')
        elif app_name == 'ao' or app_name == 'ao_smb':
            app_id_dict[output_write_idx_map.get(app_name)] = get_app_id('ao')
        else:
            app_id_dict[output_write_idx_map.get(app_name)] = get_app_id(app_name)

    return app_id_dict


def check_over_write(appliances_to_disagg, item_input_object, item_output_object, run_successful):

    """
    This function is used to identify the appliances for which merging is needed
    Parameters:
        appliances_to_disagg            (list)      : List containing appliances
        item_output_object              (dict)      : Contains pipeline output objects
        run_successful            (bool)              : true if hybrid v2 run was successful

    Returns:
        check_over_write_dict           (dict)      : Boolean mapping for each appliance
    """

    check_over_write_dict = dict()

    pilot = item_input_object.get("config").get("pilot_id")

    post_processing_wh_enabled = 0

    if ('wh' in GlobalConfigParams.disagg_postprocess_enabled_app) and \
            (pilot not in PilotConstants.HYBRID_WH_DISABLED_PILOTS):
        post_processing_wh_enabled = 1

    if ('seasonal_wh' in item_output_object.get('special_outputs').keys()) and \
            item_output_object.get('special_outputs').get('seasonal_wh').get('user_detection') == 1:
        post_processing_wh_enabled = 1

    hybrid_v2_enabled = item_input_object.get('item_input_params').get('run_hybrid_v2_flag') and run_successful

    # For each appliance check if the merging is needed or not

    for appliance in appliances_to_disagg:

        # Set default

        check_over_write_dict[appliance] = False

        # If the appliance's hld is 1 then merging can be performed

        if appliance in ['others', 'pp', 'ev', 'hvac', 'li', 'wh']:
            check_over_write_dict[appliance] = True

        if post_processing_wh_enabled and appliance == 'wh':
            check_over_write_dict[appliance] = True

        if appliance in ['ref', 'ao', 'wh'] and hybrid_v2_enabled:
            check_over_write_dict[appliance] = True

        if appliance in GlobalConfigParams.hybrid_v2_additional_app:
            check_over_write_dict[appliance] = True

    return check_over_write_dict


def populate_gb_monthly_output(app_id_dict, column_idx, out_bill_cycles, bill_cycle_raw_idx, value):
    """
    Parameters:
        app_id_dict         (dict)      : Contains mapping of app ids
        column_idx          (dict)      : Contains mapping of column indexes
        out_bill_cycles     (np.ndarray): Array of out billing cycles
        bill_cycle_raw_idx  (int)       : Contains current bill cycle raw index
        value               (float)     : Contains bill cycle level consumption value

    Return:
        output_instance     (dict)      : Dictionary containing gb monthly output instance
    """
    output_instance = {
        "appId": app_id_dict[column_idx],
        "start": int(out_bill_cycles[bill_cycle_raw_idx, 0]),
        "end": int(out_bill_cycles[bill_cycle_raw_idx, 1]),
        "value": value,
        "tb": "null",
        "weekday": "null",
        "detectionConfidence": 1,
        "estimationConfidence": 1
    }

    return output_instance


def get_gb_monthly_output(monthly_output_maker):

    """
    Function to return the gb monthly output payload

    Parameters:
        monthly_output_maker    (dict)  : Contains objects needed to prepare gb monthly output instances

    Returns:
        gb_monthly_output       (list)  : Contains gb monthly output in list form for each billing cycle
    """

    out_bill_cycles = monthly_output_maker.get('out_bill_cycles')
    bill_cycle_raw_idx = monthly_output_maker.get('bill_cycle_raw_idx')
    bill_cycle_estimate = monthly_output_maker.get('bill_cycle_estimate')
    app_id_dict = monthly_output_maker.get('app_id_dict')
    gb_monthly_output = monthly_output_maker.get('gb_monthly_output')

    bc_start = out_bill_cycles[bill_cycle_raw_idx, 0]
    bill_cycle_idx = np.where(bill_cycle_estimate[:, 0] == bc_start)[0][0]

    for column_idx in app_id_dict.keys():

        value = bill_cycle_estimate[bill_cycle_idx, column_idx]

        if np.isnan(value) or (value == 0) or (app_id_dict[column_idx] == -1):
            continue

        output_instance = populate_gb_monthly_output(app_id_dict, column_idx, out_bill_cycles,
                                                     bill_cycle_raw_idx, value)

        gb_monthly_output.append(output_instance)

    return gb_monthly_output


def populate_tb_instance(column_idx, item_output_object, app_id, params_dict):

    """
    Populates tb instance dictionary for a given app id

    Parameters:
        column_idx          (int)               : The column in which the data for the appliance is present
        item_output_object(dict)              : Contains all outputs generated by running the pipeline
        app_id              (int)               : The appliance id for which the dict is to be prepared
        params_dict         (dict)              : Dictionary containing parameters used to populate the instance

    Returns:
        skip_bool           (bool)              : Boolean indicating if the entry should be skipped
        tb_instance         (dict)              : Dictionary containing time band outputs for appliance id
    """

    # Extract Parameters

    epoch_estimate = params_dict.get('epoch_estimate')
    hourly_estimate = params_dict.get('hourly_estimate')
    valid_idx = params_dict.get('valid_idx')
    valid_idx_hr = params_dict.get('valid_idx_hr')
    start_arr_hr = params_dict.get('start_arr_hr')
    end_arr_hr = params_dict.get('end_arr_hr')
    start_arr = params_dict.get('start_arr')
    end_arr = params_dict.get('end_arr')
    bc_start = params_dict.get('bc_start')
    bc_end = params_dict.get('bc_end')
    sampling_rate = params_dict.get('sampling_rate')

    # Initialize column name constants

    seq_val_idx = 0
    seq_start_idx = 1
    seq_end_idx = 2

    if column_idx in item_output_object.get('output_write_idx_map').get('va', []):

        # Prepare output for vacation

        # Extract value array

        value_arr = np.round(hourly_estimate[valid_idx_hr, column_idx], 5)

        # Check if all values are nan do not write

        non_nan_count = np.sum(np.logical_not(np.isnan(value_arr)))

        if non_nan_count == 0:
            return True, {}

        # Set all nans to 0 to avoid errors

        value_arr[np.isnan(value_arr)] = 0

        # Compress the data to write by merging together intervals with the same value

        val_seq_arr = find_seq(value_arr, min_seq_length=0)

        comp_start_arr = start_arr_hr[val_seq_arr[:, seq_start_idx].astype(int)]
        comp_end_arr = end_arr_hr[val_seq_arr[:, seq_end_idx].astype(int)]
        comp_val_arr = val_seq_arr[:, seq_val_idx]

        tb_instance = {
            "appId": app_id,
            "start": bc_start,
            "end": bc_end,
            "granuality": Cgbdisagg.SEC_IN_HOUR,
            "tbStartList": comp_start_arr.tolist(),
            "tbEndList": comp_end_arr.tolist(),
            "tbValues": comp_val_arr.tolist()
        }

    else:

        # Extract value array

        value_arr = np.round(epoch_estimate[valid_idx, column_idx], 5)

        # Check if all values are nan do not write

        non_nan_count = np.sum(np.logical_not(np.isnan(value_arr)))

        if non_nan_count == 0:
            return True, {}

        # Set all nans to 0 to avoid errors

        value_arr[np.isnan(value_arr)] = 0

        # Compress the data to write by merging together intervals with the same value

        val_seq_arr = find_seq(value_arr, min_seq_length=0)

        comp_start_arr = start_arr[val_seq_arr[:, seq_start_idx].astype(int)]
        comp_end_arr = end_arr[val_seq_arr[:, seq_end_idx].astype(int)]
        comp_val_arr = val_seq_arr[:, seq_val_idx]

        tb_instance = {
            "appId": app_id,
            "start": bc_start,
            "end": bc_end,
            "granuality": sampling_rate,
            "tbStartList": comp_start_arr.tolist(),
            "tbEndList": comp_end_arr.tolist(),
            "tbValues": comp_val_arr.tolist()
        }

    return False, tb_instance


def prepare_time_band_disagg(item_input_object, item_output_object, epoch_estimate, app_id_dict):

    """
    Prepares timestamp level output objects containing disagg values

    Parameters:
        item_input_object (dict)              : Contains all inputs required to run the pipeline
        item_output_object(dict)              : Contains all outputs generated by running the pipeline
        app_id_dict         (dict)              : Contains mapping of estimate col idx to appliance id

    Returns:
        tb_output           (list)              : List of dictionaries containing all time band outputs
    """

    # Initialize column name constants

    ts_col_idx = 0
    bc_start_idx = 0
    bc_end_idx = 1

    # Initialize the tb output list

    tb_output = []

    # Down sample each appliance output to hour level

    sampling_rate = item_input_object.get('config').get('sampling_rate')

    # Initialize out bill cycles

    out_bill_cycles = item_input_object.get('out_bill_cycles_by_module').get('disagg_tou')
    num_out_billing_cycles = out_bill_cycles.shape[0]

    # Extract appliances to write TOU for

    tou_out_app_id_list = None

    if item_input_object.get('gb_pipeline_event').get('disaggRunMode') is not None:
        disagg_run_mode = item_input_object.get('gb_pipeline_event').get('disaggRunMode').get('applianceDisaggRunMode')
        tou_out_app_id_list = disagg_run_mode.get('timebandDisaggModeData').get('applianceIds')

    # Extract hourly data for vacation

    input_data = deepcopy(item_input_object.get('input_data'))

    bill_cycle_hourly_idx = (input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] -
                             input_data[0, Cgbdisagg.INPUT_BILL_CYCLE_IDX]) // Cgbdisagg.SEC_IN_HOUR

    epoch_hourly_idx = bill_cycle_hourly_idx + (input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] -
                                                input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]) // Cgbdisagg.SEC_IN_HOUR

    _, epoch_hourly_unique_idx = np.unique(epoch_hourly_idx, axis=0, return_index=True)

    hourly_estimate = epoch_estimate[epoch_hourly_unique_idx, :]

    # In case the first timestamp is out of sync in granularity adjust

    first_point_mod = (hourly_estimate[0, 0] - input_data[0, Cgbdisagg.INPUT_BILL_CYCLE_IDX]) % Cgbdisagg.SEC_IN_HOUR

    if first_point_mod > 0:
        hourly_estimate[0, 0] = hourly_estimate[0, 0] - first_point_mod

    # For each appliance and for each billing cycle prepare the tb output dictionary

    for bill_cycle_idx in range(num_out_billing_cycles):

        # Identify valid indices for the current bill cycle

        bc_start = int(out_bill_cycles[bill_cycle_idx, bc_start_idx])
        bc_end = int(out_bill_cycles[bill_cycle_idx, bc_end_idx])

        valid_idx = np.logical_and(epoch_estimate[:, ts_col_idx] >= bc_start, epoch_estimate[:, ts_col_idx] < bc_end)

        valid_idx_hr = np.logical_and(hourly_estimate[:, ts_col_idx] >= bc_start,
                                      hourly_estimate[:, ts_col_idx] < bc_end)

        # Prepare start and end array for usage

        start_arr = epoch_estimate[valid_idx, ts_col_idx]
        end_arr = start_arr + sampling_rate

        start_arr_hr = hourly_estimate[valid_idx_hr, ts_col_idx]
        end_arr_hr = start_arr_hr + Cgbdisagg.SEC_IN_HOUR

        for column_idx in app_id_dict.keys():

            # Extract app id corresponding to the column

            app_id = app_id_dict.get(column_idx)

            # Check if TOU for this appliance has to be written or not

            if (tou_out_app_id_list is not None) and (app_id not in tou_out_app_id_list):
                continue

            if app_id == -1:
                continue

            params_dict = {
                'epoch_estimate': epoch_estimate,
                'hourly_estimate': hourly_estimate,
                'valid_idx': valid_idx,
                'valid_idx_hr': valid_idx_hr,
                'start_arr_hr': start_arr_hr,
                'end_arr_hr': end_arr_hr,
                'start_arr': start_arr,
                'end_arr': end_arr,
                'bc_start': bc_start,
                'bc_end': bc_end,
                'sampling_rate': sampling_rate,
            }

            skip_bool, tb_instance = populate_tb_instance(column_idx, item_output_object, app_id, params_dict)

            if skip_bool:
                continue

            tb_output.append(tb_instance)

    return tb_output


def populate_appliance_profile(item_input_object, item_output_object, empty_appliance_profile, logger, logger_pass):

    """
    Populate appliance profile after schema validation for each billcycle

    Parameters:
        item_input_object              (dict)              : Contains all inputs required to run the pipeline
        item_output_object             (dict)              : Contains all outputs generated by running the pipeline
        empty_appliance_profile          (dict)              : Initialized appliance profile dictionary
        logger                           (logger)            : Logger object
        logger_pass                      (dict)              : Contains all variables needed for logging

    Returns:
        appliance_profile_output         (dict)              : Final prepared and validated appliance profile
    """

    appliance_profile_filled = item_output_object.get('appliance_profile')

    appliance_profile_output = deepcopy(empty_appliance_profile)

    # Writing appliance profiles for all required bill cycles

    appliance_profile_fill_bill_cycles = item_input_object.get("out_bill_cycles_by_module").get("app_profile")

    for bill_cycle_raw_idx in range(len(appliance_profile_fill_bill_cycles)):

        bc_start = appliance_profile_fill_bill_cycles[bill_cycle_raw_idx, 0]

        # validate appliance profile for this bill cycle

        valid_flag = validate_appliance_profile_schema_for_billcycle(item_output_object, bc_start, logger_pass)

        if valid_flag:

            appliance_profile_bc = appliance_profile_filled[bc_start]['profileList'][0]

            appliance_profile_output['profileList'].append(appliance_profile_bc)

            logger.info("Writing appliance profile for billcycle | %d " % bc_start)

        else:

            logger.error("Skipping writing appliance profile for billcycle | %d" % bc_start)

    return appliance_profile_output


def itemization_to_disagg(appliances_to_disagg, item_input_object, item_output_object, pipeline_output_object, run_successful):
    """
    This function is used to merge selected appliance's  Itemization output into disagg
    Parameters:
        item_input_object (dict)                     : Contains all inputs required to run the pipeline
        item_output_object(dict)                     : Contains all outputs generated by running the pipeline
        pipeline_output_object(dict)                 : Contains pipeline output objects
        run_successful            (bool)              : true if hybrid v2 run was successful

    Returns:
        itemization_api_output (dict)                : Contains the new merged API output
    """

    # Initialize the logger

    logger_base = item_input_object.get('logger').getChild('itemization_to_disagg')
    logger = logging.LoggerAdapter(logger_base, item_input_object.get('logging_dict'))

    logger_pass = {
        'logger_base': logger_base,
        'logging_dict': item_input_object.get('logging_dict'),
    }

    # Get the Pipeline API output

    itemization_api_output = deepcopy(pipeline_output_object.get('api_output'))

    # Appliances for which Itemization output should be merged with Disagg output

    output_write_idx_map = dict()

    for appliance in appliances_to_disagg:
        output_write_idx_map[appliance] = item_output_object.get('output_write_idx_map').get(appliance)

    # Get the appliances and their app id

    output_write_idx_map['cook'] = 14
    output_write_idx_map['ent'] = 13
    output_write_idx_map['ld'] = 12
    output_write_idx_map['others'] = 15

    additional_app = ['li', 'cook', 'ent', 'ld']

    app_name_list = list(output_write_idx_map.keys())
    app_id_dict_all = {}

    # Generate a mapping from column number to corresponding app id with which backend will save the results

    app_id_dict_all = prepare_appid_dict(app_name_list, output_write_idx_map, app_id_dict_all)

    # Check if Merging with Disagg is needed for each appliance

    appliances_to_disagg_copy = deepcopy(appliances_to_disagg)

    if item_input_object.get("pilot_level_config_present") is not None and item_input_object.get("pilot_level_config_present") > 0:
        appliances_to_disagg_copy = np.append(appliances_to_disagg, 'others')

    check_over_write_dict = check_over_write(appliances_to_disagg_copy, item_input_object, item_output_object, run_successful)

    # Refine the app_id_dict based on the check over write dictionary
    app_id_dict = dict()

    for appliance in appliances_to_disagg_copy:
        if check_over_write_dict[appliance]:

            if appliance == 'hvac':
                app_id = output_write_idx_map[appliance][0]
                app_id_dict[app_id] = app_id_dict_all[app_id]

                app_id = output_write_idx_map[appliance][1]
                app_id_dict[app_id] = app_id_dict_all[app_id]

            else:
                app_id = output_write_idx_map[appliance]
                app_id_dict[app_id] = app_id_dict_all[app_id]

    itemization_api_output, item_output_object = prepare_api_output(run_successful, item_input_object, item_output_object, itemization_api_output, app_id_dict, logger, logger_pass)

    # Populate gbOutputStatus

    disagg_metrics_dict = item_output_object['disagg_metrics']

    if itemization_api_output.get('gbOutputStatus') is not None and itemization_api_output.get('gbOutputStatus').get('gbAppStatus') is not None:
        gb_app_status_list = itemization_api_output.get('gbOutputStatus').get('gbAppStatus')
    else:
        gb_app_status_list = []

    for app_name in additional_app:
        if disagg_metrics_dict.get(app_name) is not None:
            status_instance = {
                "appId": get_app_id(app_name),
                "exitCode": disagg_metrics_dict.get(app_name).get('exit_status'),
                "processingTime": np.round(1000 * disagg_metrics_dict.get(app_name).get('time'), 3)
            }
            gb_app_status_list.append(status_instance)

    itemization_api_output['gbOutputStatus']['gbAppStatus'] = gb_app_status_list

    return itemization_api_output, item_output_object


def get_hybrid_v2_app_list(item_input_object, itemization_api_output):

    """
    This function is used to merge selected appliance's  Itemization output into disagg
    Parameters:
        item_input_object (dict)                     : Contains all inputs required to run the pipeline
        itemization_api_output (dict)                : Contains the API output

    Returns:
        itemization_api_output (dict)                : Contains the new merged API output
    """

    if itemization_api_output.get('hybridMetaData') is None:
        itemization_api_output['hybridMetaData'] = dict()

    stat_app_list = GlobalConfigParams.hybrid_v2_additional_app

    if item_input_object.get("item_input_params") is not None and item_input_object.get("item_input_params").get("backup_app") is not None:
        stat_app_list = np.append(stat_app_list, item_input_object["item_input_params"]["backup_app"])

    stat_app_id_list = []

    for app in stat_app_list:
        stat_app_id_list.append(get_app_id(app))

    return itemization_api_output, stat_app_id_list


def prepare_api_output(run_successful, item_input_object, item_output_object, itemization_api_output, app_id_dict, logger, logger_pass):

    """
    This function is used to merge selected appliance's  Itemization output into disagg
    Parameters:
        item_input_object (dict)                     : Contains all inputs required to run the pipeline
        item_output_object(dict)                     : Contains all outputs generated by running the pipeline
        itemization_api_output (dict)                : Contains the API output
        app_id_dict (dict)                           : app ids dict
        logger                  (logger)             : logger object

    Returns:
        itemization_api_output (dict)                : Contains the new merged API output
    """

    itemization_api_output, stat_app_id_list = get_hybrid_v2_app_list(item_input_object, itemization_api_output)

    itemization_api_output['hybridMetaData']['hybridAppIds'] = list(stat_app_id_list)
    itemization_api_output['hybridMetaData']['runHybridV2'] = bool(item_input_object.get('global_config').get('enable_hybrid_v2'))

    logger.info('Statistical appliance output list | %s', list(stat_app_id_list))
    logger.info('Hybrid v2 run output flag | %s', bool(item_input_object.get('global_config').get('enable_hybrid_v2')))
    #Prepare others output

    out_bill_cycles = item_input_object.get('out_bill_cycles_by_module').get('disagg_bc')

    # Populate gbMonthlyOutput
    bill_cycle_estimate = np.round(item_output_object.get('bill_cycle_estimate'), decimals=2)
    epoch_estimate = deepcopy(item_output_object.get('epoch_estimate'))

    bc_level_others, ts_level_others = calculate_others(item_input_object, epoch_estimate, bill_cycle_estimate, run_successful)

    bill_cycle_estimate = np.hstack((bill_cycle_estimate[:, :item_output_object.get('bill_cycle_estimate').shape[1]], bc_level_others[:, None]))
    item_output_object['bill_cycle_estimate'] = bill_cycle_estimate
    epoch_estimate =  np.hstack((epoch_estimate[:, :item_output_object.get('bill_cycle_estimate').shape[1]], ts_level_others[:, None]))

    item_output_object['epoch_estimate'] = epoch_estimate

    if len(app_id_dict):

        logger.info('Performing Merging of Itemization to Disagg | ')

        # Populate gbMonthlyOutput with the new values

        num_columns = bill_cycle_estimate.shape[1]

        num_out_billing_cycles = out_bill_cycles.shape[0]

        gb_monthly_output = []

        for bill_cycle_raw_idx in range(num_out_billing_cycles):
            monthly_output_maker = {'out_bill_cycles': out_bill_cycles,
                                    'bill_cycle_raw_idx': bill_cycle_raw_idx,
                                    'bill_cycle_estimate': bill_cycle_estimate,
                                    'num_columns': num_columns,
                                    'app_id_dict': app_id_dict,
                                    'item_output_object': item_output_object,
                                    'gb_monthly_output': gb_monthly_output}

            gb_monthly_output = get_gb_monthly_output(monthly_output_maker)

        # Update the newly added appliance Monthly outputs

        for idx in range(len(gb_monthly_output)):
            itemization_api_output['gbMonthlyOutput'].append(gb_monthly_output[idx])

        # Populate Time band level output

        gb_tb_output = prepare_time_band_disagg(item_input_object, item_output_object, epoch_estimate, app_id_dict)

        # Remove previously filled values for these appliances

        indexes = []
        for appliance in app_id_dict.values():
            for idx in range(len(itemization_api_output['gbTBOutput'])):
                if itemization_api_output['gbTBOutput'][idx]['appId'] != appliance:
                    indexes.append(idx)

        indexes = np.unique(indexes)

        # Create a new TB output array to append the newly filled Tb output

        temp = []
        for idx in indexes:
            temp.append(itemization_api_output['gbTBOutput'][idx])

        itemization_api_output['gbTBOutput'] = temp

        # Update the newly added appliance Time Band outputs

        for idx in range(len(gb_tb_output)):
            itemization_api_output['gbTBOutput'].append(gb_tb_output[idx])

        # Initialize empty appliance profile

        empty_appliance_profile = {
            "version": "v1",
            "profileList": []
        }

        # Populate the new Appliance profile values

        appliance_profile_output = populate_appliance_profile(item_input_object, item_output_object, empty_appliance_profile, logger, logger_pass)

        itemization_api_output['applianceProfile'] = appliance_profile_output

    return itemization_api_output, item_output_object


def calculate_others(item_input_object, epoch_estimate, bill_cycle_estimate, run_successful):

    """
    This function is used to calculate ts level and bill cycle level others
    Parameters:
        item_input_object       (dict)                      : Contains all inputs required to run the pipeline
        epoch_estimate          (np.ndarray)                : ts level disagg output
        bill_cycle_estimate     (np.ndarray)                : bc level disagg output

    Returns:
        bc_level_others         (np.ndarray)                : bc level others output
        ts_level_others         (np.ndarray)                : ts level others output
    """

    if item_input_object.get("input_data_without_outlier_removal") is None:
        disagg_input_data = np.zeros(len(epoch_estimate))
    else:
        disagg_input_data = deepcopy(item_input_object.get("input_data_without_outlier_removal")[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    disagg_epoch_estimate = deepcopy(item_input_object.get('disagg_epoch_estimate'))
    disagg_bill_cycle_estimate = deepcopy(np.round(item_input_object.get('disagg_bill_cycle_estimate'), decimals=2))
    billing_cycle_list = item_input_object.get("input_data")[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]

    out_bill_cycles = bill_cycle_estimate[:, 0]

    bc_level_input_data = np.zeros(len(out_bill_cycles))

    for i in range(len(out_bill_cycles)):
        bc_level_input_data[i] = np.sum(disagg_input_data[billing_cycle_list == out_bill_cycles[i]])

    bc_level_others = bc_level_input_data
    ts_level_others = disagg_input_data

    output_write_idx_map = item_input_object.get('disagg_output_write_idx_map')

    disagg_app_ids = []
    disagg_post_processing_app_ids = []

    res_true_disagg_app_list = GlobalConfigParams.residential_true_disagg_app_list

    hybrid_v2_run_successful = item_input_object.get('item_input_params').get('run_hybrid_v2_flag') and run_successful

    if hybrid_v2_run_successful:
        res_true_disagg_app_list = GlobalConfigParams.item_aer_seq

    output_write_idx_map['cook'] = 14
    output_write_idx_map['ent'] = 13
    output_write_idx_map['ld'] = 12

    for app in res_true_disagg_app_list:

        if (app in GlobalConfigParams.disagg_postprocess_enabled_app) or (app in GlobalConfigParams.itemization_seq) or hybrid_v2_run_successful:
            disagg_post_processing_app_ids = np.append(disagg_post_processing_app_ids, output_write_idx_map.get(app))
        else:
            disagg_app_ids = np.append(disagg_app_ids, output_write_idx_map.get(app))

    disagg_app_ids = np.array(disagg_app_ids).astype(int)
    disagg_post_processing_app_ids = np.array(disagg_post_processing_app_ids).astype(int)

    bc_level_others = bc_level_others - np.nansum(bill_cycle_estimate[:, disagg_post_processing_app_ids], axis=1)  - \
                      np.nansum(disagg_bill_cycle_estimate[:, disagg_app_ids], axis=1)

    ts_level_others = ts_level_others - np.nansum(epoch_estimate[:, disagg_post_processing_app_ids], axis=1)  - \
                      np.nansum(disagg_epoch_estimate[:, disagg_app_ids], axis=1)

    return bc_level_others, ts_level_others
