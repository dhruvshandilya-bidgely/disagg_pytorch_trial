"""
Author - Mayank Sharan
Date - 11th Oct 2018
initialisation pipeline input objects reads things fetched from the API and returns a list of objects for each gb pipeline event
"""

# Import python packages

import copy
import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pipeline_constants import GlobalConfigParams

from python3.config.mappings.get_app_id import get_app_name
from python3.config.mappings.get_disagg_mode import get_disagg_mode


from python3.initialisation.object_initialisations.init_pipeline_input_object import init_pipeline_input_object


def unify_pipeline_mode(pipeline_event):

    """
    Process and unify multiple sub-events if present to identify the pipeline mode to run the pipeline in

    Parameters:
        pipeline_event        (dict)              : Parameters needed to run disaggregation

    Returns:
        disagg_mode         (string)            : disagg_mode mode for the pipeline to run in
        start_ts_arr        (np.ndarray)        : An array of timestamps signifying event start for different modules
        end_ts_arr          (np.ndarray)        : An array of timestamps signifying event end for different modules
    """

    # Initialize the start and end ts array

    start_ts_arr = np.full(shape=(5,), fill_value=pipeline_event.get('start'))
    end_ts_arr = np.full(shape=(5,), fill_value=pipeline_event.get('end'))

    # Check if fragmented modes are present

    if pipeline_event.get('disaggRunMode') is None:
        pipeline_mode = get_disagg_mode(pipeline_event.get('disaggMode'))

    else:

        split_modes_dict = pipeline_event.get('disaggRunMode')

        # Extract the pipeline modes for different modules

        modes_list = [pipeline_event.get('disaggMode'),
                      split_modes_dict.get('applianceDisaggRunMode').get('monthlyDisaggModeData').get('disaggMode'),
                      split_modes_dict.get('applianceDisaggRunMode').get('timebandDisaggModeData').get('disaggMode'),
                      split_modes_dict.get('applianceProfileRunMode').get('disaggModeData').get('disaggMode'),
                      split_modes_dict.get('lifestyleProfileRunMode').get('disaggModeData').get('disaggMode')]

        # Extract the event start ts for different modules

        start_ts_arr = [pipeline_event.get('start'),
                        split_modes_dict.get('applianceDisaggRunMode').get('monthlyDisaggModeData').get('start'),
                        split_modes_dict.get('applianceDisaggRunMode').get('timebandDisaggModeData').get('start'),
                        split_modes_dict.get('applianceProfileRunMode').get('disaggModeData').get('start'),
                        split_modes_dict.get('lifestyleProfileRunMode').get('disaggModeData').get('start')]

        start_ts_arr = np.array(start_ts_arr)

        # Extract the event end ts for different modules

        end_ts_arr = [pipeline_event.get('end'),
                      split_modes_dict.get('applianceDisaggRunMode').get('monthlyDisaggModeData').get('end'),
                      split_modes_dict.get('applianceDisaggRunMode').get('timebandDisaggModeData').get('end'),
                      split_modes_dict.get('applianceProfileRunMode').get('disaggModeData').get('end'),
                      split_modes_dict.get('lifestyleProfileRunMode').get('disaggModeData').get('end')]

        end_ts_arr = np.array(end_ts_arr)

        map_modes_list = []

        for mode in modes_list:
            map_modes_list.append(get_disagg_mode(mode))

        # The functionality for split is to allow historical to run for some modules so we run all in historical
        # and selectively output the results for modules where it was not requested

        overall_event_idx = 0

        if 'historical' in map_modes_list:

            pipeline_mode = 'historical'
            start_ts_arr[overall_event_idx] = np.min(start_ts_arr)
            end_ts_arr[overall_event_idx] = np.max(end_ts_arr)

        else:

            pipeline_mode = get_disagg_mode(pipeline_event.get('disaggMode'))

    return pipeline_mode, start_ts_arr, end_ts_arr


def process_appliance_hsm(hsm_appliances, t_end):

    """
    Parameters:
        hsm_appliances      (dict)              : All hsm for all appliances are here
        t_end               (int)               : Last timestamp for which HSM should be accepted

    Returns:
        hsm_dict            (dict)              : Contains 1 hsm per appliance as we need it
    """

    hsm_dict = dict()
    app_ids = hsm_appliances.keys()

    for app_id in app_ids:

        app_hsm_list = hsm_appliances.get(app_id)

        timestamps = np.array(list(app_hsm_list.keys()))
        timestamps = timestamps.astype(int)

        if len(timestamps) > 0:
            timestamps = timestamps[timestamps < t_end]

        app_name = get_app_name(app_id)

        if len(timestamps) > 0:
            latest_ts = np.max(timestamps)
            temp_hsm = app_hsm_list[str(latest_ts)][0]

            # If multiple HSMs are written for the same timestamp get the latest onw

            if type(temp_hsm) == list:
                temp_hsm = temp_hsm[0]

            hsm_dict[app_name] = temp_hsm
        else:
            hsm_dict[app_name] = dict()

    return hsm_dict


def process_billing_cycles(bill_cycles, start_ts_arr, end_ts_arr):

    """
    Select output bill cycles by filtering them on start and end timestamps for an event

    Parameters:
        bill_cycles         (list)              : Key value pair of all billing cycles to write output for
        start_ts_arr        (np.ndarray)        : An array of timestamps signifying event start for different modules
        end_ts_arr          (np.ndarray)        : An array of timestamps signifying event end for different modules

    Returns:
        proc_bc_dict        (dict)              : Billing cycles adhering to the give t_start and t_end by module
    """

    # Initialize variables needed

    proc_bc_dict = {
        'overall': [],
        'disagg_bc': [],
        'disagg_tou': [],
        'app_profile': [],
        'lifestyle_profile': [],
    }

    modules_list = ['overall', 'disagg_bc', 'disagg_tou', 'app_profile', 'lifestyle_profile']

    num_cycles = len(bill_cycles)

    # Return no bill cycles if t_end is less than t_start happens in the case where event received has no data in the
    # period

    for cycle_idx in range(num_cycles):

        curr_cycle = bill_cycles[cycle_idx]

        # We pick any bill cycle that corresponds to the given pipeline event. Each condition is separated by an or
        # Condition 1 - The start of the bill cycle lies in between the start and end of pipeline event
        # Condition 2 - The end of the bill cycle lies in between the start and end of pipeline event
        # Condition 3 - The pipeline event is contained within the bill cycle

        for module_idx in range(len(modules_list)):

            module_key = modules_list[module_idx]

            t_start = start_ts_arr[module_idx]
            t_end = end_ts_arr[module_idx]

            temp_bc_arr = copy.deepcopy(proc_bc_dict.get(module_key))

            if t_end > curr_cycle['key'] >= t_start or t_end >= curr_cycle['value'] > t_start or \
                    (curr_cycle['value'] > t_start >= curr_cycle['key'] and
                     curr_cycle['value'] >= t_end >= curr_cycle['key']):

                temp_bc_arr.append([curr_cycle['key'], curr_cycle['value']])

            proc_bc_dict[module_key] = temp_bc_arr

    # Convert each list to numpy array

    for module_idx in range(len(modules_list)):
        module_key = modules_list[module_idx]
        temp_bc_arr = copy.deepcopy(proc_bc_dict.get(module_key))

        proc_bc_dict[module_key] = np.array(temp_bc_arr)

    return proc_bc_dict


def process_input_data(input_data, t_end, num_days, pipeline_mode):

    """
    Parameters:
        input_data         (np.ndarray)         : A list of all appliance profiles
        t_end              (int)                : The last timestamp for which data should be sent
        num_days           (int)                : Number of days for which the data needs to be extracted
        disagg_mode        (str)                : The disagg_mode mode in which the event is to be processed

    Returns:
        proc_input_data    (np.ndarray)         : Matrix containing processed input data
    """

    # This is a bug fix temporarily to handle the extra amount of data we might be getting more than 403 days

    if str.lower(pipeline_mode) in ['historical', 'incremental', 'supervised_duration_rerun']:
        num_days = 1000

    t_start = t_end - Cgbdisagg.SEC_IN_DAY * num_days

    idx_to_extract = np.logical_and(t_end > input_data[:, Cgbdisagg.INPUT_EPOCH_IDX],
                                    input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] >= t_start)
    proc_input_data = input_data[idx_to_extract, :]

    return proc_input_data


def restrict_input_data(input_data, pipeline_mode, out_bill_cycles):

    """
    Parameters:
        input_data         (np.ndarray)         : A list of all appliance profiles
        disagg_mode        (str)                : The disagg_mode in which the event is to be processed
        out_bill_cycles    (np.ndarray)         : 2 column matrix containing all bill cycles to write results for

    Returns:
        input_data         (np.ndarray)         : Matrix containing processed input data
    """

    # Get boolean wherever data is in bill cycle for output

    if str.lower(pipeline_mode) in ['historical', 'supervised_duration_rerun']:
        is_pt_in_out_bc = np.in1d(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], out_bill_cycles[:, 0])
        input_data = input_data[is_pt_in_out_bc, :]

    return input_data


def process_app_profile(app_profile):

    """
    Parameters:
        app_profile         (list)              : A list of all appliance profiles

    Returns:
        proc_app_profile    (dict)              : A dictionary of all appliance profiles
    """

    proc_app_profile = dict()

    for idx in range(len(app_profile)):
        curr_app_profile = app_profile[idx]
        curr_app_name = get_app_name(curr_app_profile['appID'])

        if curr_app_name not in GlobalConfigParams.hybrid_v2_additional_app:
            proc_app_profile[curr_app_name] = curr_app_profile
        else:
            proc_app_profile[curr_app_profile['appID']] = curr_app_profile

    return proc_app_profile


def init_pipeline_input_objects(fetch_params, pipeline_object_params, config_params, logger_pass):

    """
    Parameters:
        fetch_params        (dict)              : A dictionary containing all parameters used to fetch data
        pipeline_object_params(dict)              : Contains parameters needed to build the pipeline input object
        config_params       (dict)              : Parameters to be passed on to the config
        logger_pass         (dict)              : Dictionary containing logger and logging dictionary

    Returns:
        pipeline_input_objects(list)              : List of pipeline input objects one for each pipeline event
    """

    # Initialise logger object

    logger_base = logger_pass.get('logger_base').getChild('init_pipeline_input_objects')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Take everything out of the dictionary

    input_data = pipeline_object_params.get('input_data')[:, :Cgbdisagg.INPUT_DIMENSION]
    app_profile = pipeline_object_params.get('app_profile')
    hsm_appliances = pipeline_object_params.get('hsm_appliances')
    pipeline_run_data = pipeline_object_params.get('disagg_run_data')

    # process gb pipeline event, process hsm of appliances and process appliance profile

    # Process all static things first i.e. appliance profile, handle HSM inside pipeline input object because
    # we will want latest HSM as per the requested run time

    proc_app_profile = process_app_profile(app_profile)

    gb_pipeline_events = pipeline_run_data.get('gbDisaggEvents')
    num_events = len(gb_pipeline_events)

    # Create a pipeline input object corresponding to each event

    pipeline_input_objects = []

    for event_idx in range(num_events):

        pipeline_event = gb_pipeline_events[event_idx]

        # Process pipeline event to unify across different modules if needed

        pipeline_mode, start_ts_arr, end_ts_arr = unify_pipeline_mode(pipeline_event)

        # Decide how much input data should be passed here, + 1 for t end to include the last point

        t_end = int(min(input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX] + 1, pipeline_event.get('end')))
        num_days = pipeline_event.get('rawDataDurationInDays')

        proc_input_data = process_input_data(input_data, t_end, num_days, pipeline_mode)

        # Extract billing cycles to write data for, -1 to include the first data point

        t_start = int(max(proc_input_data[0, Cgbdisagg.INPUT_EPOCH_IDX] - 1, np.min(start_ts_arr)))
        t_end = int(min(proc_input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX] + 1, pipeline_event.get('end')))

        proc_billing_cycles_dict = process_billing_cycles(pipeline_run_data.get('outputBillCycles'),
                                                          start_ts_arr, end_ts_arr)

        proc_billing_cycles = proc_billing_cycles_dict.get('overall')

        if len(proc_billing_cycles) == 0:
            logger.warning('No out billing cycles present for pipeline event | %d to %d', t_start, t_end)
            continue

        # For historical mode restrict bill cycles

        proc_input_data = restrict_input_data(proc_input_data, pipeline_mode, proc_billing_cycles)

        # Process HSM list to decide what to send forward

        hsm_dict = process_appliance_hsm(hsm_appliances, t_end)

        # Setup the logging information dictionary to be used

        proc_logging_dict = copy.deepcopy(pipeline_object_params.get('logging_dict'))
        proc_logging_dict['t_start'] = proc_input_data[0, Cgbdisagg.INPUT_EPOCH_IDX]
        proc_logging_dict['t_end'] = proc_input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX]
        proc_logging_dict['pilot_id'] = pipeline_run_data.get('pilotId')
        proc_logging_dict['disagg_mode'] = str.lower(pipeline_event.get('disaggMode'))

        # Initialize config and pipeline input object

        init_params = {
            'app_profile': proc_app_profile,
            'billing_cycles': proc_billing_cycles,
            'billing_cycles_by_module': proc_billing_cycles_dict,
            'disagg_mode': pipeline_mode,
            'pipeline_event': pipeline_event,
            'disagg_run_data': pipeline_run_data,
            'fetch_params': fetch_params,
            'home_meta_data': pipeline_object_params.get('home_meta_data'),
            'hsm_dict': hsm_dict,
            'input_data': proc_input_data,
            'config_params': config_params,
            'logging_dict': proc_logging_dict,
            'loaded_files': fetch_params.get('loaded_files'),
        }

        # Fix issue happening in mode name for incremental

        pipeline_input_object = init_pipeline_input_object(init_params, fetch_params)
        pipeline_input_object['logging_dict']['disagg_mode'] = pipeline_input_object.get('global_config').get('disagg_mode')
        pipeline_input_object['all_hsms_appliance'] = pipeline_object_params.get('hsm_appliances')

        # Append to list of pipeline input objects

        pipeline_input_objects.append(pipeline_input_object)

    return pipeline_input_objects
