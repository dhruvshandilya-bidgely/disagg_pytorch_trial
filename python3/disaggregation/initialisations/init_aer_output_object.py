"""
Author - Mayank Sharan
Date - 15/10/18
initialisations disagg output object initiates single instance of disagg output object
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.initialisations.init_empty_appliance_profile import init_empty_appliance_profile
from python3.initialisation.object_initialisations.init_empty_lifestyle_profile import init_empty_lifestyle_profile


def init_disagg_aer_output_object(disagg_input_object):

    """
    Parameters:
        disagg_input_object (dict)              : Dictionary containing all parameters needed to initialize the object

    Returns:
        disagg_output_object(dict)              : Dictionary containing all parameters to run the pipeline
    """

    disagg_output_object = dict({
        'bill_cycle_estimate': None,
        'disagg_metrics': {},
        'created_hsm': {},
        'epoch_estimate': None,
        'output_write_idx_map': {},
        'special_outputs': {},
    })

    module_seq = np.unique(disagg_input_object.get('config').get('module_seq'))
    num_modules = len(module_seq)

    # Allow extra column for HVAC. Allow 6 extra columns for SMB

    num_columns = num_modules + 1 + int('hvac' in module_seq) + int('va' in module_seq)

    # Initialize billing cycle estimate

    out_bill_cycles = disagg_input_object.get('out_bill_cycles')
    bill_cycle_estimate = np.full(shape=(out_bill_cycles.shape[0], num_columns), fill_value=np.nan)
    bill_cycle_estimate[:, 0] = out_bill_cycles[:, 0]
    disagg_output_object['bill_cycle_estimate'] = bill_cycle_estimate

    # Initialize epoch estimate

    input_ts = disagg_input_object['input_data'][:, Cgbdisagg.INPUT_EPOCH_IDX]
    epoch_estimate = np.full(shape=(len(input_ts), num_columns), fill_value=np.nan)
    epoch_estimate[:, 0] = input_ts
    disagg_output_object['epoch_estimate'] = epoch_estimate

    # Initialize output write idx map, disagg metrics and created hsm

    write_idx = 1

    for module_name in module_seq:
        if module_name == 'hvac' or module_name == 'va':
            disagg_output_object['output_write_idx_map'][module_name] = [write_idx, write_idx + 1]
            disagg_output_object['disagg_metrics'][module_name] = None
            disagg_output_object['created_hsm'][module_name] = None
            write_idx += 2
        else:
            disagg_output_object['output_write_idx_map'][module_name] = write_idx
            disagg_output_object['disagg_metrics'][module_name] = None
            disagg_output_object['created_hsm'][module_name] = None
            write_idx += 1

    disagg_output_object['disagg_metrics']['pipeline'] = None

    # initiate appliance profile object for all out bill cycles

    disagg_output_object['appliance_profile'] = dict()

    # Get default dataRange for this gbdisagg event

    appliance_data_range_start = disagg_input_object.get('out_bill_cycles_by_module').get('app_profile')[0][0]
    appliance_data_range_end = disagg_input_object.get('out_bill_cycles_by_module').get('app_profile')[-1][-1]

    for billcycle_start, billcycle_end in out_bill_cycles:
        disagg_output_object['appliance_profile'][int(billcycle_start)] = init_empty_appliance_profile()

        # set start and end dates for each out bill cycles

        disagg_output_object['appliance_profile'][int(billcycle_start)]['profileList'][0]['start'] = int(
            billcycle_start)
        disagg_output_object['appliance_profile'][int(billcycle_start)]['profileList'][0]['end'] = int(billcycle_end)

        # set start and end for data range for each out bill cycles

        disagg_output_object['appliance_profile'][int(billcycle_start)]['profileList'][0]['dataRange']['start'] = \
            int(appliance_data_range_start)
        disagg_output_object['appliance_profile'][int(billcycle_start)]['profileList'][0]['dataRange']['end'] = \
            int(appliance_data_range_end)

    return disagg_output_object
