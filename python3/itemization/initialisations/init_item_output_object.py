"""
Author - Sahana M
Date - 30/08/2021
initialisations itemization output object initiates single instance of itemization output object
"""

# Import python packages
import numpy as np
from copy import deepcopy

# Import packages from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.itemization.initialisations.init_empty_appliance_profile import init_empty_appliance_profile


def init_item_aer_output_object(pipeline_input_object, pipeline_output_object, item_input_object):
    """
    This function is used to create the itemization output object
    Parameters:
        pipeline_input_object           (dict)          : Contains all the input required for the pipeline
        pipeline_output_object          (dict)          : Contains all the output object
        item_input_object               (dict)          : Itemization input object
    Returns:
        item_input_object               (dict)          : Itemization input object
    """

    item_output_object = dict({
        'bill_cycle_estimate': None,
        'itemization_metrics': {},
        'created_hsm': {},
        'epoch_estimate': None,
        'output_write_idx_map': {},
        'special_outputs': {},
        'run_time': np.zeros(40),
        'disagg_metrics': {},
    })

    # Initialise a Itemization metric key to keep track of errors

    item_output_object['itemization_metrics'] = {
        'itemization_pipeline' : {
            'time': 0,
            'exit_status': {
                'exit_code': 0,
                'error_list': [],
                'itemization_pipeline_status': True
    }}}

    # If disagg outputs are not present then create the output keys from scratch

    if not item_input_object.get('disagg_data_present'):

        module_seq = item_input_object.get('config').get('disagg_aer_module_seq')
        num_modules = len(module_seq)

        # Allow extra column for HVAC. Allow 6 extra columns for SMB

        num_columns = num_modules + 1 + int('hvac' in module_seq)

        # Initialize billing cycle estimate

        out_bill_cycles = item_input_object.get('out_bill_cycles')
        bill_cycle_estimate = np.full(shape=(out_bill_cycles.shape[0], num_columns), fill_value=np.nan)
        bill_cycle_estimate[:, 0] = out_bill_cycles[:, 0]
        item_output_object['bill_cycle_estimate'] = bill_cycle_estimate

        # Initialize epoch estimate

        input_ts = item_input_object['input_data'][:, Cgbdisagg.INPUT_EPOCH_IDX]
        epoch_estimate = np.full(shape=(len(input_ts), num_columns), fill_value=np.nan)
        epoch_estimate[:, 0] = input_ts
        item_output_object['epoch_estimate'] = epoch_estimate

        # Initialize output write idx map, disagg metrics and created hsm

        write_idx = 1

        for module_name in module_seq:
            if module_name == 'hvac' or module_name == 'va':
                item_output_object['output_write_idx_map'][module_name] = [write_idx, write_idx + 1]
                item_output_object['itemization_metrics'][module_name] = None
                item_output_object['created_hsm'][module_name] = None
                write_idx += 2
            else:
                item_output_object['output_write_idx_map'][module_name] = write_idx
                item_output_object['itemization_metrics'][module_name] = None
                item_output_object['created_hsm'][module_name] = None
                write_idx += 1

        item_output_object['itemization_metrics']['pipeline'] = None

        # initiate appliance profile object for all out bill cycles

        item_output_object['appliance_profile'] = dict()

        # Get default dataRange for this gbdisagg event

        appliance_data_range_start = item_input_object.get('out_bill_cycles_by_module').get('app_profile')[0][0]
        appliance_data_range_end = item_input_object.get('out_bill_cycles_by_module').get('app_profile')[-1][-1]

        for billcycle_start, billcycle_end in out_bill_cycles:
            item_output_object['appliance_profile'][int(billcycle_start)] = init_empty_appliance_profile()

            # set start and end dates for each out bill cycles

            item_output_object['appliance_profile'][int(billcycle_start)]['profileList'][0]['start'] = int(
                billcycle_start)
            item_output_object['appliance_profile'][int(billcycle_start)]['profileList'][0]['end'] = int(billcycle_end)

            # set start and end for data range for each out bill cycles

            item_output_object['appliance_profile'][int(billcycle_start)]['profileList'][0]['dataRange']['start'] = \
                int(appliance_data_range_start)
            item_output_object['appliance_profile'][int(billcycle_start)]['profileList'][0]['dataRange']['end'] = \
                int(appliance_data_range_end)

    else:

        item_output_object['bill_cycle_estimate'] = deepcopy(item_input_object.get('disagg_bill_cycle_estimate'))
        item_output_object['epoch_estimate'] = deepcopy(item_input_object.get('disagg_epoch_estimate'))
        item_output_object['output_write_idx_map'] = deepcopy(item_input_object.get('disagg_output_write_idx_map'))

        item_output_object['epoch_estimate'] = np.hstack((item_output_object['epoch_estimate'], np.zeros((len(item_output_object['epoch_estimate']), 3))))
        item_output_object['bill_cycle_estimate'] = np.hstack((item_output_object['bill_cycle_estimate'], np.zeros((len(item_output_object['bill_cycle_estimate']), 3))))

        item_output_object['appliance_profile'] = deepcopy(item_input_object.get('disagg_appliance_profile'))
        item_output_object['special_outputs'] =  deepcopy(item_input_object.get('disagg_special_outputs'))
        item_output_object['created_hsm'] = deepcopy(item_input_object.get('created_hsm'))

    return item_output_object
