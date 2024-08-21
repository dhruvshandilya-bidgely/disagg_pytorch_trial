"""
Author - Mayank Sharan
Date - 15/10/18
initialisations disagg output object initiates single instance of disagg output object
"""

# Import python packages

from copy import deepcopy

# Import functions from within the project

from python3.initialisation.object_initialisations.init_empty_lifestyle_profile import init_empty_lifestyle_profile


def prepare_analytics_output_object(analytics_input_object, pipeline_output_object):

    """
    Parameters:
        analytics_input_object (dict)              : Dictionary containing all parameters needed to initialize the object
        pipeline_output_object (dict)              : Contains pipeline_output_object

    Returns:
        disagg_output_object(dict)              : Dictionary containing all parameters to run the pipeline
    """

    disagg_output_object = pipeline_output_object.get('disagg_output_object')
    temp = deepcopy(disagg_output_object)
    item_temp_object = pipeline_output_object.get('item_output_object')
    behavioural_profile = deepcopy(pipeline_output_object.get('item_output_object').get('profile_attributes'))

    disagg_output_object = dict({
        'bill_cycle_estimate': temp.get('bill_cycle_estimate'),
        'disagg_metrics': temp.get('disagg_metrics'),
        'created_hsm': temp.get('created_hsm'),
        'epoch_estimate': temp.get('epoch_estimate'),
        'output_write_idx_map': temp.get('output_write_idx_map'),
        'special_outputs': temp.get('special_outputs'),
        'api_output': deepcopy(pipeline_output_object.get('api_output')),
        'hvac_debug': temp.get('hvac_debug'),
        'created_hsm_analytics': dict({}),
        'appliance_profile': item_temp_object.get('appliance_profile'),
        'behavioural_profile': behavioural_profile
    })

    out_bill_cycles = analytics_input_object.get('out_bill_cycles')

    # initiate lifestyle profile object for all out bill cycles

    lifestyle_data_range_start = analytics_input_object.get('out_bill_cycles_by_module').get('lifestyle_profile')[0][0]
    lifestyle_data_range_end = analytics_input_object.get('out_bill_cycles_by_module').get('lifestyle_profile')[-1][-1]

    disagg_output_object['lifestyle_profile'] = dict()

    for billcycle_start, billcycle_end in out_bill_cycles:
        disagg_output_object['lifestyle_profile'][int(billcycle_start)] = init_empty_lifestyle_profile()

        # set start and end dates for each out bill cycles
        disagg_output_object['lifestyle_profile'][int(billcycle_start)]['profileList'][0]['start'] = \
            int(billcycle_start)
        disagg_output_object['lifestyle_profile'][int(billcycle_start)]['profileList'][0]['end'] = int(billcycle_end)

        # set start and end for data range for each out bill cycles
        disagg_output_object['lifestyle_profile'][int(billcycle_start)]['profileList'][0]['dataRange']['start'] = \
            int(lifestyle_data_range_start)
        disagg_output_object['lifestyle_profile'][int(billcycle_start)]['profileList'][0]['dataRange']['end'] = \
            int(lifestyle_data_range_end)

    return disagg_output_object
