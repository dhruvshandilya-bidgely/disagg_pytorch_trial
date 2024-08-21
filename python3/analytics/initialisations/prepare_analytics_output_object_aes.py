"""
Author - Mayank Sharan
Date - 15/10/18
initialisations disagg output object initiates single instance of disagg output object
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
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

    disagg_output_object = dict({
        'bill_cycle_estimate': temp.get('bill_cycle_estimate'),
        'disagg_metrics': temp.get('disagg_metrics'),
        'created_hsm': temp.get('created_hsm'),
        'epoch_estimate': temp.get('epoch_estimate'),
        'output_write_idx_map': temp.get('output_write_idx_map'),
        'special_outputs': temp.get('special_outputs'),
        'api_output': deepcopy(pipeline_output_object.get('api_output')),
        'appliance_profile': temp.get('appliance_profile')
    })

    disagg_output_object['lifestyle_profile'] = temp.get('lifestyle_profile')

    return disagg_output_object
