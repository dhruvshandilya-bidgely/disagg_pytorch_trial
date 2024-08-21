"""
Author - Mayank Sharan
Date - 27/01/19
This function calls all operations we need to do after we run the disagg
"""

# Import python packages

import logging

# Import functions from within the project

from python3.disaggregation.post_disagg_ops.remove_ref_from_ao import remove_ref_from_ao
from python3.disaggregation.post_disagg_ops.adjust_for_multi_ref import adjust_for_multi_ref
from python3.disaggregation.post_disagg_ops.adjust_for_multi_ref import adjust_for_multi_ref_epoch


def aer_post_disagg_ops(disagg_input_object, disagg_output_object, pipeline_input_object):

    """
    Parameters:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        disagg_output_object(dict)              : Dictionary containing all outputs
        pipeline_input_object(dict)             : Dictionary containing pipeline inputs

    Returns:
        disagg_output_object(dict)              : Dictionary containing all outputs
        pipeline_input_object(dict)             : Dictionary containing pipeline inputs
    """

    # Initialize the logger

    logger_base = disagg_input_object.get('logger').getChild('post_disagg_ops')
    logger = logging.LoggerAdapter(logger_base, disagg_input_object.get('logging_dict'))

    # Remove "ref" from ao

    disagg_output_object = remove_ref_from_ao(disagg_input_object, disagg_output_object, logger)

    # Adjust ref for multi ref

    disagg_output_object = adjust_for_multi_ref(disagg_input_object, disagg_output_object, logger)
    disagg_output_object = adjust_for_multi_ref_epoch(disagg_input_object, disagg_output_object, logger)

    # If the solar module has run update the following :-
    # 'input_data' :- corrected input data (solar added)
    # 'input_data_without_solar' :- input data without solar (negative values capped at 0)

    if 'input_data_without_solar' in disagg_input_object.keys():
        pipeline_input_object['input_data'] = disagg_input_object.get('input_data')
        pipeline_input_object['input_data_without_solar'] = disagg_input_object.get('input_data_without_solar')

    return disagg_output_object, pipeline_input_object
