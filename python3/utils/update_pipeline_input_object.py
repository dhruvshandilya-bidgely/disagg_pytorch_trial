"""
Author - Mayank Sharan
Date - 28/1/19
queue mode runs the pipeline on servers or local machine by reading sqs queue messages and using multiprocessing
"""

# Import functions from within the project
import numpy as np
from copy import deepcopy

from python3.config.Cgbdisagg import Cgbdisagg


def get_hsm_in_right_format(hsm_dict):
    """
    Code to check if hsm attributes are in proper format and if not convert it

    Parameters:
        hsm_dict(dict):     Input hsm dict

    Return:
        hsm_dict(dict):     Updated hsm dict
    """
    hsm_attributes = deepcopy(hsm_dict.get('attributes'))
    for attribute, value in hsm_attributes.items():
        if not (isinstance(value, list) or isinstance(value, np.ndarray)):
            hsm_attributes[attribute] = [value]

    hsm_dict['attributes'] = hsm_attributes
    return hsm_dict


def update_hsm(pipeline_input_object, pipeline_output_object):

    """
    Parameters:
        pipeline_input_object (dict)              : Contains all inputs required to run the pipeline
        pipeline_output_object(dict)              : Dictionary containing all outputs

    Returns:
        pipeline_input_object (dict)              : Contains all inputs required to run the pipeline
    """

    # Extract end time and hsm dictionaries

    end_time = pipeline_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX]
    old_hsm_dict = pipeline_input_object.get('appliances_hsm')
    new_hsm_dict = pipeline_output_object.get('created_hsm')

    updated_hsm_dict = {}

    # For all appliances update hsm if a new one is created nd it falls in the region needed

    appliances_list = list(old_hsm_dict.keys())

    for app_name in appliances_list:
        new_hsm = new_hsm_dict.get(app_name)
        old_hsm = old_hsm_dict.get(app_name)

        if new_hsm is None:
            updated_hsm_dict[app_name] = old_hsm
            continue

        new_hsm_time = new_hsm.get('timestamp')

        if old_hsm is None or len(old_hsm) == 0:
            if new_hsm_time < end_time:
                updated_hsm_dict[app_name] = get_hsm_in_right_format(new_hsm)
                continue
            else:
                updated_hsm_dict[app_name] = None
                continue

        old_hsm_time = old_hsm.get('timestamp')

        if end_time > new_hsm_time >= old_hsm_time:
            updated_hsm_dict[app_name] = get_hsm_in_right_format(new_hsm)
        else:
            updated_hsm_dict[app_name] = old_hsm

    pipeline_input_object['appliances_hsm'] = updated_hsm_dict

    return pipeline_input_object


def update_pipeline_input_object(pipeline_input_object, pipeline_output_objects):

    """
    Parameters:
        pipeline_input_object (dict)                 : Contains all inputs required to run the pipeline
        pipeline_output_objects  (list)              : List of dictionary containing all outputs

    Returns:
        pipeline_input_object (dict)              : Contains all inputs required to run the pipeline
    """

    # Update HSM in the disagg input object using previous disagg output objects

    for pipeline_output_object in pipeline_output_objects:
        pipeline_input_object = update_hsm(pipeline_input_object, pipeline_output_object)

    return pipeline_input_object

