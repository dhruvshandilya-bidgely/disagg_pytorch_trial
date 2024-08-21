"""
Author - Sahana M
Date - 24th March 2021
preprocess director is a module which control and directs all the preprocessing related functions
"""

# Import python packages

# Import functions from within the project
from python3.config.pipeline_constants import GlobalConfigParams
from python3.utils.update_pipeline_input_object import update_pipeline_input_object
from python3.master_pipeline.preprocessing.preprocess_ami_data import preprocess_ami_electric


def preprocess_director(pipeline_input_object, pipeline_output_objects):
    """
    This function does preprocessing and sanity check on each pipeline_input_object
    Parameters:
        pipeline_input_object    (dict)             : Contains all inputs required to run the pipeline
        pipeline_output_objects  (list)             : List of dictionary containing all outputs

    Returns:
        pipeline_input_object   (dict)              : Contains all inputs required to run the pipeline
    """

    # Initialize pipeline logger

    logger_pipeline_base = pipeline_input_object['logger'].getChild('preprocess_director')
    pipeline_input_object['logger'] = logger_pipeline_base

    # Stage 1 - Update HSM

    pipeline_input_object = update_pipeline_input_object(pipeline_input_object, pipeline_output_objects)

    config = pipeline_input_object.get('global_config')
    data_type = config.get('data_type')
    fuel_type = config.get('fuel_type')

    if fuel_type == GlobalConfigParams.fuel_type.get('electric') and \
            data_type == GlobalConfigParams.data_type.get('ami'):

        # -------------------------------------- RUN DISAGG AMI PREPROCESSING ---------------------------------

        # Call the preprocessing module specific for Disagg

        pipeline_input_object = preprocess_ami_electric(pipeline_input_object)

    return pipeline_input_object
