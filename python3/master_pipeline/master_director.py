"""
Author - Mayank Sharan
Date - 19th Sep 2018
Master director runs the complete pipeline for 1 user
"""

# Import python packages

import logging
import warnings
import traceback

from datetime import datetime

# Import packages from within the project

from python3.utils.time.get_time_diff import get_time_diff
from python3.initialisation.init_director import init_director
from python3.master_pipeline.run_pipelines_director import run_pipelines_director
from python3.master_pipeline.preprocessing.preprocess_director import preprocess_director
from python3.master_pipeline.hsm_preparation.hsm_preparation_director import hsm_preparation
from python3.master_pipeline.post_pipeline_ops.post_pipeline_results import post_pipeline_output

# To suppress warnings being printed in the console
warnings.simplefilter('ignore')


def master_director(fetch_params, run_params_dict, logger_pass):

    """
    Master director runs the complete pipeline for a given user

    Parameters:
        fetch_params            (dict)              : Contains uuid, t_start, t_end and disagg_mode
        run_params_dict         (dict)              : Dictionary with all custom run parameters provided
        logger_pass             (dict)              :

    Returns:
        delete_message          (bool)              : Boolean indicating if the processing was as expected
    """

    # Initialize the logging dictionary for this run

    logger_master = logger_pass.get('logger')
    logging_dict = logger_pass.get('logging_dict')
    logger_master_adapter = logging.LoggerAdapter(logger_master, logging_dict)

    # Extract basic parameters for the run

    uuid = fetch_params.get('uuid')
    api_env = fetch_params.get('api_env')

    # ----------------------------------------- INITIALISATIONS -------------------------------------------------------

    t_init_start = datetime.now()
    pipeline_input_objects, delete_message = init_director(fetch_params, run_params_dict, logger_master, logging_dict)
    t_init_end = datetime.now()

    if pipeline_input_objects is None:
        logger_master_adapter.info('User processing time combined is %.3f s |', get_time_diff(t_init_start, t_init_end))
        return delete_message

    # Run the pipeline

    pipeline_output = []
    pipeline_output_objects = []

    # Run the pipeline for each disagg event

    t_before_user_process = datetime.now()

    for pipeline_input_object in pipeline_input_objects:

        # noinspection PyBroadException
        try:

            # -------------------------------------- PREPROCESS INPUT PIPELINE ----------------------------------------

            t_preprocess_start = datetime.now()

            pipeline_input_object['logger'] = logger_master
            pipeline_input_object = preprocess_director(pipeline_input_object, pipeline_output_objects)

            t_preprocess_end = datetime.now()

            logger_master_adapter.info('Preprocessing ran in | %.3f s', get_time_diff(t_preprocess_start,
                                                                                      t_preprocess_end))

            # -------------------------------------- RUN PIPELINE -----------------------------------------------------

            t_before_pipeline = datetime.now()
            pipeline_output_object = {}
            pipeline_input_object['logger'] = logger_master
            pipeline_input_object, pipeline_output_object = \
                run_pipelines_director(pipeline_input_object, pipeline_output_object)

            t_after_pipeline = datetime.now()

            logger_master_adapter.info('Pipeline ran in | %.3f s', get_time_diff(t_before_pipeline, t_after_pipeline))

            # -------------------------------------- HSM PREPARATION & POSTING ----------------------------------------

            t_hsm_start = datetime.now()

            pipeline_input_object['logger'] = logger_master
            hsm_preparation(pipeline_input_object, pipeline_output_object)

            t_hsm_end = datetime.now()

            logger_master_adapter.info('HSM posting ran in | %.3f s', get_time_diff(t_hsm_start, t_hsm_end))

            # merging of api_output code here

            pipeline_output.append(pipeline_output_object.get('api_output'))
            pipeline_output_objects.append(pipeline_output_object)

        except Exception:

            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger_master_adapter.error('Pipeline run failed | %s', error_str)

    # ----------------------------------------- OUTPUT POSTING -----------------------------------------------------
    # Do not update write results to True in local runs
    if run_params_dict.get('write_results'):

        # noinspection PyBroadException
        try:
            t_before_pipeline_write = datetime.now()
            post_success, status_code = post_pipeline_output(uuid, api_env, pipeline_output, fetch_params, logger_master,
                                                             logging_dict)
            t_after_pipeline_write = datetime.now()

            logger_master_adapter.info('Pipeline output write took %.3f s |', get_time_diff(t_before_pipeline_write,
                                                                                            t_after_pipeline_write))

            if post_success:
                logger_master_adapter.info('Pipeline output write successful status code %d |', status_code)
            else:
                delete_message = False
                logger_master_adapter.info('Pipeline output write failed status code %d |', status_code)

        except Exception:
            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger_master_adapter.error('Something went wrong in posting output | %s', error_str)

    t_after_user_process = datetime.now()

    # Log overall run time
    logger_master_adapter.info('User processing time combined is %.3f s |', get_time_diff(t_before_user_process,
                                                                                          t_after_user_process))

    return delete_message
