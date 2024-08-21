"""
Author - Sahana M
Date - 23rd Mar 2021
This module contains hsm preparation and posting functions
"""

# Import python packages
import logging
from datetime import datetime

# Import functions form within the project
from python3.utils.time.get_time_diff import get_time_diff
from python3.master_pipeline.hsm_preparation.post_hsm import post_hsm
from python3.master_pipeline.post_pipeline_ops.dump_tou_file import dump_tou_file
from python3.master_pipeline.hsm_preparation.prepare_hsm_output import prepare_hsm_output


def hsm_preparation(pipeline_input_object, pipeline_output_object):
    """
    This modules directs the pipelines to be run in a predefined sequence
    Parameters:
        pipeline_input_object           (dict)      : Contains all the input required in for the pipeline
        pipeline_output_object          (dict)      : Contains all the output required for the pipeline
    Returns:

    """

    # Initialize pipeline logger

    logger_base = pipeline_input_object.get('logger').getChild('hsm_preparation')
    pipeline_input_object['logger'] = logger_base
    logger_pipeline = logging.LoggerAdapter(logger_base, pipeline_input_object.get('logging_dict'))

    # Initialize logger pass to send to functions

    logger_pass = {
        'logger_base': logger_base,
        'logging_dict': pipeline_input_object.get('logging_dict'),
    }

    # Do not update write results to True in local runs
    write_results = pipeline_input_object.get('global_config', {}).get('write_results', {}) and \
        pipeline_input_object.get('data_quality_metrics', {}).get('disagg_data_quality', {}).get('run_pipeline', False)

    # Write HSM only if the pipeline has been run and if run mode is historical or incremental
    if write_results and pipeline_input_object.get('global_config').get('disagg_mode') in ['historical', 'incremental']:

        t_before_hsm_write = datetime.now()
        hsm_write_output = prepare_hsm_output(pipeline_output_object, logger_pass)
        post_success, status_code = post_hsm(pipeline_input_object, hsm_write_output, logger_pass)
        t_after_hsm_write = datetime.now()

        logger_pipeline.info('HSM write took | %.3f s', get_time_diff(t_before_hsm_write, t_after_hsm_write))

        if post_success:
            logger_pipeline.info('HSM write successful status code | %d', status_code)
        else:
            logger_pipeline.info('HSM write failed status code | %d', status_code)

    # Dump TOU output as CSV

    dump_tou_file(pipeline_input_object, pipeline_output_object, logger_pipeline)

    return
