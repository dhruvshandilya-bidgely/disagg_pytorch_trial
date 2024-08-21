"""
Author - Sahana M
Date - 23rd Mar 2021
run event director manages the flow of disaggregation & itemization modules
"""

# Import python packages
import logging
import traceback
from datetime import datetime

# Import functions from within the project
from python3.utils.time.get_time_diff import get_time_diff
from python3.disaggregation.aer.aer_director import run_aer_pipeline
from python3.disaggregation.aes.aes_director import run_aes_pipeline
from python3.analytics.analytics_aer_director import run_analytics_pipeline_aer
from python3.analytics.analytics_aes_director import run_analytics_pipeline_aes
from python3.itemization.itemization_director import run_itemization_pipeline


def run_itemization(pipeline_input_object, pipeline_output_object):

    """
    This module runs the Itemization module

    Parameters:
        pipeline_input_object           (dict)          : Contains all the input required for the pipeline
        pipeline_output_object          (dict)          : Contains all the output object

    Returns:
        pipeline_input_object           (dict)          : Contains all the input required for the pipeline
        pipeline_output_object          (dict)          : Contains all the output object
    """

    # Initialise logger

    logger_base = pipeline_input_object.get('logger').getChild('run_itemization')
    pipeline_input_object['logger'] = logger_base
    logger = logging.LoggerAdapter(logger_base, pipeline_input_object.get('logging_dict'))

    # Call the Residential/ SMB pipeline based on the customer type key in global config

    if pipeline_input_object.get('global_config').get('user_type') != 'smb':
        t_start = datetime.now()
        pipeline_input_object, pipeline_output_object = \
            run_itemization_pipeline(pipeline_input_object, pipeline_output_object)
        t_end = datetime.now()

        logger.info('Residential AMI Itemization ran in | %.3fs ', get_time_diff(t_start, t_end))

    return pipeline_input_object, pipeline_output_object


def run_analytics(pipeline_input_object, pipeline_output_object):

    """
    This module runs the User profile module

    Parameters:
        pipeline_input_object           (dict)          : Contains all the input required for the pipeline
        pipeline_output_object          (dict)          : Contains all the output object

    Returns:
        pipeline_input_object           (dict)          : Contains all the input required for the pipeline
        pipeline_output_object          (dict)          : Contains all the output object
    """

    # Initialise logger

    logger_base = pipeline_input_object.get('logger').getChild('run_analytics')
    pipeline_input_object['logger'] = logger_base
    logger = logging.LoggerAdapter(logger_base, pipeline_input_object.get('logging_dict'))

    data_quality_dict = pipeline_input_object.get('data_quality_metrics').get('disagg_data_quality')
    run_pipeline = data_quality_dict.get('run_pipeline')

    if run_pipeline:
        t_start = datetime.now()

        if 'smb' in pipeline_input_object.get('global_config').get('user_type'):
            pipeline_input_object, pipeline_output_object =\
                run_analytics_pipeline_aes(pipeline_input_object, pipeline_output_object)
        else:
            pipeline_input_object, pipeline_output_object = \
                run_analytics_pipeline_aer(pipeline_input_object, pipeline_output_object)

        t_end = datetime.now()

        logger.info('Analytics Module ran in | %.3f s ', get_time_diff(t_start, t_end))
    else:

        logger.info('Not running analytics pipeline due to bad data quality |')

    return pipeline_input_object, pipeline_output_object


def run_disaggregation(pipeline_input_object, pipeline_output_object):

    """
    This module runs the disaggregation module based on the type of customer - Ex: Residential / SMB

    Parameters:
        pipeline_input_object           (dict)          : Contains all the input required for the pipeline
        pipeline_output_object          (dict)          : Contains all the output object

    Returns:
        pipeline_input_object           (dict)          : Contains all the input required for the pipeline
        pipeline_output_object          (dict)          : Contains all the output object
    """

    # Initialise logger

    logger_base = pipeline_input_object.get('logger').getChild('run_disaggregation')
    pipeline_input_object['logger'] = logger_base
    logger = logging.LoggerAdapter(logger_base, pipeline_input_object.get('logging_dict'))

    # Full forms
    # ami - advanced metering infra
    # aer - ami electric residential
    # aes - ami electric smb

    # Call the Residential/ SMB pipeline based on the customer type key in global config

    if 'res' in pipeline_input_object.get('global_config').get('user_type'):

        t_start = datetime.now()
        pipeline_input_object, pipeline_output_object =\
            run_aer_pipeline(pipeline_input_object, pipeline_output_object)
        t_end = datetime.now()

        logger.info('Residential AMI Disaggregation ran in | %.3fs ', get_time_diff(t_start, t_end))

    pipeline_input_object['logger'] = logger_base

    if 'smb' in pipeline_input_object.get('global_config').get('user_type'):

        t_start = datetime.now()
        pipeline_input_object, pipeline_output_object =\
            run_aes_pipeline(pipeline_input_object, pipeline_output_object)
        t_end = datetime.now()

        logger.info('SMB AMI Disaggregation ran in | %.3fs ', get_time_diff(t_start, t_end))

    return pipeline_input_object, pipeline_output_object


def run_pipelines_director(pipeline_input_object, pipeline_output_object):

    """
    This modules directs the pipelines to be run in a predefined sequence

    Parameters:
        pipeline_input_object           (dict)      : Contains all the input required in for the pipeline
        pipeline_output_object          (dict)      : Contains all the output required for the pipeline

    Returns:
        pipeline_input_object           (dict)      : Contains all inputs required to run the pipeline
        pipeline_output_object          (dict)      : Contains the output object required for the pipeline
    """

    # Initialize pipeline logger

    logger_base = pipeline_input_object.get('logger').getChild('run_pipelines_director')
    pipeline_input_object['logger'] = logger_base
    logger = logging.LoggerAdapter(logger_base, pipeline_input_object.get('logging_dict'))

    # -------------------------------------------- RUN PIPELINES IN SEQUENCE ------------------------------------------

    # Extract the pipeline sequence

    pipeline_seq = pipeline_input_object.get('global_config').get('pipeline_seq')

    pipeline_output_object['exit_status'] = {}

    for pipeline in pipeline_seq:

        # Run Disaggregation pipeline

        if pipeline == 'disagg':

            pipeline_output_object['exit_status']['disagg_pipeline_status'] = True

            try:
                pipeline_input_object, pipeline_output_object = \
                    run_disaggregation(pipeline_input_object, pipeline_output_object)

            except Exception:

                pipeline_output_object['exit_status']['disagg_pipeline_status'] = False
                error_str = (traceback.format_exc()).replace('\n', ' ')
                logger.error('Something went wrong in the Disaggregation pipeline | %s', error_str)

        # Run Analytics pipeline

        elif pipeline == 'analytics':

            pipeline_output_object['exit_status']['analytics_pipeline_status'] = True

            try:
                pipeline_input_object, pipeline_output_object = \
                    run_analytics(pipeline_input_object, pipeline_output_object)

            except Exception:

                pipeline_output_object['exit_status']['analytics_pipeline_status'] = False
                error_str = (traceback.format_exc()).replace('\n', ' ')
                logger.error('Something went wrong in the Analytics pipeline | %s', error_str)

        # Run Itemization pipeline

        elif pipeline == 'itemization':

            pipeline_output_object['exit_status']['itemization_pipeline_status'] = True

            try:
                pipeline_input_object, pipeline_output_object = \
                    run_itemization(pipeline_input_object, pipeline_output_object)

            except Exception:

                pipeline_output_object['exit_status']['itemization_pipeline_status'] = False
                error_str = (traceback.format_exc()).replace('\n', ' ')
                logger.error('Something went wrong in the Itemization pipeline | %s', error_str)

        pipeline_input_object['logger'] = logger_base

    return pipeline_input_object, pipeline_output_object
