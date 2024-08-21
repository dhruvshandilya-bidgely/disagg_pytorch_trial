"""
Author - Mayank Sharan
Date - 19th Sep 2018
run gb pipeline is a wrapper to call all functions as needed to run the gb disaggregation pipeline
"""

# Import python packages
import logging
import traceback
from copy import deepcopy
from datetime import datetime

# Import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff
from python3.disaggregation.aes.init_aes_config import init_aes_config
from python3.utils.get_rej_error_code_list import get_rej_error_code_list
from python3.disaggregation.aes.wrappers.ao_smb_disagg_wrapper import ao_smb_disagg_wrapper
from python3.disaggregation.pre_aes_disagg_ops.pre_aes_disagg_ops import pre_aes_disagg_ops
from python3.disaggregation.aes.wrappers.hvac_smb_disagg_wrapper import hvac_smb_disagg_wrapper
from python3.disaggregation.aes.wrappers.work_hour_disagg_wrapper import work_hour_smb_disagg_wrapper
from python3.disaggregation.aes.wrappers.lighting_smb_disagg_wrapper import lighting_smb_disagg_wrapper
from python3.disaggregation.initialisations.init_aes_output_object import init_disagg_aes_output_object
from python3.disaggregation.prepare_results.prepare_disagg_aes_results import prepare_disagg_aes_results


def call_ao_smb(disagg_input_object, disagg_output_object, exit_status, logger_pipeline):
    """
    Parameters:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        disagg_output_object(dict)              : Dictionary containing all outputs
        exit_status         (dict)              : Contains the error code and error list information for the pipeline
        logger_pipeline     (logger)            : The logger to use here

    Returns:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        disagg_output_object(dict)              : Dictionary containing all outputs
        exit_status         (dict)              : Contains the error code and error list information for the pipeline
    """

    t_before_ao = 0

    # noinspection PyBroadException
    try:

        t_before_ao = datetime.now()
        disagg_input_object, disagg_output_object = ao_smb_disagg_wrapper(disagg_input_object, disagg_output_object)

    except Exception:

        t_after_ao = datetime.now()

        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger_pipeline.error('Something went wrong in ao smb | %s', error_str)

        # Set the default exit status for the appliance

        disagg_output_object['disagg_metrics']['ao_smb'] = {
            'time': get_time_diff(t_before_ao, t_after_ao),
            'exit_status': {
                'exit_code': -1,
                'error_list': [error_str]
            }
        }

        # Change the pipeline exit code

        exit_status['error_list'].append(error_str)

    return disagg_input_object, disagg_output_object, exit_status


def call_lighting_smb(disagg_input_object, disagg_output_object, exit_status, logger_pipeline):
    """
    Parameters:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        disagg_output_object(dict)              : Dictionary containing all outputs
        exit_status         (dict)              : Contains the error code and error list information for the pipeline
        logger_pipeline     (logger)            : The logger to use here

    Returns:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        disagg_output_object(dict)              : Dictionary containing all outputs
        exit_status         (dict)              : Contains the error code and error list information for the pipeline
    """

    t_before_lighting = 0

    # noinspection PyBroadException
    try:

        t_before_lighting = datetime.now()
        disagg_input_object, disagg_output_object = lighting_smb_disagg_wrapper(disagg_input_object,
                                                                                disagg_output_object)

    except Exception:

        t_after_lighting = datetime.now()

        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger_pipeline.error('Something went wrong in lighting smb | %s', error_str)

        # Set the default exit status for the appliance
        disagg_output_object['disagg_metrics']['lighting_smb'] = {
            'time': get_time_diff(t_before_lighting, t_after_lighting),
            'exit_status': {
                'exit_code': -1,
                'error_list': [error_str]
            }
        }

        # Change the pipeline exit code

        exit_status['error_list'].append(error_str)

    return disagg_input_object, disagg_output_object, exit_status


def call_work_hour_smb(disagg_input_object, disagg_output_object, exit_status, logger_pipeline):
    """
    Parameters:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        disagg_output_object(dict)              : Dictionary containing all outputs
        exit_status         (dict)              : Contains the error code and error list information for the pipeline
        logger_pipeline     (logger)            : The logger to use here

    Returns:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        disagg_output_object(dict)              : Dictionary containing all outputs
        exit_status         (dict)              : Contains the error code and error list information for the pipeline
    """

    t_before_work_hour = 0

    # noinspection PyBroadException
    try:

        t_before_work_hour = datetime.now()
        disagg_input_object, disagg_output_object = work_hour_smb_disagg_wrapper(disagg_input_object,
                                                                                 disagg_output_object)

    except Exception:

        t_after_work_hour = datetime.now()

        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger_pipeline.error('Something went wrong in work_hour smb | %s', error_str)

        # Set the default exit status for the appliance
        disagg_output_object['disagg_metrics']['work_hour_smb'] = {
            'time': get_time_diff(t_before_work_hour, t_after_work_hour),
            'exit_status': {
                'exit_code': -1,
                'error_list': [error_str]
            }
        }

        # Change the pipeline exit code

        exit_status['error_list'].append(error_str)

    return disagg_input_object, disagg_output_object, exit_status


def call_hvac_smb(disagg_input_object, disagg_output_object, exit_status, logger_pipeline):
    """
    Parameters:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        disagg_output_object(dict)              : Dictionary containing all outputs
        exit_status         (dict)              : Contains the error code and error list information for the pipeline
        logger_pipeline     (logger)            : The logger to use here

    Returns:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        disagg_output_object(dict)              : Dictionary containing all outputs
        exit_status         (dict)              : Contains the error code and error list information for the pipeline
    """

    t_before_hvac = 0

    # noinspection PyBroadException
    try:

        t_before_hvac = datetime.now()
        disagg_input_object, disagg_output_object = hvac_smb_disagg_wrapper(disagg_input_object, disagg_output_object)

    except Exception:

        t_after_hvac = datetime.now()

        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger_pipeline.error('Something went wrong in hvac smb | %s', error_str)

        # Set the default exit status for the appliance
        disagg_output_object['disagg_metrics']['hvac_smb'] = {
            'time': get_time_diff(t_before_hvac, t_after_hvac),
            'exit_status': {
                'exit_code': -1,
                'error_list': [error_str]
            }
        }

        # Change the pipeline exit code

        exit_status['error_list'].append(error_str)

    return disagg_input_object, disagg_output_object, exit_status


def call_disagg_aes_modules(disagg_input_object, disagg_output_object, exit_status, logger_pipeline, module_seq):
    """
    Parameters:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        disagg_output_object(dict)              : Dictionary containing all outputs
        logger_pipeline     (logger)            : The logger to use for the pipeline
        exit_status         (dict)              : Contains the exit code and errors of the run so far
        module_seq          (list)              : Sequence to execute the modules in

    Returns:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        disagg_output_object(dict)              : Dictionary containing all outputs
        exit_status         (dict)              : Contains the error code and error list information for the pipeline
    """

    for module_code in module_seq:

        if module_code == 'ao_smb':
            disagg_input_object, disagg_output_object, exit_status = \
                call_ao_smb(disagg_input_object, disagg_output_object, exit_status, logger_pipeline)

        elif module_code == 'li_smb':
            disagg_input_object, disagg_output_object, exit_status = \
                call_lighting_smb(disagg_input_object, disagg_output_object, exit_status, logger_pipeline)

        elif module_code == 'work_hours':
            disagg_input_object, disagg_output_object, exit_status = \
                call_work_hour_smb(disagg_input_object, disagg_output_object, exit_status, logger_pipeline)

        elif module_code == 'hvac_smb':
            disagg_input_object, disagg_output_object, exit_status = \
                call_hvac_smb(disagg_input_object, disagg_output_object, exit_status, logger_pipeline)

        else:
            logger_pipeline.warning('Unrecognized module code %s |', module_code)

    return disagg_input_object, disagg_output_object, exit_status


def init_aes_output_object(disagg_input_object):
    """
    Initialises all the disagg custom input keys from the pipeline input object
    Parameters:
        disagg_input_object            (dict)           : Contains all the disagg input keys

    Returns:
        disagg_output_object            (dict)          : Contains all the output required for disagg
    """

    # First initialise all the common keys

    disagg_output_object = init_disagg_aes_output_object(disagg_input_object)

    return disagg_output_object


def init_aes_input_object(pipeline_input_object):
    """
    Initialises all the disagg custom input keys from the pipeline input object
    Parameters:
        pipeline_input_object          (dict)           : Contains all the input required for the pipeline

    Returns:
        disagg_input_object             (dict)          : Contains all the input required for disagg
    """

    # First initialise all the common keys

    temp_object = deepcopy(pipeline_input_object)

    disagg_input_object = {'appliances_hsm': temp_object.get('appliances_hsm'),
                           'app_profile': temp_object.get('app_profile'),
                           'data_quality_metrics': temp_object.get('data_quality_metrics'),
                           'gb_pipeline_event': temp_object.get('gb_pipeline_event'),
                           'home_meta_data': temp_object.get('home_meta_data'),
                           'input_data': temp_object.get('input_data'),
                           'input_data_with_neg_and_nan': temp_object.get('input_data_with_neg_and_nan'),
                           'loaded_files': temp_object.get('loaded_files'),
                           'logging_dict': temp_object.get('logging_dict'),
                           'original_input_data': temp_object.get('original_input_data'),
                           'input_data_without_outlier_removal': temp_object.get('input_data_without_outlier_removal'),
                           'out_bill_cycles': temp_object.get('out_bill_cycles'),
                           'out_bill_cycles_by_module': temp_object.get('out_bill_cycles_by_module'),
                           'config': init_aes_config(temp_object),
                           'store_tb_in_cassandra': temp_object.get('store_tb_in_cassandra')
                           }

    # Initialise disagg config from global config

    return disagg_input_object


def run_aes_pipeline(pipeline_input_object, pipeline_output_object):
    """
    Contains all the operations related to the AMI electric residential disaggregation
    Parameters:
        pipeline_input_object           (dict)          : Contains all the input required for the pipeline
        pipeline_output_object          (dict)          : Contains all the output object

    Returns:
        pipeline_input_object           (dict)          : Contains all the input required for the pipeline
        pipeline_output_object          (dict)          : Contains all the output obtained in this pipeline
    """

    # Initialize pipeline logger

    logger = pipeline_input_object.get('logger').getChild('run_aes_pipeline')
    pipeline_input_object.pop('logger')
    logger_base = logging.LoggerAdapter(logger, pipeline_input_object.get('logging_dict'))

    # Initialise required variables

    exit_status = {
        'exit_code': 1,
        'error_codes': [],
        'runtime': None,
        'error_list': [],
    }

    # ----------------------------------------------- INITIALISATION --------------------------------------------------

    t_init_start = datetime.now()
    disagg_input_object = init_aes_input_object(pipeline_input_object)
    disagg_output_object = init_aes_output_object(disagg_input_object)
    disagg_input_object['logger'] = logger
    pipeline_input_object['logger'] = logger
    t_init_end = datetime.now()

    logger_base.info('Disaggregation AES initialisation took | %.3fs ', get_time_diff(t_init_start, t_init_end))

    data_quality_dict = disagg_input_object.get('data_quality_metrics').get('disagg_data_quality')
    run_pipeline = data_quality_dict.get('run_pipeline')

    t_before_pipeline = datetime.now()

    if run_pipeline:

        # ----------------------------------------------- PRE DISAGG OPS ---------------------------------------------

        t_pre_disagg_ops_start = datetime.now()
        disagg_input_object = pre_aes_disagg_ops(disagg_input_object)
        t_pre_disagg_ops_end = datetime.now()

        logger_base.info('Disaggregation AES Pre disagg operations took | %.3fs ', get_time_diff(t_pre_disagg_ops_start,
                                                                                                 t_pre_disagg_ops_end))

        # ----------------------------------------------- RUN APPLIANCE MODULE -----------------------------------------

        # Extract the module sequence to be run
        module_seq = disagg_input_object.get('config').get('module_seq')

        t_pipeline_start = datetime.now()
        disagg_input_object, disagg_output_object, exit_status = \
            call_disagg_aes_modules(disagg_input_object, disagg_output_object, exit_status, logger_base, module_seq)
        t_pipeline_end = datetime.now()

        logger_base.info('Disaggregation AES appliance modules took | %.3fs ', get_time_diff(t_pipeline_start,
                                                                                             t_pipeline_end))
    else:

        logger_base.info('Not running disaggregation pipeline due to bad data quality |')

        exit_status['exit_code'] = -2
        exit_status['error_codes'] = get_rej_error_code_list(data_quality_dict.get('rejection_reasons'))

    # ----------------------------------------------- PREPARE RESULTS -------------------------------------------------

    t_after_pipeline = datetime.now()
    pipeline_runtime = get_time_diff(t_before_pipeline, t_after_pipeline)
    logger_base.info('Disaggregation AES Pipeline ran in | %.3f s', pipeline_runtime)

    exit_status['runtime'] = pipeline_runtime
    disagg_output_object['disagg_metrics']['aes_pipeline'] = exit_status

    t_start = datetime.now()
    api_aes_disagg_output = prepare_disagg_aes_results(disagg_input_object, disagg_output_object)
    t_end = datetime.now()

    logger_base.info('Disaggregation AES results preparation took | %.3fs ', get_time_diff(t_start, t_end))

    # ----------------------------------------------- END OF AES DISAGG -----------------------------------------------

    # Combine the api_aer_disagg_output in pipeline_output_object

    pipeline_output_object['disagg_output_object'] = disagg_output_object
    pipeline_output_object['api_output'] = api_aes_disagg_output
    pipeline_output_object['exit_status'] = exit_status

    # Update the created hsm key in the pipeline output object if it is Historical or Incremental mode

    if pipeline_input_object.get('global_config').get('disagg_mode') == 'historical' or \
            pipeline_input_object.get('global_config').get('disagg_mode') == 'incremental':
        pipeline_output_object.update({
            "created_hsm": disagg_output_object.get('created_hsm')
        })

    return pipeline_input_object, pipeline_output_object
