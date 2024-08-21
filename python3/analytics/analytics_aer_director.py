"""
Author - Mayank Sharan
Date - 19th Sep 2018
run gb pipeline is a wrapper to call all functions as needed to run the gb diasggregation pipeline
"""

# Import python packages
import logging
import traceback
from copy import deepcopy
from datetime import datetime

# Import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff
from python3.analytics.init_analytics_config import init_analytics_config
from python3.analytics.functions.pre_analytics_ops import pre_analytics_ops
from python3.analytics.wrappers.ev_propensity_wrapper import ev_propensity_wrapper
from python3.analytics.wrappers.lifestyle_profile_wrapper import lifestyle_profile_wrapper
from python3.analytics.wrappers.hvac_inefficiency_wrapper import hvac_inefficiency_wrapper
from python3.analytics.wrappers.solar_propensity_wrapper import solar_propensity_wrapper
from python3.analytics.functions.prepare_analytics_results import prepare_analytics_results
from python3.analytics.initialisations.prepare_analytics_output_object import prepare_analytics_output_object


def call_hvac_inefficiency(analytics_input_objects, analytics_output_objects, exit_status, logger_pipeline):
    """
    Parameters:
        analytics_input_objects     (dict)              : Contains all inputs required to run the pipeline
        analytics_output_objects    (dict)              : Dictionary containing all outputs
        exit_status                 (dict)              : Contains the error code and error list info for the pipeline
        logger_pipeline             (logger)            : The logger to use here

    Returns:
        analytics_input_objects (dict)              : Contains all inputs required to run the pipeline
        analytics_output_objects(dict)              : Dictionary containing all outputs
        exit_status         (dict)              : Contains the error code and error list information for the pipeline
    """

    t_before_hvac_ineff = datetime.now()

    # noinspection PyBroadException

    try:
        analytics_input_objects, analytics_output_objects =\
            hvac_inefficiency_wrapper(analytics_input_objects, analytics_output_objects)

    except Exception:
        t_after_hvac_ineff = datetime.now()

        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger_pipeline.error('Something went wrong in HVAC Inefficiency | %s', error_str)

        # Set the default exit status for the appliance

        analytics_output_objects['disagg_metrics']['hvac_inefficiency'] = {
            'time': get_time_diff(t_before_hvac_ineff, t_after_hvac_ineff),
            'exit_status': {
                'exit_code': -1,
                'error_list': [error_str]
            }
        }

        # Change the pipeline exit code

        exit_status['error_list'].append(error_str)

    return analytics_input_objects, analytics_output_objects, exit_status


def call_lifestyle(analytics_input_object, analytics_output_object, exit_status, logger_pipeline):
    """
    Parameters:
        analytics_input_object     (dict)              : Contains all inputs required to run the pipeline
        analytics_output_object    (dict)              : Dictionary containing all outputs
        exit_status                (dict)              : Contains the error code and error list information for the pipeline
        logger_pipeline            (logger)            : The logger to use here
    Returns:
        analytics_output_object    (dict)              : Dictionary containing all outputs
        exit_status                (dict)              : Contains the error code and error list information for the pipeline
    """

    t_before_lifestyle = 0

    # noinspection PyBroadException
    try:
        t_before_lifestyle = datetime.now()
        analytics_output_object = lifestyle_profile_wrapper(analytics_input_object, analytics_output_object)
    except Exception:
        t_after_lifestyle = datetime.now()

        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger_pipeline.error('Something went wrong in lifestyle | %s', error_str)

        # Set the default exit status for the appliance

        analytics_output_object['disagg_metrics']['lifestyle'] = {
            'time': get_time_diff(t_before_lifestyle, t_after_lifestyle),
            'exit_status': {
                'exit_code': -1,
                'error_list': [error_str]
            }
        }

        # Change the pipeline exit code

        exit_status['error_code'] = -1
        exit_status['error_list'].append(error_str)

    return analytics_output_object, exit_status


def call_ev_propensity(analytics_input_object, analytics_output_object, exit_status, logger_pipeline):
    """
    Parameters:
        analytics_input_object     (dict)          : Contains all inputs required to run the pipeline
        analytics_output_object    (dict)          : Dictionary containing all outputs
        exit_status                (dict)          : Contains the error code and error list information for the pipeline
        logger_pipeline            (logger)        : The logger to use here
    Returns:
        analytics_output_object    (dict)          : Dictionary containing all outputs
        exit_status                (dict)          : Contains the error code and error list information for the pipeline
    """

    t_before_ev_propensity = 0

    # noinspection PyBroadException
    try:
        t_before_ev_propensity = datetime.now()
        analytics_output_object = ev_propensity_wrapper(analytics_input_object, analytics_output_object)
    except Exception:
        t_after_ev_propensity = datetime.now()

        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger_pipeline.error('Something went wrong in EV propensity | %s', error_str)

        # Set the default exit status for the appliance

        analytics_output_object['disagg_metrics']['ev_propensity'] = {
            'time': get_time_diff(t_before_ev_propensity, t_after_ev_propensity),
            'exit_status': {
                'exit_code': -1,
                'error_list': [error_str]
            }
        }
        # Change the pipeline exit code

        exit_status['error_code'] = -1
        exit_status['error_list'].append(error_str)

    return analytics_output_object, exit_status


def call_solar_propensity(analytics_input_object, analytics_output_object, exit_status, logger_pipeline):

    """
    Parameters:
        analytics_input_object (dict)              : Contains all inputs required to run the pipeline
        analytics_output_object(dict)              : Dictionary containing all outputs
        exit_status         (dict)              : Contains the error code and error list information for the pipeline
        logger_pipeline     (logger)            : The logger to use here
    Returns:
        analytics_output_object(dict)              : Dictionary containing all outputs
        exit_status            (dict)              : Contains the error code and error list information for the pipeline
    """

    t_before_solar = 0

    # noinspection PyBroadException
    try:
        t_before_solar = datetime.now()
        logger_pipeline.info('Calling solar propensity module for the user')
        analytics_input_object, analytics_output_object = solar_propensity_wrapper(analytics_input_object, analytics_output_object)
    except Exception:
        t_after_solar = datetime.now()

        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger_pipeline.error('Something went wrong in solar | %s', error_str)

        # Set the default exit status for the appliance

        analytics_output_object['disagg_metrics']['solar'] = {
            'time': get_time_diff(t_before_solar, t_after_solar),
            'exit_status': {
                'exit_code': -1,
                'error_list': [error_str]
            }
        }

        # Change the pipeline exit code

        exit_status['error_code'] = -1
        exit_status['error_list'].append(error_str)

    return analytics_output_object, exit_status


def call_user_profile_modules(analytics_input_object, analytics_output_object, exit_status, logger_pipeline, user_profile_module_seq):
    """
    Parameters:
        analytics_input_object     (dict)              : Contains all inputs required to run the pipeline
        analytics_output_object    (dict)              : Dictionary containing all outputs
        logger_pipeline            (logger)            : The logger to use for the pipeline
        exit_status                (dict)              : Contains the exit code and errors of the run so far
        user_profile_module_seq    (list)              : Sequence to execute the modules in

    Returns:
        analytics_input_object     (dict)              : Contains all the inputs required to run the pipeline
        analytics_output_object    (dict)              : Contains all the outputs obtained from the user profile module
        exit_status                (dict)              : Contains the exit code and errors of the run so far
    """

    for module_code in user_profile_module_seq:
        if module_code == 'life':
            analytics_output_object, exit_status = \
                call_lifestyle(analytics_input_object, analytics_output_object, exit_status, logger_pipeline)
        elif module_code == 'hvac_ineff':
            analytics_input_object, analytics_output_object, exit_status =\
                call_hvac_inefficiency(analytics_input_object, analytics_output_object, exit_status, logger_pipeline)
        elif module_code == "ev_propensity":
            analytics_output_object, exit_status = \
                call_ev_propensity(analytics_input_object, analytics_output_object, exit_status, logger_pipeline)
        elif module_code == 'so_propensity':
            analytics_output_object, exit_status = \
                call_solar_propensity(analytics_input_object, analytics_output_object, exit_status,
                                      logger_pipeline)
        else:
            logger_pipeline.warning('Unrecognized user profile module code %s |', module_code)

    return analytics_input_object, analytics_output_object, exit_status


def init_analytics_output_object(analytics_input_object, pipeline_output_object):
    """
    Initialises all the analytics custom input keys from the pipeline input object
    Parameters:
        analytics_input_object            (dict)           : Contains all the analytic input keys
        pipeline_output_object            (dict)           : Contains all the pipeline output keys

    Returns:
        analytics_output_object            (dict)          : Contains all the output required for disagg
    """

    # First initialise all the common keys

    analytics_output_object = prepare_analytics_output_object(analytics_input_object, pipeline_output_object)

    return analytics_output_object


def init_analytics_input_object(pipeline_input_object):
    """
    Initialises all the analytics custom input keys from the pipeline input object
    Parameters:
        pipeline_input_object          (dict)           : Contains all the input required for the pipeline

    Returns:
        analytics_input_object             (dict)          : Contains all the input required for disagg
    """

    # First initialise all the common keys

    temp_object = deepcopy(pipeline_input_object)

    analytics_input_object = {
        'appliances_hsm': temp_object.get('appliances_hsm'),
        'app_profile': temp_object.get('app_profile'),
        'data_quality_metrics': temp_object.get('data_quality_metrics'),
        'gb_pipeline_event': temp_object.get('gb_pipeline_event'),
        'home_meta_data': temp_object.get('home_meta_data'),
        'input_data': temp_object.get('input_data'),
        'input_data_with_neg_and_nan': temp_object.get('input_data_with_neg_and_nan'),
        'loaded_files': temp_object.get('loaded_files'),
        'logging_dict': temp_object.get('logging_dict'),
        'original_input_data': temp_object.get('original_input_data'),
        'out_bill_cycles': temp_object.get('out_bill_cycles'),
        'out_bill_cycles_by_module': temp_object.get('out_bill_cycles_by_module'),
        'all_hsms_appliance': temp_object.get('all_hsms_appliance'),
    }

    # Initialise disagg config from global config

    analytics_input_object['config'] = init_analytics_config(temp_object)

    return analytics_input_object


def run_analytics_pipeline_aer(pipeline_input_object, pipeline_output_object):
    """
    Contains all the operations related to the User profile analysis
    Parameters:
        pipeline_input_object           (dict)          : Contains all the input required for the pipeline
        pipeline_output_object          (dict)          : Contains all the output object

    Returns:
        pipeline_input_object           (dict)          : Contains all the input required for the pipeline
        pipeline_output_object          (dict)          : Contains all the output obtained in this pipeline
    """

    # Initialize pipeline logger

    logger = pipeline_input_object.get('logger').getChild('run_analytics_pipeline')
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

    analytics_input_object = init_analytics_input_object(pipeline_input_object)
    analytics_output_object = init_analytics_output_object(analytics_input_object, pipeline_output_object)
    analytics_input_object['logger'] = logger
    pipeline_input_object['logger'] = logger
    t_init_end = datetime.now()

    logger_base.info('Analytics module initialisation took | %.3fs ', get_time_diff(t_init_start, t_init_end))

    # ----------------------------------------------- PRE DISAGG OPS --------------------------------------------------

    t_pre_disagg_ops_start = datetime.now()
    analytics_input_object = pre_analytics_ops(analytics_input_object)
    t_pre_disagg_ops_end = datetime.now()

    logger_base.info('Analytics Pre disagg operations took | %.3fs ', get_time_diff(t_pre_disagg_ops_start,
                                                                                    t_pre_disagg_ops_end))

    # ----------------------------------------------- RUN APPLIANCE MODULE --------------------------------------------

    # Extract the module sequence to be run

    module_seq = analytics_input_object.get('config').get('module_seq')

    t_pipeline_start = datetime.now()
    analytics_input_object, analytics_output_object, exit_status = \
        call_user_profile_modules(analytics_input_object, analytics_output_object, exit_status, logger_base, module_seq)
    t_pipeline_end = datetime.now()

    logger_base.info('Analytics modules took | %.3fs ', get_time_diff(t_pipeline_start, t_pipeline_end))

    exit_status['runtime'] = get_time_diff(t_pipeline_start, t_pipeline_end)
    analytics_output_object['disagg_metrics']['analytics_pipeline'] = exit_status

    # ----------------------------------------------- PREPARE RESULTS -------------------------------------------------

    non_mtd_mode = pipeline_input_object.get('global_config').get('disagg_mode') in ['historical', 'incremental']

    if (non_mtd_mode) and (pipeline_output_object.get('created_hsm') is  not None) and \
            (pipeline_output_object.get('created_hsm').get('li') is not None) and \
            (pipeline_output_object.get('created_hsm').get('li').get('attributes') is  not None):
        pipeline_output_object.get('created_hsm').get('li').get('attributes').update(analytics_output_object.get('created_hsm').get('li').get('attributes'))

    t_start = datetime.now()
    api_analytics_output = prepare_analytics_results(analytics_input_object, analytics_output_object)
    t_end = datetime.now()

    logger_base.info('Analytics modules preparing results took | %.3fs ', get_time_diff(t_start, t_end))

    # ----------------------------------------------- END OF USER PROFILE MODULES -------------------------------------

    # Append all the necessary outputs to the pipeline_output_object

    pipeline_output_object['analytics_output_object'] = analytics_output_object
    pipeline_output_object['api_output']['lifestyleProfile'] = api_analytics_output.get('lifestyleProfile')
    pipeline_output_object['api_output']['applianceProfile'] = api_analytics_output.get('applianceProfile')
    pipeline_output_object['disagg_output_object']['appliance_profile'] = analytics_output_object['appliance_profile']
    pipeline_output_object['exit_status'] = exit_status

    return pipeline_input_object, pipeline_output_object
