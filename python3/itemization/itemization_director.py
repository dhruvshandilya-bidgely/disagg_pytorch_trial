"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Master file for itemization pipeline
"""
# Import python packages

import os
import copy
import pickle
import logging
import traceback
import numpy as np
from copy import deepcopy
from datetime import datetime
from numpy.random import RandomState

# import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.itemization.init_itemization_config import random_gen_config
from python3.itemization.init_itemization_config import init_itemization_params

from python3.utils.time.get_time_diff import get_time_diff
from python3.utils.get_rej_error_code_list import get_rej_error_code_list
from python3.itemization.init_itemization_config import init_itemization_config
from python3.itemization.initialisations.prepare_item_objects import prepare_item_object
from python3.itemization.initialisations.init_item_output_object import init_item_aer_output_object
from python3.itemization.pre_itemization_ops.aer_pre_itemization_ops import aer_pre_itemization_ops
from python3.itemization.prepare_results.prepare_aer_results import prepare_itemization_aer_results
from python3.itemization.prepare_results.update_ts_and_bc_level_results_with_hybrid_v2_output import update_results

from python3.itemization.wrappers.ref_hybrid_wrapper import ref_hybrid_wrapper
from python3.itemization.wrappers.cooking_hybrid_wrapper import cooking_hybrid_wrapper
from python3.itemization.wrappers.laundry_hybrid_wrapper import laundry_hybrid_wrapper
from python3.itemization.wrappers.lighting_hybrid_wrapper import lighting_hybrid_wrapper
from python3.itemization.wrappers.waterheater_hybrid_wrapper import waterheater_hybrid_wrapper
from python3.itemization.wrappers.entertainment_hybrid_wrapper import entertainment_hybrid_wrapper

from python3.itemization.aer.behavioural_analysis.home_profile.get_energy_profile import get_energy_profile
from python3.itemization.aer.behavioural_analysis.home_profile.get_occupancy_profile import get_occupancy_profile
from python3.itemization.aer.behavioural_analysis.home_profile.get_weekday_weekend_profile import weekend_analysis

from python3.itemization.aer.behavioural_analysis.clean_day_score.get_clean_days import get_day_cleanliness_score
from python3.itemization.aer.behavioural_analysis.activity_profile.get_activity_profile import get_activity_profile
from python3.itemization.aer.behavioural_analysis.activity_profile.get_profile_attributes import get_profile_attributes

from python3.itemization.aer.raw_energy_itemization.raw_energy_itemization_module import run_raw_energy_itemization_modules
from python3.itemization.prepare_results.update_appliance_profile import update_appliance_profile_based_on_disagg_postprocessing


def run_behavioural_analysis_modules(item_input_object, item_output_object, error_code, logger_pass):

    """
    run modules behavioural analysis modules

    Parameters:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('run_behavioural_analysis_modules')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    run_itemization_pipeline = True
    run_hybrid_v2 = item_input_object.get('item_input_params').get('run_hybrid_v2_flag')

    # Mask clean days and calculate cleanliness score for each day

    try:

        item_input_object, item_output_object = get_day_cleanliness_score(item_input_object, item_output_object, logger_pass)

        logger.debug("Calculated cleanliness score for all days | ")

        # prepare activity curve for the user

        item_input_object, item_output_object = get_activity_profile(item_input_object, item_output_object, logger_pass)

        logger.debug("Calculated activity profile | ")

        if np.all(item_input_object.get("activity_curve") == 0) or np.all(np.diff(item_input_object.get("activity_curve")) == 0):

            logger.warning('Adding default activity curve since all values in activity curve are either same or zero')
            seed = RandomState(random_gen_config.seed_value)
            item_input_object.get("activity_curve")[:] =  seed.normal(0.5, 0.1, item_input_object.get("activity_curve").shape)

        # calculate user attributes using activity curve
        item_input_object, item_output_object = get_profile_attributes(item_input_object, item_output_object, logger_pass)
        logger.debug("Calculated activity profile attributes | ")

        # calculate energy profile
        item_input_object, item_output_object = get_energy_profile(item_input_object, item_output_object, logger_pass)
        logger.debug("Calculated energy profile | ")

        if run_hybrid_v2:

            # calculate occupancy profile
            item_input_object, item_output_object = get_occupancy_profile(item_input_object, item_output_object, logger_pass)
            logger.debug("Calculated occupancy profile | ")

            # calculate weekend weekend difference profile

            item_input_object, item_output_object = weekend_analysis(item_input_object, item_output_object, logger_pass)
            logger.debug("Calculated weekday/weekend difference profile | ")

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in Behavioural analysis, not running itemization pipeline | %s', error_str)

        run_itemization_pipeline = False

        return item_input_object, item_output_object, run_itemization_pipeline, -1

    return item_input_object, item_output_object, run_itemization_pipeline, error_code


def call_waterheater(item_input_object, item_output_object, pipeline_output_object, exit_status, logger):

    """
    Call water heater module

    Parameters:
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        pipeline_output_object      (dict)      : Dict containing all disagg outputs
        exit_status                 (dict)      : Contains the exit code and errors of the run so far
        logger                      (logger)    : logger object

    Returns:
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
    """

    # Call water heater hybrid wrapper

    # noinspection PyBroadException
    try:
        t_before_wh = datetime.now()
        item_input_object, item_output_object, pipeline_output_object = \
            waterheater_hybrid_wrapper(item_input_object, item_output_object, pipeline_output_object)

    # General exception
    except Exception:

        t_after_wh = datetime.now()
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in hybrid water heater | %s', error_str)

        item_output_object['disagg_metrics'] = {
            'wh': {
                'time': get_time_diff(t_before_wh, t_after_wh),
                'exit_status': {
                    'exit_code': -1,
                    'error_list': [error_str]
            }}}

        # Change the pipeline exit code
        exit_status['error_list'].append(error_str)

    return item_input_object, item_output_object, pipeline_output_object, exit_status


def call_lighting(item_input_object, item_output_object, disagg_output_object, exit_status, logger):

    """
    Call lighting module

    Parameters:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
        disagg_output_object        (dict)      : Dict containing all disagg outputs
        logger                      (logger)    : logger object

    Returns:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
    """

    # Call lighting hybrid wrapper

    t_before = datetime.now()

    # noinspection PyBroadException
    try:
        item_input_object, item_output_object, disagg_output_object = \
            lighting_hybrid_wrapper(item_input_object, item_output_object, disagg_output_object)

    except Exception:

        ts_list = item_input_object.get("item_input_params").get("ts_list")
        samples_per_hour = int(item_input_object.get("item_input_params").get("samples_per_hour"))
        t_after = datetime.now()

        results = {
            'sleep_hours': np.zeros(int(samples_per_hour*Cgbdisagg.HRS_IN_DAY)),
            'lighting_capacity': -1
        }

        hsm_in = {
            'timestamp':  ts_list[-1],
            'attributes': results,
        }

        item_output_object['created_hsm']['li'] = hsm_in

        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in hybrid lighting | %s', error_str)

        item_output_object['disagg_metrics']['li'] = {
            'time': get_time_diff(t_before, t_after),
            'exit_status': -1
        }

    if (not item_input_object.get('item_input_params').get('run_hybrid_v2_flag')) or exit_status == -1:
        return item_input_object, item_output_object, disagg_output_object

    created_hsm = dict({
        'item_weekend_delta': np.nan_to_num(item_output_object['energy_profile']['weekend_energy_profile']),
        'item_weekday_delta': np.nan_to_num(item_output_object['energy_profile']['weekday_energy_profile']),
        'item_occ_count': item_output_object['occupants_profile']['occupants_count'],
        'item_occ_prob': item_output_object['occupants_profile']['occupants_prob'],
        'item_occ_prof': item_output_object['occupants_profile']['occupants_features'],
        'item_lunch': item_output_object['occupants_profile']['user_attributes']['lunch_present']
    })

    hsm_posting_allowed = (item_input_object.get('config').get('disagg_mode') == 'historical') or \
                          (item_input_object.get('config').get('disagg_mode') == 'incremental' and
                           len(item_input_object.get('item_input_params').get('day_input_data')) >= 70)

    if hsm_posting_allowed and \
            (item_output_object.get('created_hsm').get('li') is not None) and \
            (item_output_object.get('created_hsm').get('li').get('attributes') is not None):
        item_output_object['created_hsm']['li']['attributes'].update(created_hsm)

    valid_life_hsm = item_input_object.get("item_input_params").get('valid_life_hsm')

    li_hsm_present = \
        valid_life_hsm and item_input_object.get("item_input_params").get('life_hsm') is not None and \
        item_input_object.get("item_input_params").get('life_hsm').get('item_lunch') is not None

    if not li_hsm_present:
        return item_input_object, item_output_object, disagg_output_object

    life_hsm = item_input_object.get("item_input_params").get('life_hsm').get('item_lunch')
    life_hsm2 = item_input_object.get("item_input_params").get('life_hsm')

    if (life_hsm is not None) and isinstance(life_hsm, list):
        item_lunch = life_hsm2.get('item_lunch')[0]
        item_occ_prof = np.array(life_hsm2.get('item_occ_prof'))
        item_occ_prob = np.array(life_hsm2.get('item_occ_prob'))
        item_occ_count = life_hsm2.get('item_occ_count')[0]
        item_weekend_delta = np.array(life_hsm2.get('item_weekend_delta')).round(1)
        item_weekday_delta = np.array(life_hsm2.get('item_weekday_delta')).round(1)

        if item_lunch is not None and item_weekday_delta is not None:
            item_output_object['energy_profile']['weekend_energy_profile'] = item_weekend_delta
            item_output_object['energy_profile']['weekday_energy_profile'] = item_weekday_delta
            item_output_object['occupants_profile']['occupants_count'] = item_occ_count
            item_output_object['occupants_profile']['occupants_features'] = item_occ_prof
            item_output_object['occupants_profile']['occupants_prob'] = item_occ_prob
            item_output_object['occupants_profile']['user_attributes']['lunch_present'] = item_lunch

    elif (life_hsm is not None):
        life_hsm = item_input_object.get("item_input_params").get('life_hsm')
        item_lunch = life_hsm.get('item_lunch')
        item_occ_prof = np.array(life_hsm2.get('item_occ_prof'))
        item_occ_prob = np.array(life_hsm2.get('item_occ_prob'))
        item_occ_count = life_hsm2.get('item_occ_count')
        item_weekend_delta = np.array(life_hsm2.get('item_weekend_delta')).round(1)
        item_weekday_delta = np.array(life_hsm2.get('item_weekday_delta')).round(1)

        if item_lunch is not None and item_weekday_delta is not None:
            item_output_object['energy_profile']['weekend_energy_profile'] = item_weekend_delta
            item_output_object['energy_profile']['weekday_energy_profile'] = item_weekday_delta
            item_output_object['occupants_profile']['occupants_count'] = item_occ_count
            item_output_object['occupants_profile']['occupants_features'] = item_occ_prof
            item_output_object['occupants_profile']['occupants_prob'] = item_occ_prob
            item_output_object['occupants_profile']['user_attributes']['lunch_present'] = item_lunch

    return item_input_object, item_output_object, disagg_output_object


def call_ref(item_input_object, item_output_object, disagg_output_object, logger):

    """
    Call ref module

    Parameters:
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        disagg_output_object        (dict)      : Dict containing all disagg outputs
        logger                      (logger)    : logger object

    Returns:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
        disagg_output_object        (dict)      : Dict containing all disagg outputs

    """
    # noinspection PyBroadException
    try:
        item_input_object, item_output_object, disagg_output_object = \
            ref_hybrid_wrapper(item_input_object, item_output_object, disagg_output_object)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in hybrid ref | %s', error_str)

    return item_input_object, item_output_object, disagg_output_object


def call_cooking(item_input_object, item_output_object, disagg_output_object, logger):

    """
    Call cooking module

    Parameters:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
        disagg_output_object        (dict)      : Dict containing all disagg outputs
        logger                      (logger)    : logger object

    Returns:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
    """
    # Call cooking hybrid wrapper

    t_before = datetime.now()

    # noinspection PyBroadException
    try:

        item_input_object, item_output_object, disagg_output_object = \
            cooking_hybrid_wrapper(item_input_object, item_output_object, disagg_output_object)

    except Exception:
        t_after = datetime.now()
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in hybrid cooking | %s', error_str)

        item_output_object['disagg_metrics']['cook'] = {
            'time': get_time_diff(t_before, t_after),
            'exit_status': -1
        }

    return item_input_object, item_output_object, disagg_output_object


def call_entertainment(item_input_object, item_output_object, disagg_output_object, logger):

    """
    Call lighting module

    Parameters:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
        disagg_output_object      (dict)      : Dict containing all disagg outputs
        logger                    (logger)    : logger object

    Returns:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
        disagg_output_object      (dict)      : Dict containing all disagg outputs
    """
    # noinspection PyBroadException

    t_before = datetime.now()

    try:
        item_input_object, item_output_object, disagg_output_object = \
            entertainment_hybrid_wrapper(item_input_object, item_output_object, disagg_output_object)

    except Exception:
        t_after = datetime.now()
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in hybrid entertainment | %s', error_str)

        item_output_object['disagg_metrics']['ent'] = {
            'time': get_time_diff(t_before, t_after),
            'exit_status': -1
        }

    return item_input_object, item_output_object, disagg_output_object


def call_laundry(item_input_object, item_output_object, disagg_output_object, logger):

    """
    Call laundry module

    Parameters:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
        disagg_output_object        (dict)      : Dict containing all disagg outputs
        logger                      (logger)    : logger object

    Returns:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
        disagg_output_object        (dict)      : Dict containing all disagg outputs

    """

    t_before = datetime.now()

    # noinspection PyBroadException
    try:
        item_input_object, item_output_object, disagg_output_object = \
            laundry_hybrid_wrapper(item_input_object, item_output_object, disagg_output_object)

    except Exception:
        t_after = datetime.now()
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in hybrid laundry | %s', error_str)

        item_output_object['disagg_metrics']['ld'] = {
            'time': get_time_diff(t_before, t_after),
            'exit_status': -1
        }

    return item_input_object, item_output_object, disagg_output_object


def call_itemization_modules(item_input_object, item_output_object, pipeline_output_object, exit_status, error_code, logger_pass):

    """
    Call different hybrid modules

    Parameters:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
        pipeline_output_object      (dict)      : Dict containing all disagg outputs
        exit_status                 (dict)      : Contains the exit code and errors of the run so far
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('call_itemization_modules')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    # Extract the module sequence to be run

    item_mod_seq = copy.deepcopy(item_input_object.get('config').get('itemization_module_seq'))
    run_hybrid_v2 = item_input_object.get('item_input_params').get('run_hybrid_v2_flag')

    if (not run_hybrid_v2) or (error_code == -1):
        item_mod_seq = list(set(item_mod_seq).difference(['cook', 'ent', 'ld', 'ref']))

    logger.info('Itemization module seq %s |', item_mod_seq)

    # Run all the appliance in the item_mod_seq

    for module_code in item_mod_seq:

        if module_code == 'wh':
            item_input_object, item_output_object, pipeline_output_object, exit_status = \
                call_waterheater(item_input_object, item_output_object, pipeline_output_object, exit_status, logger)

        elif module_code == 'li':
            item_input_object, item_output_object, disagg_output_object = \
                call_lighting(item_input_object, item_output_object, pipeline_output_object, error_code, logger)

        elif module_code == 'ref':
            item_input_object, item_output_object, disagg_output_object = \
                call_ref(item_input_object, item_output_object, pipeline_output_object, logger)

        elif module_code == 'cook':
            item_input_object, item_output_object, disagg_output_object = \
                call_cooking(item_input_object, item_output_object, pipeline_output_object, logger)

        elif module_code == 'ent':
            item_input_object, item_output_object, disagg_output_object = \
                call_entertainment(item_input_object, item_output_object, pipeline_output_object, logger)

        elif module_code == 'ld':
            item_input_object, item_output_object, disagg_output_object = \
                call_laundry(item_input_object, item_output_object, pipeline_output_object, logger)

        else:
            logger.warning('Unrecognized module code %s |', module_code)

    return item_input_object, item_output_object, pipeline_output_object


def init_item_input_object(pipeline_input_object, pipeline_output_object):

    """
    This function is used to initialise the itemization input object
    Parameters:
        pipeline_input_object           (dict)          : Contains all the input required for the pipeline
        pipeline_output_object          (dict)          : Contains all the output object
    Returns:
        item_input_object               (dict)          : Itemization input object
    """

    # First initialise all the common keys

    temp_object = deepcopy(pipeline_input_object)

    item_input_object = {
        'appliances_hsm': temp_object.get('appliances_hsm'),
        'app_profile': temp_object.get('app_profile'),
        'data_quality_metrics': temp_object.get('data_quality_metrics'),
        'gb_pipeline_event': temp_object.get('gb_pipeline_event'),
        'home_meta_data': temp_object.get('home_meta_data'),
        'input_data': temp_object.get('input_data'),
        'global_config': temp_object.get('global_config'),
        'logging_dict': temp_object.get('logging_dict'),
        'original_input_data': temp_object.get('original_input_data'),
        'input_data_without_outlier_removal': temp_object.get('input_data_without_outlier_removal'),
        'out_bill_cycles': temp_object.get('out_bill_cycles'),
        'loaded_files': temp_object.get('loaded_files'),
        'index': temp_object.get('index'),
        'out_bill_cycles_by_module': temp_object.get('out_bill_cycles_by_module'),
        'input_data_with_neg_and_nan': temp_object.get('input_data_with_neg_and_nan'),
    }

    if pipeline_output_object.get("disagg_output_object") is not None:
        item_input_object['ao_seasonality'] = copy.deepcopy(pipeline_output_object.get("disagg_output_object").get("ao_seasonality"))

    # Initialise the configurations -
    item_input_object = init_itemization_config(temp_object, pipeline_output_object, item_input_object)

    return item_input_object


def init_item_output_object(pipeline_input_object, pipeline_output_object, item_input_object):

    """
    This function is used to create the itemisation output object
    Parameters:
        pipeline_input_object           (dict)          : Contains all the input required for the pipeline
        pipeline_output_object          (dict)          : Contains all the output object
        item_input_object               (dict)          : Itemization input object
    Returns:
        item_input_object               (dict)          : Itemization input object
    """

    item_output_object = init_item_aer_output_object(pipeline_input_object, pipeline_output_object, item_input_object)

    return item_output_object


def run_itemization_pipeline(pipeline_input_object, pipeline_output_object):

    """
    Contains all the operations related to the Itemization pipeline
    Parameters:
        pipeline_input_object           (dict)          : Contains all the input required for the pipeline
        pipeline_output_object          (dict)          : Contains all the output object

    Returns:
        pipeline_input_object           (dict)          : Contains all the input required for the pipeline
        pipeline_output_object          (dict)          : Contains all the output obtained in this pipeline
    """

    # Initialize pipeline logger

    logger_itemization_base = pipeline_input_object.get('logger').getChild('run_itemization_pipeline')
    pipeline_input_object.pop('logger')
    logger_itemization = logging.LoggerAdapter(logger_itemization_base, pipeline_input_object.get('logging_dict'))
    logger_itemization_pass = {
        'logger_base': logger_itemization_base,
        'logging_dict': pipeline_input_object.get('logging_dict'),
    }

    # Initialise required variables

    exit_status = {
        'exit_code': 1,
        'error_codes': [],
        'runtime': None,
        'error_list': [],
    }

    # ----------------------------------------------- INITIALISATION --------------------------------------------------

    # Preparing Itemization input and output object
    t_init_start = datetime.now()

    item_input_object = init_item_input_object(pipeline_input_object, pipeline_output_object)

    item_output_object = init_item_output_object(pipeline_input_object, pipeline_output_object, item_input_object)

    item_input_object['logger'] = logger_itemization_base
    pipeline_input_object['logger'] = logger_itemization_base

    t_init_end = datetime.now()

    logger_itemization.info('Itemization AER initialisation took | %.3fs ', get_time_diff(t_init_start, t_init_end))

    # ----------------------------------------------- PREPARE ITEMIZATION DATA ----------------------------------------

    t_init_start = datetime.now()

    # ----------------------------------------------- PRE ITEMIZATION OPS -----------------------------------------

    t_pre_itemization_ops_start = datetime.now()

    item_input_object = aer_pre_itemization_ops(item_input_object)
    t_pre_itemization_ops_end = datetime.now()

    logger_itemization.info('AER Pre Itemization operations took | %.3fs ', get_time_diff(t_pre_itemization_ops_start, t_pre_itemization_ops_end))

    item_input_object, item_output_object, faulty_input_data, error_code = prepare_item_object(item_input_object, item_output_object, logger_itemization_pass)
    t_init_end = datetime.now()

    logger_itemization.info('Itemization Data Preparation took | %.3fs ', get_time_diff(t_init_start, t_init_end))

    # ------------------------------------------------ RUN ITEMIZATION CHECKS -----------------------------------------

    data_quality_dict = pipeline_input_object.get('data_quality_metrics').get('disagg_data_quality')
    run_pipeline = data_quality_dict.get('run_pipeline')

    itemization_data_quality_dict = item_output_object.get('itemization_metrics').get('itemization_pipeline')
    run_itemization = itemization_data_quality_dict.get('exit_status').get('itemization_pipeline_status') and not np.bool(faulty_input_data)

    # ----------------------------------------------- RUN APPLIANCE MODULE --------------------------------------------

    t_before_pipeline = datetime.now()

    run_successful = 0

    run_disagg_pipeline = pipeline_input_object.get('data_quality_metrics').get('disagg_data_quality').get('run_pipeline')

    if run_itemization and run_disagg_pipeline and (error_code != -1):

        # ----------------------------------------------- RUN APPLIANCE MODULE ----------------------------------------

        # Call individual appliance hybrid module

        t_itemization_start = datetime.now()
        t0 = datetime.now()

        item_input_object, item_output_object, run_itemization_pipeline, error_code = \
            run_behavioural_analysis_modules(item_input_object, item_output_object, error_code, logger_itemization_pass)
        t1 = datetime.now()

        item_input_object, item_output_object, pipeline_output_object = \
            call_itemization_modules(item_input_object, item_output_object, pipeline_output_object, exit_status, error_code, logger_itemization_pass)
        t2 = datetime.now()

        item_output_object, error_code = run_final_itemization(item_input_object, item_output_object, logger_itemization_pass, error_code, logger_itemization)
        t3 = datetime.now()

        t_itemization_end = datetime.now()

        logger_itemization.info('Running of behavior modules took | %.3f s ', get_time_diff(t0, t1))
        logger_itemization.info('Running of appliance modules took | %.3f s ', get_time_diff(t1, t2))
        logger_itemization.info('Running of itemization modules took | %.3f s ', get_time_diff(t2, t3))
        logger_itemization.info('Running of Itemization pipeline took | %.3f s ', get_time_diff(t_itemization_start, t_itemization_end))

    elif error_code == -1:
        logger_itemization.info('Not running Itemization pipeline due to error in hybrid v2 data prep module |')
        error_code = -1
        exit_status['exit_code'] = error_code

    else:
        logger_itemization.info('Not running Itemization pipeline due to bad data quality |')
        error_code = -2

        exit_status['exit_code'] = -2
        if not run_pipeline:
            exit_status['error_codes'] = get_rej_error_code_list(data_quality_dict.get('rejection_reasons'))
        else:
            exit_status['error_codes'] = get_rej_error_code_list(itemization_data_quality_dict.get('exit_status').get('error_list'))

    pipeline_input_object["item_input_object"] = item_input_object
    pipeline_output_object["item_output_object"] = item_output_object

    if error_code == -4:
        logger_itemization.info('Not posting Itemization results due to missing hybrid v2 file |')
        exit_status['exit_code'] = error_code
        pipeline_output_object['api_output']['gbOutputStatus']['exitCode'] = -4

    elif error_code == -5:
        logger_itemization.info('Not posting Itemization results due to improper hybrid v2 file |')
        exit_status['exit_code'] = error_code
        pipeline_output_object['api_output']['gbOutputStatus']['exitCode'] = -5

    elif error_code == -2:
        logger_itemization.info('Not posting Itemization results due to data quality issue |')

    elif error_code == -1:
        logger_itemization.info('Not posting Itemization results due to unexpected error in hybrid v2 module | ')
        exit_status['exit_code'] = error_code * (item_input_object.get('global_config').get('enable_hybrid_v2') > 0) + \
                                   0 * (item_input_object.get('global_config').get('enable_hybrid_v2') == 0)
        pipeline_output_object['api_output']['gbOutputStatus']['exitCode'] = exit_status['exit_code']

    else:
        run_successful = 1

        item_output_object, item_output_map, ts_est_aft_item, bc_est_aft_item = \
            prepare_debug_object(pipeline_input_object, item_input_object, item_output_object, logger_itemization)

        item_output_object = \
            update_results(pipeline_input_object, item_input_object, item_output_object, item_output_map,
                           ts_est_aft_item, bc_est_aft_item, logger_itemization, run_successful)

        item_output_object = update_appliance_profile_based_on_disagg_postprocessing(item_input_object, item_output_object, run_successful, logger_itemization)

    # ----------------------------------------------- PREPARE RESULTS -------------------------------------------------

    t_after_pipeline = datetime.now()
    pipeline_runtime = get_time_diff(t_before_pipeline, t_after_pipeline)
    logger_itemization.info('AER Itemization Pipeline ran in | %.3f s', pipeline_runtime)

    exit_status['runtime'] = pipeline_runtime
    item_output_object['itemization_metrics']['aer_pipeline'] = exit_status

    t_start = datetime.now()
    api_aer_item_output, item_output_object = prepare_itemization_aer_results(item_input_object, item_output_object, pipeline_output_object, run_successful)
    t_end = datetime.now()

    logger_itemization.info('Itemization AER preparing results took | %.3fs ', get_time_diff(t_start, t_end))

    if item_input_object.get('global_config').get('enable_hybrid_v2') and error_code == -1:
        api_aer_item_output['gbMonthlyOutput'] = []
        api_aer_item_output['gbTBOutput'] = []

    pipeline_output_object['api_output'] = api_aer_item_output

    # ----------------------------------------------- END OF AER ITEMIZATION -----------------------------------------

    # Combine the api_aer_disagg_output in pipeline_output_object

    pipeline_output_object['item_output_object'] = item_output_object

    if 'item_input_object' in pipeline_input_object.keys():
        pipeline_input_object.pop('item_input_object')

    # Update the created hsm key in the pipeline output object if it is Historical or Incremental mode

    if pipeline_input_object.get('global_config').get('disagg_mode') == 'historical' or \
            pipeline_input_object.get('global_config').get('disagg_mode') == 'incremental':
        pipeline_output_object.update({
            "created_hsm": item_output_object.get('created_hsm')
        })

    return pipeline_input_object, pipeline_output_object


def run_final_itemization(item_input_object, item_output_object, logger_pass, error_code, logger_itemization):

    """
    Perform 100% itemization

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
    """

    try:
        item_input_object, item_output_object = \
            run_raw_energy_itemization_modules(item_input_object, item_output_object, logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger_itemization.error('Something went wrong while running hybrid v2 module | %s', error_str)
        return item_output_object, -1

    return item_output_object, error_code


def prepare_debug_object(pipeline_input_object, item_input_object, item_output_object, logger_itemization):


    """
    prepare debug object for NDQ

    Parameters:
        pipeline_input_object     (dict)      : master Dict containing all inputs
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger_itemization        (dict)      : logger object

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
        item_output_map           (dict)      : app index map
        ts_est_aft_item           (np.ndarray): ts level itemization results
        bc_est_aft_item           (np.ndarray): ts level itemization results
    """

    try:

        final_debug = dict()

        # fetching output of itemization pipeline

        bc_est_aft_disagg = item_input_object.get("disagg_bill_cycle_estimate")
        ts_est_aft_disagg = item_input_object.get("disagg_epoch_estimate")

        bc_est_aft_bh_disagg = copy.deepcopy(item_output_object.get("bill_cycle_estimate"))
        ts_est_aft_bh_disagg = copy.deepcopy(item_output_object.get("epoch_estimate"))

        disagg_output_map = copy.deepcopy(item_input_object.get("disagg_output_write_idx_map"))
        item_output_map = copy.deepcopy(item_output_object.get("output_write_idx_map"))
        final_output = item_output_object.get("final_itemization").get("tou_itemization")

        bc_list = np.max(item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX], axis=1)
        ts_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_EPOCH_IDX]

        ts_input_data = np.reshape(item_input_object.get("item_input_params").get("original_input_data"),
                                   (np.size(final_output[1]), 1))
        ts_temp_data = np.reshape(
            item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_TEMPERATURE_IDX],
            (np.size(final_output[1]), 1))
        bc_list1 = np.reshape(
            item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX],
            (np.size(final_output[1])))

        ts_est_aft_item = np.zeros((np.size(final_output[0]), ts_est_aft_bh_disagg.shape[1]))
        bc_est_aft_item = np.zeros_like(bc_est_aft_bh_disagg)

        disagg_input_data = item_input_object.get("input_data")[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

        unique_bcs = np.unique(bc_list)

        # preparing billing cycle level and ts level output in the required format
        # that will be used for further posting of results
        # and preparing results dump for ndq testing

        bc_est_aft_item = np.hstack((bc_est_aft_item, np.zeros((len(bc_est_aft_item), 1))))
        bc_est_aft_disagg = np.hstack((bc_est_aft_disagg, np.zeros((len(bc_est_aft_disagg), 1))))
        bc_est_aft_bh_disagg = np.hstack((bc_est_aft_bh_disagg, np.zeros((len(bc_est_aft_bh_disagg), 1))))

        for i in range(len(unique_bcs)):

            bc_est_aft_item[bc_est_aft_disagg[:, 0] == unique_bcs[i], 1] = np.sum(
                final_output[1, bc_list == unique_bcs[i]])
            bc_est_aft_item[bc_est_aft_disagg[:, 0] == unique_bcs[i], 2] = np.sum(
                final_output[2, bc_list == unique_bcs[i]])
            bc_est_aft_item[bc_est_aft_disagg[:, 0] == unique_bcs[i], 3] = np.sum(
                final_output[3, bc_list == unique_bcs[i]])
            bc_est_aft_item[bc_est_aft_disagg[:, 0] == unique_bcs[i], 4] = np.sum(
                final_output[4, bc_list == unique_bcs[i]])
            bc_est_aft_item[bc_est_aft_disagg[:, 0] == unique_bcs[i], 5] = np.sum(
                final_output[5, bc_list == unique_bcs[i]])
            bc_est_aft_item[bc_est_aft_disagg[:, 0] == unique_bcs[i], 6] = np.sum(
                final_output[6, bc_list == unique_bcs[i]])
            bc_est_aft_item[bc_est_aft_disagg[:, 0] == unique_bcs[i], 7] = np.sum(
                final_output[7, bc_list == unique_bcs[i]])
            bc_est_aft_item[bc_est_aft_disagg[:, 0] == unique_bcs[i], 11] = np.sum(
                final_output[10, bc_list == unique_bcs[i]])
            bc_est_aft_item[bc_est_aft_disagg[:, 0] == unique_bcs[i], 12] = np.sum(
                final_output[11, bc_list == unique_bcs[i]])
            bc_est_aft_item[bc_est_aft_disagg[:, 0] == unique_bcs[i], 13] = np.sum(
                final_output[12, bc_list == unique_bcs[i]])
            bc_est_aft_item[bc_est_aft_disagg[:, 0] == unique_bcs[i], 14] = np.sum(
                final_output[13, bc_list == unique_bcs[i]])

            bc_est_aft_disagg[bc_est_aft_disagg[:, 0] == unique_bcs[i], 9] = np.sum(
                item_input_object.get('item_input_params').get('vac_v1')[bc_list == unique_bcs[i]])
            bc_est_aft_disagg[bc_est_aft_disagg[:, 0] == unique_bcs[i], 10] = np.sum(
                item_input_object.get('item_input_params').get('vac_v2')[bc_list == unique_bcs[i]])

            bc_est_aft_bh_disagg[bc_est_aft_bh_disagg[:, 0] == unique_bcs[i], -1] = \
                np.sum(disagg_input_data[item_input_object.get("input_data")[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == unique_bcs[i]])

            bc_est_aft_disagg[bc_est_aft_disagg[:, 0] == unique_bcs[i], -1] = \
                np.sum(disagg_input_data[item_input_object.get("input_data")[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == unique_bcs[i]])

            bc_est_aft_bh_disagg[bc_est_aft_item[:, 0] == unique_bcs[i], -1] = \
                np.sum(ts_input_data[bc_list1 == unique_bcs[i]])

        ts_est_aft_item[:, 1] = np.reshape(final_output[1], (np.size(final_output[1])))
        ts_est_aft_item[:, 2] = np.reshape(final_output[2], (np.size(final_output[2])))
        ts_est_aft_item[:, 3] = np.reshape(final_output[3], (np.size(final_output[3])))
        ts_est_aft_item[:, 4] = np.reshape(final_output[4], (np.size(final_output[4])))
        ts_est_aft_item[:, 5] = np.reshape(final_output[5], (np.size(final_output[5])))
        ts_est_aft_item[:, 6] = np.reshape(final_output[6], (np.size(final_output[6])))
        ts_est_aft_item[:, 7] = np.reshape(final_output[7], (np.size(final_output[1])))
        ts_est_aft_item[:, 11] = np.reshape(final_output[10], (np.size(final_output[1])))
        ts_est_aft_item[:, 12] = np.reshape(final_output[11], (np.size(final_output[1])))
        ts_est_aft_item[:, 13] = np.reshape(final_output[12], (np.size(final_output[1])))
        ts_est_aft_item[:, 14] = np.reshape(final_output[13], (np.size(final_output[1])))
        ts_est_aft_item[:, 0] = np.reshape(ts_list, (np.size(final_output[1])))

        ts_est_aft_item = np.hstack((ts_est_aft_item, ts_input_data))
        ts_est_aft_disagg = np.vstack(
            (ts_est_aft_disagg.T, item_input_object.get("input_data")[:, Cgbdisagg.INPUT_CONSUMPTION_IDX].T)).T
        ts_est_aft_bh_disagg = np.vstack(
            (ts_est_aft_bh_disagg.T, item_input_object.get("input_data")[:, Cgbdisagg.INPUT_CONSUMPTION_IDX].T)).T

        ts_est_aft_item = np.hstack((ts_est_aft_item, ts_temp_data))
        ts_est_aft_disagg = np.vstack(
            (ts_est_aft_disagg.T, item_input_object.get("input_data")[:, Cgbdisagg.INPUT_TEMPERATURE_IDX].T)).T
        ts_est_aft_bh_disagg = np.vstack(
            (ts_est_aft_bh_disagg.T, item_input_object.get("input_data")[:, Cgbdisagg.INPUT_TEMPERATURE_IDX].T)).T

        bc_est_aft_item = np.hstack((bc_est_aft_item, np.zeros((len(bc_est_aft_item), 1))))
        bc_est_aft_disagg = np.hstack((bc_est_aft_disagg, np.zeros((len(bc_est_aft_disagg), 1))))
        bc_est_aft_bh_disagg = np.hstack((bc_est_aft_bh_disagg, np.zeros((len(bc_est_aft_bh_disagg), 1))))

        # preparing residuaal data for results dumping

        bc_est_aft_item[:, 0] = bc_est_aft_bh_disagg[:, 0]

        ts_res = ts_est_aft_item[:, 15] - ts_est_aft_item[:, [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14]].sum(axis=1)
        ts_est_aft_item = np.hstack((ts_est_aft_item, ts_res[:, None]))

        bc_est_aft_item[:, 15] = bc_est_aft_bh_disagg[:, 15]
        bc_res = bc_est_aft_item[:, 15] - bc_est_aft_item[:, [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14]].sum(axis=1)
        bc_est_aft_item = np.hstack((bc_est_aft_item, bc_res[:, None]))

        disagg_output_map['total'] = 12
        item_output_map['total'] = 15
        disagg_output_map['temp'] = 13
        item_output_map['temp'] = 16

        item_output_map['others'] = 17

        disagg_output_map.pop('cook')
        disagg_output_map.pop('ent')
        disagg_output_map.pop('ld')

        res_aft_item = np.reshape(final_output[14], (np.size(final_output[1])))

        _, ts_disagg_idx, ts_item_idx = np.intersect1d(ts_est_aft_disagg[:, 0], ts_est_aft_item[:, 0],
                                                       return_indices=1)

        ts_est_aft_disagg = ts_est_aft_disagg[ts_disagg_idx]
        ts_est_aft_item = ts_est_aft_item[ts_item_idx]
        ts_est_aft_bh_disagg = ts_est_aft_bh_disagg[ts_disagg_idx]

        bc_est_aft_disagg[:, disagg_output_map.get('wh')] = np.maximum(
            np.nan_to_num(bc_est_aft_bh_disagg[:, item_output_map.get('wh')]),
            np.nan_to_num(bc_est_aft_disagg[:, disagg_output_map.get('wh')]))

        ts_est_aft_disagg[:, disagg_output_map.get('wh')] = np.maximum(
            np.nan_to_num(ts_est_aft_bh_disagg[:, item_output_map.get('wh')]),
            np.nan_to_num(ts_est_aft_disagg[:, disagg_output_map.get('wh')]))

        bill_cycle_list = np.fmin(item_input_object['input_data'][-1, 5], item_input_object.get('out_bill_cycles'))

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger_itemization.error('Something went wrong while preparing results | %s', error_str)

        return item_output_object, item_output_map, ts_est_aft_item, bc_est_aft_item

    if init_itemization_params().get('dump_results'):

        final_debug.update({
            "bc_est_aft_disagg": bc_est_aft_disagg,
            "bc_est_aft_bh_disagg": bc_est_aft_bh_disagg,
            "bc_est_aft_item": bc_est_aft_item,
            "ts_est_aft_disagg": ts_est_aft_disagg,
            "ts_est_aft_bh_disagg": ts_est_aft_bh_disagg,
            "ts_est_aft_item": ts_est_aft_item,
            "res_aft_item": res_aft_item,
            "map_aft_disagg": disagg_output_map,
            "map_aft_bh_disagg": item_output_map,
            "map_aft_item": item_output_map,
            "bill_cycle_timestamp": bill_cycle_list
        })

        logger_itemization.info('Successfully created dictionary to be used for non dev qa testing | ')

        uuid = item_input_object.get('config').get('uuid')
        disagg_mode = item_input_object.get('config').get('disagg_mode')
        t_start = pipeline_input_object.get('gb_pipeline_event').get('start')
        t_end = pipeline_input_object.get('gb_pipeline_event').get('end')

        path_name = init_itemization_params().get('results_folder')
        if not os.path.isdir(path_name + uuid):
            os.makedirs(path_name + uuid)

        path_name = path_name + uuid + "/" + uuid + "_" + disagg_mode + "_" + str(t_start) + "_" + str(
            t_end) + ".pkl"

        with open(path_name, 'wb') as f:
            pickle.dump(final_debug, f)
        f.close()

        logger_itemization.info('Successfully dumped the dictionary to be used for non dev qa testing | ')

    return item_output_object, item_output_map, ts_est_aft_item, bc_est_aft_item
